// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Matmul execution for the GPU backend: the tile loop, per-shape autotuning, and
//! on-device timing. Owns the autotune cache (one `Matmul` lives on the backend).
//! Uses the generic `autotune` helper + the generated kernels from `configs.zig`.
//!
//! v1 scope: rank-2 (non-batched) f32 matmul. Mirrors the CPU tile loop — for each
//! output tile (ti_m, ti_n), accumulate over the k-tiles, applying the original
//! beta only on the first k-tile (1.0 thereafter).

const std = @import("std");
const wgpu = @import("../wgpu.zig");
const pipelines = @import("../pipelines.zig");
const autotune = @import("../autotune.zig");
const context = @import("../context.zig");
const codegen = @import("../matmul/codegen.zig");
const configs = @import("../matmul/configs.zig");
const backend_mod = @import("../../backend.zig");
const tensor_store_mod = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const c = wgpu.c;
const Ctx = context.Ctx;
const Frame = @import("../frame.zig").Frame;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const TensorMeta = tensor_store_mod.TensorMeta;
const MatmulConfig = codegen.MatmulConfig;
const Generated = codegen.Generated;
const StepMatMul = executable.StepMatMulTiled;

/// Shared dequant module (also used by the MatMulNT executor). We use its
/// `q8_to_f32` entry to materialize a q8_0 B tile as f32 for the plain GEMM.
const dequant_kernel: pipelines.KernelDesc = .{ .name = "dequant", .wgsl = @embedFile("../kernels/dequant.wgsl") };
/// Fused q8_0 (K-major) matvec for M==1 (decode): folds the dequant into the dot
/// so B is read once, no f32 scratch. See kernels/matmul_gemv.wgsl.
const gemv_kernel: pipelines.KernelDesc = .{ .name = "matmul_gemv", .wgsl = @embedFile("../kernels/matmul_gemv.wgsl") };
/// Column-pairs per GEMV workgroup (matches `COLS` in matmul_gemv.wgsl).
const GEMV_COLS: u32 = 32;
const Q8_BLOCK_ELEMS: u32 = 32;
const Q8_BLOCK_BYTES: u32 = 34;
const DEQUANT_WG: u32 = 64;

/// Uniform params for the fused GEMV; field order matches `struct Params` in
/// matmul_gemv.wgsl: { k, n, _p0, _p1, alpha, beta }.
const GemvParams = extern struct { k: u32, n: u32, _p0: u32 = 0, _p1: u32 = 0, alpha: f32, beta: f32 };
/// Field order matches `struct Params` in dequant.wgsl.
const DequantParams = extern struct { n: u32, k: u32, src_wpr: u32, dst_row: u32, count: u32, _p0: u32 = 0, _p1: u32 = 0, _p2: u32 = 0 };

/// Uniform params for the tiled GEMM kernel; field order matches the WGSL
/// `struct Params { dims: vec4<u32>, strides: vec4<u32>, ab: vec4<f32> }`.
/// Shared with the MatMulNT executor (matmul_nt.zig), which drives the same GEMM
/// pipelines over a dequantized scratch B.
pub const MatMulParams = extern struct {
    m: u32,
    n: u32,
    k: u32,
    _d3: u32 = 0,
    a_row: u32,
    b_row: u32,
    c_row: u32,
    _s3: u32 = 0,
    alpha: f32,
    beta: f32,
    _a2: f32 = 0,
    _a3: f32 = 0,
};

fn syncDevice(ctx: Ctx) void {
    _ = c.wgpuDevicePoll(ctx.gpu.device, 1, null);
}

/// A config is eligible only when it fits this device's limits and, for vec4
/// loads, A/B row strides are 16-byte aligned. (Also used by nt.zig.)
pub fn eligibleConfig(cfg: MatmulConfig, limits: wgpu.Limits, a_row_bytes: isize, b_row_bytes: isize) bool {
    if (codegen.sharedBytes(cfg) > limits.max_shared_bytes) return false;
    if (cfg.threads() > limits.max_invocations) return false;
    if (cfg.threads() > limits.max_workgroup_size_x) return false;
    if (!cfg.vec4_load) return true;
    return @rem(a_row_bytes, 16) == 0 and @rem(b_row_bytes, 16) == 0;
}

fn tileBytes(meta: TensorMeta) ?u64 {
    var elems: u64 = 1;
    for (meta.tile_shape) |d| {
        const dim = std.math.cast(u64, d) orelse return null;
        elems = std.math.mul(u64, elems, dim) catch return null;
    }
    const bytes_per_elem = std.math.cast(u64, meta.dtype.info().block_bytes) orelse return null;
    return std.math.mul(u64, elems, bytes_per_elem) catch return null;
}

fn ensureTileBindingFits(ctx: Ctx, meta: TensorMeta) ExecuteProgramError!void {
    const bytes = tileBytes(meta) orelse return error.Unsupported;
    if (bytes > ctx.gpu.limits.max_storage_binding_bytes) return error.Unsupported;
}

fn fullStorageTiles(meta: TensorMeta) bool {
    if (meta.shape.len != meta.tile_shape.len) return false;
    for (meta.shape, meta.tile_shape) |shape, tile| {
        if (tile == 0 or shape % tile != 0) return false;
    }
    return true;
}

fn fullTileCompatible(cfg: MatmulConfig, c_meta: TensorMeta, a_meta: TensorMeta, b_meta: TensorMeta) bool {
    if (cfg.bounds_check) return true;
    if (c_meta.tile_shape.len != 2 or a_meta.tile_shape.len != 2 or b_meta.tile_shape.len != 2) return false;
    if (!fullStorageTiles(c_meta) or !fullStorageTiles(a_meta) or !fullStorageTiles(b_meta)) return false;
    return c_meta.tile_shape[0] % cfg.bm == 0 and
        c_meta.tile_shape[1] % cfg.bn == 0 and
        a_meta.tile_shape[0] % cfg.bm == 0 and
        a_meta.tile_shape[1] % cfg.bk == 0 and
        b_meta.tile_shape[0] % cfg.bk == 0 and
        b_meta.tile_shape[1] % cfg.bn == 0;
}

fn forcedConfigIndex(generated: []const Generated, limits: wgpu.Limits, a_row_bytes: isize, b_row_bytes: isize, c_meta: TensorMeta, a_meta: TensorMeta, b_meta: TensorMeta) ExecuteProgramError!?usize {
    const env = std.c.getenv("AION_MATMUL_CONFIG") orelse return null;
    const raw = std.mem.span(env);

    const maybe_idx = std.fmt.parseInt(usize, raw, 10) catch null;
    if (maybe_idx) |idx| {
        if (idx >= generated.len) return error.Unsupported;
        const cfg = generated[idx].cfg;
        if (!fullTileCompatible(cfg, c_meta, a_meta, b_meta) or !eligibleConfig(cfg, limits, a_row_bytes, b_row_bytes)) return error.Unsupported;
        return idx;
    }

    for (generated, 0..) |g, idx| {
        if (std.mem.eql(u8, raw, g.entry)) {
            if (!fullTileCompatible(g.cfg, c_meta, a_meta, b_meta) or !eligibleConfig(g.cfg, limits, a_row_bytes, b_row_bytes)) return error.Unsupported;
            return idx;
        }
    }
    return error.Unsupported;
}

test "matmul eligibility respects device limits" {
    const small: MatmulConfig = .{ .bm = 128, .bn = 128, .bk = 8, .tm = 8, .tn = 8, .vec4_load = true };
    const big: MatmulConfig = .{ .bm = 128, .bn = 128, .bk = 32, .tm = 8, .tn = 8, .vec4_load = true };

    try std.testing.expect(eligibleConfig(small, .{ .max_shared_bytes = 16 * 1024, .max_invocations = 256 }, 512, 512));
    try std.testing.expect(!eligibleConfig(big, .{ .max_shared_bytes = 16 * 1024, .max_invocations = 256 }, 512, 512));
    try std.testing.expect(eligibleConfig(big, .{ .max_shared_bytes = 48 * 1024, .max_invocations = 256 }, 512, 512));
    try std.testing.expect(!eligibleConfig(small, .{ .max_shared_bytes = 16 * 1024, .max_invocations = 128 }, 512, 512));
    try std.testing.expect(!eligibleConfig(small, .{ .max_shared_bytes = 16 * 1024, .max_invocations = 256 }, 516, 512));
}

pub const Matmul = struct {
    tune: autotune.Cache,
    last_choice: ?usize = null,
    /// Arena holding the runtime-generated kernel WGSL (`generated`), built once at
    /// init. Owned here; freed in `deinit`.
    arena: std.heap.ArenaAllocator,
    generated: []const Generated,

    /// Pooled f32 scratch holding one dequantized B tile [k, n] for the q8_0-B
    /// GEMM path. Grows monotonically; freed in `deinit`.
    dq_scratch: ?c.WGPUBuffer = null,
    dq_scratch_cap: u64 = 0,

    pub fn init(allocator: std.mem.Allocator) Matmul {
        var arena = std.heap.ArenaAllocator.init(allocator);
        const generated = configs.generate(arena.allocator());
        return .{ .tune = autotune.Cache.init(allocator), .arena = arena, .generated = generated };
    }
    pub fn deinit(self: *Matmul) void {
        self.tune.deinit();
        self.arena.deinit();
        if (self.dq_scratch) |s| c.wgpuBufferRelease(s);
    }

    fn ensureDqScratch(self: *Matmul, ctx: Ctx, bytes: u64) ExecuteProgramError!c.WGPUBuffer {
        if (self.dq_scratch) |s| {
            if (self.dq_scratch_cap >= bytes) return s;
            c.wgpuBufferRelease(s);
            self.dq_scratch = null;
            self.dq_scratch_cap = 0;
        }
        const MiB: u64 = 1024 * 1024;
        const cap = (bytes + MiB - 1) / MiB * MiB;
        const buf = wgpu.createBuffer(ctx.gpu.device, cap, c.WGPUBufferUsage_Storage) catch return error.ExecutionFailed;
        self.dq_scratch = buf;
        self.dq_scratch_cap = cap;
        return buf;
    }

    pub fn lastChoiceEntry(self: *const Matmul) ?[:0]const u8 {
        const idx = self.last_choice orelse return null;
        if (idx >= self.generated.len) return null;
        return self.generated[idx].entry;
    }

    pub fn lastChoiceThreads(self: *const Matmul) ?u32 {
        const idx = self.last_choice orelse return null;
        if (idx >= self.generated.len) return null;
        return self.generated[idx].cfg.threads();
    }

    pub fn lastChoiceSharedBytes(self: *const Matmul) ?u32 {
        const idx = self.last_choice orelse return null;
        if (idx >= self.generated.len) return null;
        return codegen.sharedBytes(self.generated[idx].cfg);
    }

    pub fn exec(self: *Matmul, ctx: Ctx, frame: *Frame, s: StepMatMul) ExecuteProgramError!void {
        const hs = ctx.rstore.tensorStore();
        const c_meta = hs.meta(s.c) catch return error.ExecutionFailed;
        const a_meta = hs.meta(s.a) catch return error.ExecutionFailed;
        const b_meta = hs.meta(s.b) catch return error.ExecutionFailed;
        if (c_meta.dtype != .f32 or a_meta.dtype != .f32) return error.Unsupported;
        // Quantized B (q8_0): dequant each B tile to f32 scratch, then plain GEMM.
        if (b_meta.dtype == .q8_0) return self.execQuantB(ctx, frame, s, c_meta, a_meta, b_meta);
        if (b_meta.dtype != .f32) return error.Unsupported;
        // Batched matmul (rank > 2): the last two dims are the M×N matrix; leading
        // dims are batch (broadcast when a/b have size 1 there). Mirrors the CPU
        // `execMatMulTiledBatched`.
        if (c_meta.tile_counts.len != 2 or a_meta.tile_counts.len != 2 or b_meta.tile_counts.len != 2) {
            return self.execBatched(ctx, frame, s, c_meta, a_meta, b_meta);
        }
        try ensureTileBindingFits(ctx, c_meta);
        try ensureTileBindingFits(ctx, a_meta);
        try ensureTileBindingFits(ctx, b_meta);

        const idx = try self.chooseConfig(ctx, s, c_meta, a_meta, b_meta);
        const g = self.generated[idx];
        const built = try ctx.pipes.get(g.desc, g.entry);

        const tc_m = c_meta.tile_counts[0];
        const tc_n = c_meta.tile_counts[1];
        var ti_m: usize = 0;
        while (ti_m < tc_m) : (ti_m += 1) {
            var ti_n: usize = 0;
            while (ti_n < tc_n) : (ti_n += 1) {
                try recordOutputTile(ctx, frame, s, g, built, ti_m, ti_n, c_meta, a_meta, b_meta);
            }
        }
    }

    /// Batched matmul: iterate every output tile by linear index, broadcasting
    /// leading batch dims, and record the last-two-dims GEMM per tile. Uses a
    /// single universally-safe (bounds-checked, non-vec4) config rather than the
    /// per-shape autotuner — batched matmuls in these models are small, and the
    /// bounds-checked kernel handles any M/N/K tile without alignment constraints.
    fn execBatched(self: *Matmul, ctx: Ctx, frame: *Frame, s: StepMatMul, c_meta: TensorMeta, a_meta: TensorMeta, b_meta: TensorMeta) ExecuteProgramError!void {
        const rank = c_meta.tile_counts.len;
        if (rank < 3 or rank > 8) return error.Unsupported;
        if (a_meta.tile_counts.len != rank or b_meta.tile_counts.len != rank) return error.Unsupported;

        try ensureTileBindingFits(ctx, c_meta);
        try ensureTileBindingFits(ctx, a_meta);
        try ensureTileBindingFits(ctx, b_meta);

        const idx = try self.batchedConfigIndex(ctx);
        const g = self.generated[idx];
        const built = try ctx.pipes.get(g.desc, g.entry);
        self.last_choice = idx;

        var tile_total: usize = 1;
        for (c_meta.tile_counts) |cnt| tile_total *= cnt;
        const k_tiles = a_meta.tile_counts[rank - 1];

        var coords_buf: [8]usize = undefined;
        var a_coords_buf: [8]usize = undefined;
        var b_coords_buf: [8]usize = undefined;

        var ci: usize = 0;
        while (ci < tile_total) : (ci += 1) {
            const coords = coords_buf[0..rank];
            tensor_store_mod.decodeTileCoords(c_meta, ci, coords) catch return error.ExecutionFailed;
            const ti_m = coords[rank - 2];
            const ti_n = coords[rank - 1];

            var ti_k: usize = 0;
            while (ti_k < k_tiles) : (ti_k += 1) {
                const beta: f32 = if (ti_k == 0) s.beta else 1.0;

                const a_coords = a_coords_buf[0..rank];
                const b_coords = b_coords_buf[0..rank];
                var bd: usize = 0;
                while (bd + 2 < rank) : (bd += 1) {
                    a_coords[bd] = if (a_meta.shape[bd] == 1) 0 else coords[bd];
                    b_coords[bd] = if (b_meta.shape[bd] == 1) 0 else coords[bd];
                }
                a_coords[rank - 2] = ti_m;
                a_coords[rank - 1] = ti_k;
                b_coords[rank - 2] = ti_k;
                b_coords[rank - 1] = ti_n;

                const a_lin = tensor_store_mod.encodeTileIndex(a_meta, a_coords) catch return error.ExecutionFailed;
                const b_lin = tensor_store_mod.encodeTileIndex(b_meta, b_coords) catch return error.ExecutionFailed;
                try recordTileGemmLinear(ctx, frame, s, g, built, ci, a_lin, b_lin, beta, rank);
            }
        }
    }

    /// First bounds-checked, non-vec4 config that fits this device's limits — a
    /// correct fallback for any tile shape (no block-alignment / vec4 stride needs).
    fn batchedConfigIndex(self: *Matmul, ctx: Ctx) ExecuteProgramError!usize {
        for (self.generated, 0..) |gg, idx| {
            const cfg = gg.cfg;
            if (cfg.vec4_load or !cfg.bounds_check) continue;
            if (eligibleConfig(cfg, ctx.gpu.limits, 0, 0)) return idx;
        }
        return error.Unsupported;
    }

    /// Matmul with a q8_0 B (weights in normal [.., K, N] layout, blocks along N):
    /// dequant each B tile to an f32 scratch [K, n] then run the plain GEMM. Handles
    /// rank-2 and batched uniformly by iterating output tiles by linear index.
    /// A/C are f32 (checked by caller). Requires n_tile % 64 == 0 (word-aligned
    /// q8_0 block pairs per row); other shapes fall back to `Unsupported`.
    fn execQuantB(self: *Matmul, ctx: Ctx, frame: *Frame, s: StepMatMul, c_meta: TensorMeta, a_meta: TensorMeta, b_meta: TensorMeta) ExecuteProgramError!void {
        const rank = c_meta.tile_counts.len;
        if (rank < 2 or rank > 8) return error.Unsupported;
        if (a_meta.tile_counts.len != rank or b_meta.tile_counts.len != rank) return error.Unsupported;

        // c/a fit checked with the f32 helper; b is q8_0 (tileBytes would misjudge
        // its packed size), so its per-tile device length is checked at record time.
        try ensureTileBindingFits(ctx, c_meta);
        try ensureTileBindingFits(ctx, a_meta);

        const idx = try self.batchedConfigIndex(ctx);
        const g = self.generated[idx];
        const built = try ctx.pipes.get(g.desc, g.entry);
        self.last_choice = idx;

        var tile_total: usize = 1;
        for (c_meta.tile_counts) |cnt| tile_total *= cnt;
        const k_tiles = a_meta.tile_counts[rank - 1];

        var coords_buf: [8]usize = undefined;
        var a_coords_buf: [8]usize = undefined;
        var b_coords_buf: [8]usize = undefined;

        var ci: usize = 0;
        while (ci < tile_total) : (ci += 1) {
            const coords = coords_buf[0..rank];
            tensor_store_mod.decodeTileCoords(c_meta, ci, coords) catch return error.ExecutionFailed;
            const ti_m = coords[rank - 2];
            const ti_n = coords[rank - 1];

            var ti_k: usize = 0;
            while (ti_k < k_tiles) : (ti_k += 1) {
                const beta: f32 = if (ti_k == 0) s.beta else 1.0;
                const a_coords = a_coords_buf[0..rank];
                const b_coords = b_coords_buf[0..rank];
                var bd: usize = 0;
                while (bd + 2 < rank) : (bd += 1) {
                    a_coords[bd] = if (a_meta.shape[bd] == 1) 0 else coords[bd];
                    b_coords[bd] = if (b_meta.shape[bd] == 1) 0 else coords[bd];
                }
                a_coords[rank - 2] = ti_m;
                a_coords[rank - 1] = ti_k;
                b_coords[rank - 2] = ti_k;
                b_coords[rank - 1] = ti_n;

                const a_lin = tensor_store_mod.encodeTileIndex(a_meta, a_coords) catch return error.ExecutionFailed;
                const b_lin = tensor_store_mod.encodeTileIndex(b_meta, b_coords) catch return error.ExecutionFailed;
                try self.recordQuantTileGemm(ctx, frame, s, g, built, ci, a_lin, b_lin, beta, rank);
            }
        }
    }

    fn recordQuantTileGemm(
        self: *Matmul,
        ctx: Ctx,
        frame: *Frame,
        s: StepMatMul,
        g: Generated,
        built: pipelines.Built,
        c_lin: usize,
        a_lin: usize,
        b_lin: usize,
        beta: f32,
        rank: usize,
    ) ExecuteProgramError!void {
        const hs = ctx.rstore.tensorStore();
        const da = ctx.rstore.acquireTileDeviceConstLinear(s.a, a_lin) catch return error.ExecutionFailed;
        const db = ctx.rstore.acquireTileDeviceConstLinear(s.b, b_lin) catch return error.ExecutionFailed;
        const dc = ctx.rstore.acquireTileDeviceMutLinear(s.c, c_lin) catch return error.ExecutionFailed;
        defer {
            hs.releaseConst(da.token);
            hs.releaseConst(db.token);
            hs.releaseMut(dc.token);
        }
        if (!context.storageBindingFits(ctx, da.len) or !context.storageBindingFits(ctx, dc.len)) return error.Unsupported;

        const m_dim: u32 = @intCast(dc.shape_mem[rank - 2]);
        const n_dim: u32 = @intCast(dc.shape_mem[rank - 1]);
        const k_dim: u32 = @intCast(da.shape_mem[rank - 1]);
        // q8_0 B is quantized along K (ggml MatMul-B convention): block grid
        // [K/32, N]. K must be block-aligned.
        if (k_dim % Q8_BLOCK_ELEMS != 0) return error.Unsupported;

        // M==1 (decode): fuse dequant into the dot product (one B read, no
        // scratch). A is [1, k_dim] so we read it as vec4; a_row is irrelevant.
        // Even N uses the coalesced column-PAIR kernel; odd N the per-column one.
        if (m_dim == 1 and context.storageBindingFits(ctx, db.len)) {
            const params: GemvParams = .{ .k = k_dim, .n = n_dim, .alpha = s.alpha, .beta = beta };
            const bufs = [_]c.WGPUBuffer{
                ctx.devmem.bufferFor(da.handle).?,
                ctx.devmem.bufferFor(db.handle).?,
                ctx.devmem.bufferFor(dc.handle).?,
            };
            const sizes = [_]u64{ da.len, db.len, dc.len };
            if (n_dim % 2 == 0) {
                const gemv_built = try ctx.pipes.get(gemv_kernel, "gemv_q8_kmajor");
                const groups = @max(1, context.ceilDiv(n_dim / 2, GEMV_COLS));
                try frame.recordCompute(gemv_built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups, 1, 1 });
            } else {
                const gemv_built = try ctx.pipes.get(gemv_kernel, "gemv_q8_kmajor_odd");
                const groups = @max(1, context.ceilDiv(n_dim, 256));
                try frame.recordCompute(gemv_built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups, 1, 1 });
            }
            return;
        }
        // Beyond here (M>1 dequant+GEMM) still needs the paired layout → N even.
        if (n_dim % 2 != 0) return error.Unsupported;

        // 1) Dequant the q8_0 B tile [k_dim, n_dim] (blocks along K) -> f32 [k_dim, n_dim].
        const scratch_bytes = @as(u64, k_dim) * n_dim * 4;
        if (!context.storageBindingFits(ctx, scratch_bytes)) return error.Unsupported;
        const b_tile_bytes: u64 = @as(u64, k_dim / Q8_BLOCK_ELEMS) * n_dim * Q8_BLOCK_BYTES;
        if (b_tile_bytes > db.len) return error.Unsupported;
        const scratch = try self.ensureDqScratch(ctx, scratch_bytes);

        const count: u32 = (k_dim / Q8_BLOCK_ELEMS) * (n_dim / 2);
        const dq_built = try ctx.pipes.get(dequant_kernel, "q8_kmajor_to_f32");
        const dq_bufs = [_]c.WGPUBuffer{ ctx.devmem.bufferFor(db.handle).?, scratch };
        const dq_sizes = [_]u64{ db.len, scratch_bytes };
        const dq_params: DequantParams = .{ .n = n_dim, .k = k_dim, .src_wpr = 0, .dst_row = n_dim, .count = count };
        const dq_groups = @max(1, @min(context.ceilDiv(count, DEQUANT_WG), context.MAX_GROUPS_1D));
        try frame.recordCompute(dq_built, &dq_bufs, &dq_sizes, std.mem.asBytes(&dq_params), .{ dq_groups, 1, 1 });

        // 2) Plain GEMM: A[m, k] @ scratch[k, n] -> C[m, n].
        const params: MatMulParams = .{
            .m = m_dim,
            .n = n_dim,
            .k = k_dim,
            .a_row = @intCast(@divExact(da.strides_mem[rank - 2], @sizeOf(f32))),
            .b_row = n_dim,
            .c_row = @intCast(@divExact(dc.strides_mem[rank - 2], @sizeOf(f32))),
            .alpha = s.alpha,
            .beta = beta,
        };
        const bufs = [_]c.WGPUBuffer{
            ctx.devmem.bufferFor(da.handle).?,
            scratch,
            ctx.devmem.bufferFor(dc.handle).?,
        };
        const sizes = [_]u64{ da.len, scratch_bytes, dc.len };
        try frame.recordCompute(
            built,
            &bufs,
            &sizes,
            std.mem.asBytes(&params),
            .{ context.ceilDiv(n_dim, g.cfg.bn), context.ceilDiv(m_dim, g.cfg.bm), 1 },
        );
    }

    /// Per-shape autotune over `configs.generated`: benchmark each eligible config
    /// on-device once per tile shape and cache the fastest.
    fn chooseConfig(self: *Matmul, ctx: Ctx, s: StepMatMul, c_meta: TensorMeta, a_meta: TensorMeta, b_meta: TensorMeta) ExecuteProgramError!usize {
        // Tuning re-runs output tile (0,0) on the real C tensor; with beta != 0 that
        // would accumulate junk into C(0,0) before the real pass reads it. beta == 0
        // (the lowering's default) overwrites, so tuning is safe there. For beta != 0
        // skip tuning and use config 0 (always eligible/correct).
        if (s.beta != 0.0) {
            self.last_choice = 0;
            return 0;
        }

        // Row strides (bytes) of tile (0,0) drive vec4 eligibility.
        const da0 = ctx.rstore.acquireTileDeviceConstLinear(s.a, 0) catch return error.ExecutionFailed;
        const a_row_bytes = da0.strides_mem[0];
        ctx.rstore.tensorStore().releaseConst(da0.token);
        const db0 = ctx.rstore.acquireTileDeviceConstLinear(s.b, 0) catch return error.ExecutionFailed;
        const b_row_bytes = db0.strides_mem[0];
        ctx.rstore.tensorStore().releaseConst(db0.token);

        if (try forcedConfigIndex(self.generated, ctx.gpu.limits, a_row_bytes, b_row_bytes, c_meta, a_meta, b_meta)) |idx| {
            self.last_choice = idx;
            return idx;
        }

        const TuneCtx = struct {
            ctx: Ctx,
            s: StepMatMul,
            c_meta: TensorMeta,
            a_meta: TensorMeta,
            b_meta: TensorMeta,
            a_row_bytes: isize,
            b_row_bytes: isize,
            generated: []const Generated,

            pub fn eligible(t: @This(), idx: usize) bool {
                const cfg = t.generated[idx].cfg;
                return fullTileCompatible(cfg, t.c_meta, t.a_meta, t.b_meta) and eligibleConfig(cfg, t.ctx.gpu.limits, t.a_row_bytes, t.b_row_bytes);
            }
            pub fn timeNs(t: @This(), idx: usize) ?u64 {
                const g = t.generated[idx];
                const built = t.ctx.pipes.get(g.desc, g.entry) catch return null;
                var best: ?u64 = null;
                var rep: usize = 0;
                while (rep < 2) : (rep += 1) {
                    const ns = timeConfig(t.ctx, t.s, g, built, t.c_meta, t.a_meta, t.b_meta) catch return null;
                    if (best == null or ns < best.?) best = ns;
                }
                return best;
            }
        };
        const tctx = TuneCtx{
            .ctx = ctx,
            .s = s,
            .c_meta = c_meta,
            .a_meta = a_meta,
            .b_meta = b_meta,
            .a_row_bytes = a_row_bytes,
            .b_row_bytes = b_row_bytes,
            .generated = self.generated,
        };
        const key = autotune.shapeKey(c_meta.tile_shape[0], c_meta.tile_shape[1], a_meta.tile_shape[1]);
        const idx = autotune.pickBest(&self.tune, key, self.generated.len, tctx) orelse return error.ExecutionFailed;
        self.last_choice = idx;
        return idx;
    }
};

/// Record all k-tile dispatches that compute output C tile (ti_m, ti_n) with config
/// `g`. Shared by the main execute loop and the autotuner's timing.
fn recordOutputTile(
    ctx: Ctx,
    frame: *Frame,
    s: StepMatMul,
    g: Generated,
    built: pipelines.Built,
    ti_m: usize,
    ti_n: usize,
    c_meta: TensorMeta,
    a_meta: TensorMeta,
    b_meta: TensorMeta,
) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
    const k_tiles = a_meta.tile_counts[1];
    const c_lin = tensor_store_mod.encodeTileIndex(c_meta, &[_]usize{ ti_m, ti_n }) catch return error.ExecutionFailed;
    var ti_k: usize = 0;
    while (ti_k < k_tiles) : (ti_k += 1) {
        const beta: f32 = if (ti_k == 0) s.beta else 1.0;
        const a_lin = tensor_store_mod.encodeTileIndex(a_meta, &[_]usize{ ti_m, ti_k }) catch return error.ExecutionFailed;
        const b_lin = tensor_store_mod.encodeTileIndex(b_meta, &[_]usize{ ti_k, ti_n }) catch return error.ExecutionFailed;

        const da = ctx.rstore.acquireTileDeviceConstLinear(s.a, a_lin) catch return error.ExecutionFailed;
        const db = ctx.rstore.acquireTileDeviceConstLinear(s.b, b_lin) catch return error.ExecutionFailed;
        const dc = ctx.rstore.acquireTileDeviceMutLinear(s.c, c_lin) catch return error.ExecutionFailed;
        if (!context.storageBindingFits(ctx, da.len) or !context.storageBindingFits(ctx, db.len) or !context.storageBindingFits(ctx, dc.len)) {
            hs.releaseConst(da.token);
            hs.releaseConst(db.token);
            hs.releaseMut(dc.token);
            return error.Unsupported;
        }

        const m_dim: u32 = @intCast(dc.shape_mem[0]);
        const n_dim: u32 = @intCast(dc.shape_mem[1]);
        const k_dim: u32 = @intCast(da.shape_mem[1]);
        const params: MatMulParams = .{
            .m = m_dim,
            .n = n_dim,
            .k = k_dim,
            .a_row = @intCast(@divExact(da.strides_mem[0], @sizeOf(f32))),
            .b_row = @intCast(@divExact(db.strides_mem[0], @sizeOf(f32))),
            .c_row = @intCast(@divExact(dc.strides_mem[0], @sizeOf(f32))),
            .alpha = s.alpha,
            .beta = beta,
        };
        const bufs = [_]c.WGPUBuffer{
            ctx.devmem.bufferFor(da.handle).?,
            ctx.devmem.bufferFor(db.handle).?,
            ctx.devmem.bufferFor(dc.handle).?,
        };
        const sizes = [_]u64{ da.len, db.len, dc.len };
        try frame.recordCompute(
            built,
            &bufs,
            &sizes,
            std.mem.asBytes(&params),
            .{ context.ceilDiv(n_dim, g.cfg.bn), context.ceilDiv(m_dim, g.cfg.bm), 1 },
        );

        hs.releaseConst(da.token);
        hs.releaseConst(db.token);
        hs.releaseMut(dc.token);
    }
}

/// Record the GEMM for one batched output tile (given by linear indices into the
/// c/a/b tile grids). The matrix dims are the last two of each tile; leading dims
/// were resolved by the caller (`execBatched`). One dispatch (single k-tile step).
fn recordTileGemmLinear(
    ctx: Ctx,
    frame: *Frame,
    s: StepMatMul,
    g: Generated,
    built: pipelines.Built,
    c_lin: usize,
    a_lin: usize,
    b_lin: usize,
    beta: f32,
    rank: usize,
) ExecuteProgramError!void {
    const hs = ctx.rstore.tensorStore();
    const da = ctx.rstore.acquireTileDeviceConstLinear(s.a, a_lin) catch return error.ExecutionFailed;
    const db = ctx.rstore.acquireTileDeviceConstLinear(s.b, b_lin) catch return error.ExecutionFailed;
    const dc = ctx.rstore.acquireTileDeviceMutLinear(s.c, c_lin) catch return error.ExecutionFailed;
    defer {
        hs.releaseConst(da.token);
        hs.releaseConst(db.token);
        hs.releaseMut(dc.token);
    }
    if (!context.storageBindingFits(ctx, da.len) or !context.storageBindingFits(ctx, db.len) or !context.storageBindingFits(ctx, dc.len)) {
        return error.Unsupported;
    }

    const m_dim: u32 = @intCast(dc.shape_mem[rank - 2]);
    const n_dim: u32 = @intCast(dc.shape_mem[rank - 1]);
    const k_dim: u32 = @intCast(da.shape_mem[rank - 1]);
    const params: MatMulParams = .{
        .m = m_dim,
        .n = n_dim,
        .k = k_dim,
        .a_row = @intCast(@divExact(da.strides_mem[rank - 2], @sizeOf(f32))),
        .b_row = @intCast(@divExact(db.strides_mem[rank - 2], @sizeOf(f32))),
        .c_row = @intCast(@divExact(dc.strides_mem[rank - 2], @sizeOf(f32))),
        .alpha = s.alpha,
        .beta = beta,
    };
    const bufs = [_]c.WGPUBuffer{
        ctx.devmem.bufferFor(da.handle).?,
        ctx.devmem.bufferFor(db.handle).?,
        ctx.devmem.bufferFor(dc.handle).?,
    };
    const sizes = [_]u64{ da.len, db.len, dc.len };
    try frame.recordCompute(
        built,
        &bufs,
        &sizes,
        std.mem.asBytes(&params),
        .{ context.ceilDiv(n_dim, g.cfg.bn), context.ceilDiv(m_dim, g.cfg.bm), 1 },
    );
}

/// Time `TUNE_ITERS` recomputations of output tile (0,0) with config `g` (own
/// throwaway frames + a single device sync), returning total nanoseconds.
fn timeConfig(ctx: Ctx, s: StepMatMul, g: Generated, built: pipelines.Built, c_meta: TensorMeta, a_meta: TensorMeta, b_meta: TensorMeta) ExecuteProgramError!u64 {
    const TUNE_ITERS = 24;
    {
        var f = try Frame.init(ctx.allocator, ctx.gpu);
        defer f.deinit();
        try recordOutputTile(ctx, &f, s, g, built, 0, 0, c_meta, a_meta, b_meta);
        f.submit();
    }
    syncDevice(ctx);
    const start = autotune.nowNs();
    var t: usize = 0;
    while (t < TUNE_ITERS) : (t += 1) {
        var f = try Frame.init(ctx.allocator, ctx.gpu);
        defer f.deinit();
        try recordOutputTile(ctx, &f, s, g, built, 0, 0, c_meta, a_meta, b_meta);
        f.submit();
    }
    syncDevice(ctx);
    return autotune.nowNs() - start;
}
