// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Matmul execution for the GPU backend: one GEMM dispatch per step, per-shape
//! autotuning, and on-device timing. Owns the autotune cache (one `Matmul` lives on the backend).
//! Uses the generic `autotune` helper + the generated kernels from `configs.zig`.

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
const env_util = @import("../../../env.zig");

const c = wgpu.c;
const fns = wgpu.fns; // runtime wgpu dispatch table (functions)
const Ctx = context.Ctx;
const Frame = @import("../frame.zig").Frame;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const TensorMeta = tensor_store_mod.TensorMeta;
const MatmulConfig = codegen.MatmulConfig;
const Generated = codegen.Generated;
const StepMatMul = executable.StepMatMul;

/// Shared dequant module (also used by the MatMulNT executor). We use its
/// `q8_to_f32` entry to materialize a q8_0 B as f32 for the plain GEMM.
const dequant_kernel: pipelines.KernelDesc = .{ .name = "dequant", .wgsl = @embedFile("../kernels/dequant.wgsl") };
/// Fused q8_0 (K-major) matvec for M==1 (decode): folds the dequant into the dot
/// so B is read once, no f32 scratch. See kernels/matmul_gemv.wgsl.
const gemv_kernel: pipelines.KernelDesc = .{ .name = "matmul_gemv", .wgsl = @embedFile("../kernels/matmul_gemv.wgsl") };
/// Column-pairs per GEMV workgroup (matches `COLS` in matmul_gemv.wgsl).
const GEMV_COLS: u32 = 32;
/// Same for the narrow variant (`COLS_N`), and the group count below which it wins.
/// Tuned on a 58-SM 4080 Laptop: fewer groups than this and whole SMs sit idle.
const GEMV_COLS_NARROW: u32 = 8;
const GEMV_MIN_GROUPS: u32 = 64;
const Q8_BLOCK_ELEMS: u32 = 32;
const Q8_BLOCK_BYTES: u32 = 34;
const DEQUANT_WG: u32 = 64;

/// Uniform params for the fused GEMV; field order matches `struct Params` in
/// matmul_gemv.wgsl: { k, n, _p0, _p1, alpha, beta }.
const GemvParams = extern struct { k: u32, n: u32, _p0: u32 = 0, _p1: u32 = 0, alpha: f32, beta: f32 };
/// Field order matches `struct Params` in dequant.wgsl.
const DequantParams = extern struct { n: u32, k: u32, src_wpr: u32, dst_row: u32, count: u32, _p0: u32 = 0, _p1: u32 = 0, _p2: u32 = 0 };

/// Uniform params for the GEMM kernel; field order matches the WGSL
/// `struct Params { dims: vec4<u32>, strides: vec4<u32>, ab: vec4<f32> }`.
/// Shared with the MatMulNT executor (matmul_nt.zig), which drives the same GEMM
/// pipelines over a dequantized scratch B.
pub const MatMulParams = extern struct {
    m: u32,
    n: u32,
    k: u32,
    /// Elements between consecutive batches of A (workgroup z); 0 broadcasts.
    a_batch: u32 = 0,
    a_row: u32,
    b_row: u32,
    c_row: u32,
    /// Same for B. C's batches are always `m * c_row` apart.
    b_batch: u32 = 0,
    alpha: f32,
    beta: f32,
    /// Element offset of C's first output: an NT weight chunked along N writes
    /// its columns of one C.
    c_off: u32 = 0,
    _a3: u32 = 0,
};

fn syncDevice(ctx: Ctx) void {
    _ = fns.wgpuDevicePoll(ctx.gpu.device, 1, null);
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

/// One GEMM over whole tensors: `batch` independent `[m, k] @ [k, n]` products,
/// batch `z` of A/B starting `z * a_batch` / `z * b_batch` elements in (0 broadcasts).
const Gemm = struct {
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    a_batch: usize,
    b_batch: usize,
};

/// The GEMM a `C[.., M, N] = A[.., M, K] @ B[.., K, N]` step is. Leading dims of A
/// and B each either match C's or are all 1 (broadcast); B's are right-aligned. A
/// broadcast B folds A's batches into M, so the common activations-times-weight
/// case is one plain 2-D GEMM.
fn gemmShape(c_meta: TensorMeta, a_meta: TensorMeta, b_meta: TensorMeta) ExecuteProgramError!Gemm {
    const r: usize = c_meta.rank;
    const br: usize = b_meta.rank;
    if (r < 2 or a_meta.rank != r or br < 2 or br > r) return error.Unsupported;
    const m = c_meta.shape[r - 2];
    const n = c_meta.shape[r - 1];
    const k = a_meta.shape[r - 1];
    if (a_meta.shape[r - 2] != m or b_meta.shape[br - 2] != k or b_meta.shape[br - 1] != n) return error.Unsupported;

    var batch: usize = 1;
    for (c_meta.shape[0 .. r - 2]) |d| batch *= d;
    const a_full = std.mem.eql(usize, a_meta.shape[0 .. r - 2], c_meta.shape[0 .. r - 2]);
    const a_ones = for (a_meta.shape[0 .. r - 2]) |d| {
        if (d != 1) break false;
    } else true;
    const b_lead = b_meta.shape[0 .. br - 2];
    const b_full = b_lead.len == r - 2 and std.mem.eql(usize, b_lead, c_meta.shape[0 .. r - 2]);
    const b_ones = for (b_lead) |d| {
        if (d != 1) break false;
    } else true;
    if (!(a_full or a_ones) or !(b_full or b_ones)) return error.Unsupported;

    if (batch == 1 or (b_ones and a_full)) return .{ .m = batch * m, .n = n, .k = k, .batch = 1, .a_batch = 0, .b_batch = 0 };
    return .{
        .m = m,
        .n = n,
        .k = k,
        .batch = batch,
        .a_batch = if (a_ones) 0 else m * k,
        .b_batch = if (b_ones) 0 else k * n,
    };
}

/// Plain configs (no bounds checks) assume every dim is a whole number of blocks.
fn blockAligned(cfg: MatmulConfig, g: Gemm) bool {
    if (cfg.bounds_check) return true;
    return g.m % cfg.bm == 0 and g.n % cfg.bn == 0 and g.k % cfg.bk == 0;
}

fn forcedConfigIndex(generated: []const Generated, limits: wgpu.Limits, row_bytes: [2]isize, g: Gemm) ExecuteProgramError!?usize {
    const raw = env_util.getOwned(std.heap.page_allocator, "AION_MATMUL_CONFIG") orelse return null;
    defer std.heap.page_allocator.free(raw);

    const idx = std.fmt.parseInt(usize, raw, 10) catch for (generated, 0..) |gen, i| {
        if (std.mem.eql(u8, raw, gen.entry)) break i;
    } else return error.Unsupported;
    if (idx >= generated.len) return error.Unsupported;
    const cfg = generated[idx].cfg;
    if (!blockAligned(cfg, g) or !eligibleConfig(cfg, limits, row_bytes[0], row_bytes[1])) return error.Unsupported;
    return idx;
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
    /// The implicit-GEMM conv kernels, rendered into the same arena.
    generated_conv: []const Generated,

    /// Pooled f32 scratch holding a dequantized B [k, n] for the q8_0-B GEMM path.
    /// Grows monotonically; freed in `deinit`.
    dq_scratch: ?c.WGPUBuffer = null,
    dq_scratch_cap: u64 = 0,

    pub fn init(allocator: std.mem.Allocator) Matmul {
        var arena = std.heap.ArenaAllocator.init(allocator);
        const generated = configs.generate(arena.allocator());
        const generated_conv = configs.generateConv(arena.allocator());
        return .{ .tune = autotune.Cache.init(allocator), .arena = arena, .generated = generated, .generated_conv = generated_conv };
    }
    pub fn deinit(self: *Matmul) void {
        self.tune.deinit();
        self.arena.deinit();
        if (self.dq_scratch) |s| fns.wgpuBufferRelease(s);
    }

    fn ensureDqScratch(self: *Matmul, ctx: Ctx, bytes: u64) ExecuteProgramError!c.WGPUBuffer {
        if (self.dq_scratch) |s| {
            if (self.dq_scratch_cap >= bytes) return s;
            fns.wgpuBufferRelease(s);
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
        const hs = ctx.store;
        const c_meta = hs.meta(s.c) catch return error.ExecutionFailed;
        const a_meta = hs.meta(s.a) catch return error.ExecutionFailed;
        const b_meta = hs.meta(s.b) catch return error.ExecutionFailed;
        if (c_meta.dtype != .f32 or a_meta.dtype != .f32) return error.Unsupported;
        // Every operand is one device buffer; a tensor past the binding limit is
        // chunked, which this kernel does not address.
        inline for (.{ c_meta, a_meta, b_meta }) |m| if (m.chunks != 1) return error.Unsupported;
        const g = try gemmShape(c_meta, a_meta, b_meta);
        if (g.m == 0 or g.n == 0) return;
        if (b_meta.dtype == .q8_0) return self.execQuantB(ctx, frame, s, g);
        if (b_meta.dtype != .f32) return error.Unsupported;

        // Batched products are small in these models: one bounds-checked config
        // rather than a tune per shape.
        const idx = if (g.batch == 1) try self.chooseConfig(ctx, s, g) else try self.safeConfigIndex(ctx);
        self.last_choice = idx;
        const gen = self.generated[idx];
        const built = try ctx.pipes.get(gen.desc, gen.entry);
        try recordGemm(ctx, frame, s, gen, built, g);
    }

    /// First bounds-checked, non-vec4 config that fits this device's limits — a
    /// correct fallback for any shape (no block-alignment / vec4 stride needs).
    fn safeConfigIndex(self: *Matmul, ctx: Ctx) ExecuteProgramError!usize {
        for (self.generated, 0..) |gg, idx| {
            const cfg = gg.cfg;
            if (cfg.vec4_load or !cfg.bounds_check) continue;
            if (eligibleConfig(cfg, ctx.gpu.limits, 0, 0)) return idx;
        }
        return error.Unsupported;
    }

    /// Matmul with a q8_0 B (normal [K, N] layout, blocks along K). M == 1 (decode)
    /// fuses the dequant into the dot product; otherwise B is dequantized to an f32
    /// scratch once and the plain GEMM runs. B is a weight, so batches of A fold
    /// into M; a batched quantized B is not supported.
    fn execQuantB(self: *Matmul, ctx: Ctx, frame: *Frame, s: StepMatMul, g: Gemm) ExecuteProgramError!void {
        if (g.batch != 1) return error.Unsupported;
        const m_dim = std.math.cast(u32, g.m) orelse return error.Unsupported;
        const n_dim = std.math.cast(u32, g.n) orelse return error.Unsupported;
        const k_dim = std.math.cast(u32, g.k) orelse return error.Unsupported;
        if (k_dim % Q8_BLOCK_ELEMS != 0) return error.Unsupported;

        const hs = ctx.store;
        const da = hs.acquireConst(s.a) catch return error.ExecutionFailed;
        defer hs.releaseConst(da.token);
        const db = hs.acquireConst(s.b) catch return error.ExecutionFailed;
        defer hs.releaseConst(db.token);
        const dc = hs.acquireMut(s.c) catch return error.ExecutionFailed;
        defer hs.releaseMut(dc.token);
        inline for (.{ da.len, db.len, dc.len }) |len| if (!context.storageBindingFits(ctx, len)) return error.Unsupported;

        if (m_dim == 1) {
            const params: GemvParams = .{ .k = k_dim, .n = n_dim, .alpha = s.alpha, .beta = s.beta };
            const bufs = [_]c.WGPUBuffer{
                ctx.devmem.bufferFor(da.handle).?,
                ctx.devmem.bufferFor(db.handle).?,
                ctx.devmem.bufferFor(dc.handle).?,
            };
            const sizes = [_]u64{ da.len, db.len, dc.len };
            // Even N uses the coalesced column-PAIR kernel; odd N the per-column one.
            if (n_dim % 2 == 0) {
                // The group count is pairs/COLS, so a small N leaves most of the GPU
                // idle at the wide kernel's COLS=32. Below the occupancy threshold the
                // narrow kernel quarters COLS for 4x the groups at the same total
                // threads and the same B traffic.
                const pairs = n_dim / 2;
                const wide_groups = @max(1, context.ceilDiv(pairs, GEMV_COLS));
                if (wide_groups < GEMV_MIN_GROUPS) {
                    const gemv_built = try ctx.pipes.get(gemv_kernel, "gemv_q8_kmajor_narrow");
                    const groups = @max(1, context.ceilDiv(pairs, GEMV_COLS_NARROW));
                    try frame.recordCompute(gemv_built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups, 1, 1 });
                } else {
                    const gemv_built = try ctx.pipes.get(gemv_kernel, "gemv_q8_kmajor");
                    try frame.recordCompute(gemv_built, &bufs, &sizes, std.mem.asBytes(&params), .{ wide_groups, 1, 1 });
                }
            } else {
                const gemv_built = try ctx.pipes.get(gemv_kernel, "gemv_q8_kmajor_odd");
                const groups = @max(1, context.ceilDiv(n_dim, 256));
                try frame.recordCompute(gemv_built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups, 1, 1 });
            }
            return;
        }
        // The dequant kernel works on column pairs.
        if (n_dim % 2 != 0) return error.Unsupported;

        const scratch_bytes = @as(u64, k_dim) * n_dim * 4;
        if (!context.storageBindingFits(ctx, scratch_bytes)) return error.Unsupported;
        if (@as(u64, k_dim / Q8_BLOCK_ELEMS) * n_dim * Q8_BLOCK_BYTES > db.len) return error.Unsupported;
        const scratch = try self.ensureDqScratch(ctx, scratch_bytes);

        const count: u32 = (k_dim / Q8_BLOCK_ELEMS) * (n_dim / 2);
        const dq_built = try ctx.pipes.get(dequant_kernel, "q8_kmajor_to_f32");
        const dq_bufs = [_]c.WGPUBuffer{ ctx.devmem.bufferFor(db.handle).?, scratch };
        const dq_sizes = [_]u64{ db.len, scratch_bytes };
        const dq_params: DequantParams = .{ .n = n_dim, .k = k_dim, .src_wpr = 0, .dst_row = n_dim, .count = count };
        const dq_groups = @max(1, @min(context.ceilDiv(count, DEQUANT_WG), context.MAX_GROUPS_1D));
        try frame.recordCompute(dq_built, &dq_bufs, &dq_sizes, std.mem.asBytes(&dq_params), .{ dq_groups, 1, 1 });

        const idx = try self.safeConfigIndex(ctx);
        self.last_choice = idx;
        const gen = self.generated[idx];
        const built = try ctx.pipes.get(gen.desc, gen.entry);
        const gy = context.ceilDiv(m_dim, gen.cfg.bm);
        if (gy > context.MAX_GROUPS_PER_DIM) return error.Unsupported;
        const params: MatMulParams = .{ .m = m_dim, .n = n_dim, .k = k_dim, .a_row = k_dim, .b_row = n_dim, .c_row = n_dim, .alpha = s.alpha, .beta = s.beta };
        const bufs = [_]c.WGPUBuffer{ ctx.devmem.bufferFor(da.handle).?, scratch, ctx.devmem.bufferFor(dc.handle).? };
        const sizes = [_]u64{ da.len, scratch_bytes, dc.len };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ context.ceilDiv(n_dim, gen.cfg.bn), gy, 1 });
    }

    /// Per-shape autotune over `configs.generated`: benchmark each eligible config
    /// on-device once per `(m, n, k)` and cache the fastest.
    fn chooseConfig(self: *Matmul, ctx: Ctx, s: StepMatMul, g: Gemm) ExecuteProgramError!usize {
        // Tuning re-runs the product on the real C; with beta != 0 that would
        // accumulate junk into C before the real pass reads it. beta == 0 (the
        // lowering's default) overwrites, so tuning is safe there. For beta != 0 skip
        // tuning and use config 0 (always eligible/correct).
        if (s.beta != 0.0) return 0;

        // Row strides (bytes) drive vec4 eligibility.
        const row_bytes: [2]isize = .{ @intCast(g.k * @sizeOf(f32)), @intCast(g.n * @sizeOf(f32)) };
        if (try forcedConfigIndex(self.generated, ctx.gpu.limits, row_bytes, g)) |idx| return idx;

        const TuneCtx = struct {
            ctx: Ctx,
            s: StepMatMul,
            g: Gemm,
            row_bytes: [2]isize,
            generated: []const Generated,

            pub fn eligible(t: @This(), idx: usize) bool {
                const cfg = t.generated[idx].cfg;
                return blockAligned(cfg, t.g) and eligibleConfig(cfg, t.ctx.gpu.limits, t.row_bytes[0], t.row_bytes[1]);
            }
            pub fn timeNs(t: @This(), idx: usize) ?u64 {
                const gen = t.generated[idx];
                const built = t.ctx.pipes.get(gen.desc, gen.entry) catch return null;
                var best: ?u64 = null;
                var rep: usize = 0;
                while (rep < 2) : (rep += 1) {
                    const ns = timeConfig(t.ctx, t.s, gen, built, t.g) catch return null;
                    if (best == null or ns < best.?) best = ns;
                }
                return best;
            }
        };
        const tctx = TuneCtx{ .ctx = ctx, .s = s, .g = g, .row_bytes = row_bytes, .generated = self.generated };
        return autotune.pickBest(&self.tune, autotune.shapeKey(g.m, g.n, g.k), self.generated.len, tctx) orelse error.ExecutionFailed;
    }
};

fn castU32(v: usize) ExecuteProgramError!u32 {
    return std.math.cast(u32, v) orelse error.Unsupported;
}

/// Record the one dispatch computing `g` with config `gen`. Shared by the execute
/// path and the autotuner's timing.
fn recordGemm(ctx: Ctx, frame: *Frame, s: StepMatMul, gen: Generated, built: pipelines.Built, g: Gemm) ExecuteProgramError!void {
    const hs = ctx.store;
    const da = hs.acquireConst(s.a) catch return error.ExecutionFailed;
    defer hs.releaseConst(da.token);
    const db = hs.acquireConst(s.b) catch return error.ExecutionFailed;
    defer hs.releaseConst(db.token);
    const dc = hs.acquireMut(s.c) catch return error.ExecutionFailed;
    defer hs.releaseMut(dc.token);
    inline for (.{ da.len, db.len, dc.len }) |len| if (!context.storageBindingFits(ctx, len)) return error.Unsupported;

    const params: MatMulParams = .{
        .m = try castU32(g.m),
        .n = try castU32(g.n),
        .k = try castU32(g.k),
        .a_batch = try castU32(g.a_batch),
        .a_row = try castU32(g.k),
        .b_row = try castU32(g.n),
        .c_row = try castU32(g.n),
        .b_batch = try castU32(g.b_batch),
        .alpha = s.alpha,
        .beta = s.beta,
    };
    const gx = context.ceilDiv(params.n, gen.cfg.bn);
    const gy = context.ceilDiv(params.m, gen.cfg.bm);
    const gz = try castU32(g.batch);
    if (gx > context.MAX_GROUPS_PER_DIM or gy > context.MAX_GROUPS_PER_DIM or gz > context.MAX_GROUPS_PER_DIM) return error.Unsupported;
    const bufs = [_]c.WGPUBuffer{
        ctx.devmem.bufferFor(da.handle).?,
        ctx.devmem.bufferFor(db.handle).?,
        ctx.devmem.bufferFor(dc.handle).?,
    };
    const sizes = [_]u64{ da.len, db.len, dc.len };
    try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ gx, gy, gz });
}

/// Time `TUNE_ITERS` recomputations of `g` with config `gen` (own throwaway frames
/// + a single device sync), returning total nanoseconds.
fn timeConfig(ctx: Ctx, s: StepMatMul, gen: Generated, built: pipelines.Built, g: Gemm) ExecuteProgramError!u64 {
    const TUNE_ITERS = 24;
    {
        var f = try Frame.init(ctx.allocator, ctx.gpu);
        defer f.deinit();
        try recordGemm(ctx, &f, s, gen, built, g);
        f.submit();
    }
    syncDevice(ctx);
    const start = autotune.nowNs();
    var t: usize = 0;
    while (t < TUNE_ITERS) : (t += 1) {
        var f = try Frame.init(ctx.allocator, ctx.gpu);
        defer f.deinit();
        try recordGemm(ctx, &f, s, gen, built, g);
        f.submit();
    }
    syncDevice(ctx);
    return autotune.nowNs() - start;
}
