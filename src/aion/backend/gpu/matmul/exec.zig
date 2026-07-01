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
const codegen = @import("codegen.zig");
const configs = @import("configs.zig");
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

/// Uniform params for the tiled GEMM kernel; field order matches the WGSL
/// `struct Params { dims: vec4<u32>, strides: vec4<u32>, ab: vec4<f32> }`.
const MatMulParams = extern struct {
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
/// loads, A/B row strides are 16-byte aligned.
fn eligibleConfig(cfg: MatmulConfig, limits: wgpu.Limits, a_row_bytes: isize, b_row_bytes: isize) bool {
    if (codegen.sharedBytes(cfg) > limits.max_shared_bytes) return false;
    if (cfg.threads() > limits.max_invocations) return false;
    if (cfg.threads() > limits.max_workgroup_size_x) return false;
    // Subgroup kernels are specialized to a 32-wide subgroup and need the feature.
    if (cfg.subgroup and limits.fixedSubgroupSize() != codegen.SUBGROUP_SIZE) return false;
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

fn forcedConfigIndex(ctx: Ctx, limits: wgpu.Limits, a_row_bytes: isize, b_row_bytes: isize, c_meta: TensorMeta, a_meta: TensorMeta, b_meta: TensorMeta) ExecuteProgramError!?usize {
    _ = ctx;
    const env = std.c.getenv("AION_MATMUL_CONFIG") orelse return null;
    const raw = std.mem.span(env);

    const maybe_idx = std.fmt.parseInt(usize, raw, 10) catch null;
    if (maybe_idx) |idx| {
        if (idx >= configs.generated.len) return error.Unsupported;
        const cfg = configs.generated[idx].cfg;
        if (!fullTileCompatible(cfg, c_meta, a_meta, b_meta) or !eligibleConfig(cfg, limits, a_row_bytes, b_row_bytes)) return error.Unsupported;
        return idx;
    }

    for (configs.generated, 0..) |g, idx| {
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

    pub fn init(allocator: std.mem.Allocator) Matmul {
        return .{ .tune = autotune.Cache.init(allocator) };
    }
    pub fn deinit(self: *Matmul) void {
        self.tune.deinit();
    }

    pub fn lastChoiceEntry(self: *const Matmul) ?[:0]const u8 {
        const idx = self.last_choice orelse return null;
        if (idx >= configs.generated.len) return null;
        return configs.generated[idx].entry;
    }

    pub fn lastChoiceThreads(self: *const Matmul) ?u32 {
        const idx = self.last_choice orelse return null;
        if (idx >= configs.generated.len) return null;
        return configs.generated[idx].cfg.threads();
    }

    pub fn lastChoiceSharedBytes(self: *const Matmul) ?u32 {
        const idx = self.last_choice orelse return null;
        if (idx >= configs.generated.len) return null;
        return codegen.sharedBytes(configs.generated[idx].cfg);
    }

    pub fn exec(self: *Matmul, ctx: Ctx, frame: *Frame, s: StepMatMul) ExecuteProgramError!void {
        const hs = ctx.rstore.tensorStore();
        const c_meta = hs.meta(s.c) catch return error.ExecutionFailed;
        const a_meta = hs.meta(s.a) catch return error.ExecutionFailed;
        const b_meta = hs.meta(s.b) catch return error.ExecutionFailed;
        if (c_meta.dtype != .f32 or a_meta.dtype != .f32 or b_meta.dtype != .f32) return error.Unsupported;
        // v1 scope: 2D tiling only (leading batch dims not yet handled).
        if (c_meta.tile_counts.len != 2 or a_meta.tile_counts.len != 2 or b_meta.tile_counts.len != 2) {
            return error.Unsupported;
        }
        try ensureTileBindingFits(ctx, c_meta);
        try ensureTileBindingFits(ctx, a_meta);
        try ensureTileBindingFits(ctx, b_meta);

        const idx = try self.chooseConfig(ctx, s, c_meta, a_meta, b_meta);
        const g = configs.generated[idx];
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

        if (try forcedConfigIndex(ctx, ctx.gpu.limits, a_row_bytes, b_row_bytes, c_meta, a_meta, b_meta)) |idx| {
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

            pub fn eligible(t: @This(), idx: usize) bool {
                const cfg = configs.generated[idx].cfg;
                return fullTileCompatible(cfg, t.c_meta, t.a_meta, t.b_meta) and eligibleConfig(cfg, t.ctx.gpu.limits, t.a_row_bytes, t.b_row_bytes);
            }
            pub fn timeNs(t: @This(), idx: usize) ?u64 {
                const g = configs.generated[idx];
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
        };
        const key = autotune.shapeKey(c_meta.tile_shape[0], c_meta.tile_shape[1], a_meta.tile_shape[1]);
        const idx = autotune.pickBest(&self.tune, key, configs.generated.len, tctx) orelse return error.ExecutionFailed;
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
