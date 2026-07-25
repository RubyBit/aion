// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! MatMulNT execution for the GPU backend:
//!   C[m, n] = alpha * sum_k A[m, k] * B[n, k]  +  beta * C[m, n]
//! with A f32 (single tile spanning [M, K]) and B [N, K] in q8_0 or f32,
//! N-tiled in lockstep with C's trailing axis (compile-enforced, same contract
//! as the CPU executor in `backend/cpu/exec/matmul_nt.zig`).
//!
//! Two regimes, split on M:
//!   - M == 1 (decode): bandwidth-bound matvec — `kernels/matmul_nt_gemv.wgsl`
//!     reads B exactly once (dequant-in-registers for q8_0).
//!   - M > 1 (prefill): compute-bound — a dequant/transpose pass materializes
//!     the B tile as f32 [K, n] in a pooled scratch buffer, then the existing
//!     autotuned f32 GEMM pipeline runs on it. Scratch is reused across tiles;
//!     the frame's compute-pass ordering serializes dequant(i+1) after gemm(i).
//!
//! q8_0 path requires K % 64 == 0 (whole word-aligned block pairs per row —
//! see the kernel headers); other shapes fall back to `error.Unsupported`.

const std = @import("std");
const wgpu = @import("../wgpu.zig");
const pipelines = @import("../pipelines.zig");
const context = @import("../context.zig");
const codegen = @import("../matmul/codegen.zig");
const matmul_exec = @import("matmul.zig");
const backend_mod = @import("../../backend.zig");
const tensor_store_mod = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const c = wgpu.c;
const fns = wgpu.fns; // runtime wgpu dispatch table (functions)
const Ctx = context.Ctx;
const Frame = @import("../frame.zig").Frame;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const KernelDesc = pipelines.KernelDesc;
const Generated = codegen.Generated;

const gemv_kernel: KernelDesc = .{ .name = "matmul_nt_gemv", .wgsl = @embedFile("../kernels/matmul_nt_gemv.wgsl") };
const dequant_kernel: KernelDesc = .{ .name = "dequant", .wgsl = @embedFile("../kernels/dequant.wgsl") };

const Q8_BLOCK_ELEMS: u32 = 32;
const Q8_BLOCK_BYTES: u32 = 34;
/// Rows per workgroup in the GEMV kernel — must match RPW in the WGSL.
const GEMV_RPW: u32 = 8;
const DEQUANT_WG: u32 = 64;

/// Field order matches `struct Params` in matmul_nt_gemv.wgsl.
const GemvParams = extern struct { k: u32, n: u32, b_wpr: u32, _pad: u32 = 0, alpha: f32, beta: f32 };
/// Field order matches `struct Params` in dequant.wgsl.
const DequantParams = extern struct { n: u32, k: u32, src_wpr: u32, dst_row: u32, count: u32, _p0: u32 = 0, _p1: u32 = 0, _p2: u32 = 0 };

pub const MatmulNt = struct {
    /// Pooled f32 scratch holding one dequantized/transposed B tile [K, n_tile]
    /// for the GEMM path. Grows monotonically, freed in `deinit`.
    scratch: ?c.WGPUBuffer = null,
    scratch_cap: u64 = 0,

    pub fn deinit(self: *MatmulNt) void {
        if (self.scratch) |s| fns.wgpuBufferRelease(s);
        self.* = undefined;
    }

    fn ensureScratch(self: *MatmulNt, ctx: Ctx, bytes: u64) ExecuteProgramError!c.WGPUBuffer {
        if (self.scratch) |s| {
            if (self.scratch_cap >= bytes) return s;
            fns.wgpuBufferRelease(s);
            self.scratch = null;
            self.scratch_cap = 0;
        }
        const MiB: u64 = 1024 * 1024;
        const cap = (bytes + MiB - 1) / MiB * MiB;
        const buf = wgpu.createBuffer(ctx.gpu.device, cap, c.WGPUBufferUsage_Storage) catch return error.ExecutionFailed;
        self.scratch = buf;
        self.scratch_cap = cap;
        return buf;
    }

    pub fn exec(self: *MatmulNt, ctx: Ctx, frame: *Frame, s: executable.StepMatMulNTTiled, generated: []const Generated) ExecuteProgramError!void {
        const hs = ctx.rstore.tensorStore();
        const c_meta = hs.meta(s.c) catch return error.ExecutionFailed;
        const a_meta = hs.meta(s.a) catch return error.ExecutionFailed;
        const b_meta = hs.meta(s.b) catch return error.ExecutionFailed;

        if (c_meta.dtype != .f32 or a_meta.dtype != .f32) return error.Unsupported;
        if (b_meta.dtype != .q8_0 and b_meta.dtype != .f32) return error.Unsupported;
        if (b_meta.rank != 2) return error.Unsupported;

        const k = std.math.cast(u32, b_meta.shape[1]) orelse return error.Unsupported;
        if (k % 4 != 0) return error.Unsupported; // vec4 A reads in every path
        if (b_meta.dtype == .q8_0 and k % 64 != 0) return error.Unsupported; // block pairs
        // Compile guarantees B tiles span full K and B's N tiling matches C's
        // trailing-axis tiling; A is one tile spanning [M, K].
        if (b_meta.tile_shape[1] != b_meta.shape[1]) return error.Unsupported;
        if (context.totalTiles(a_meta) != 1) return error.Unsupported;
        const n_tiles = b_meta.tile_counts[0];
        if (n_tiles != c_meta.tile_counts[@as(usize, c_meta.rank) - 1]) return error.Unsupported;

        const da = ctx.rstore.acquireTileDeviceConstLinear(s.a, 0) catch return error.ExecutionFailed;
        defer hs.releaseConst(da.token);
        if (!context.storageBindingFits(ctx, da.len)) return error.Unsupported;
        const av = context.rowView(da.rank, da.shape_mem[0..@as(usize, da.rank)], da.strides_mem[0..@as(usize, da.rank)]) orelse return error.Unsupported;
        if (av.cols != k) return error.Unsupported;
        const m_total = av.rows;

        var nt: usize = 0;
        while (nt < n_tiles) : (nt += 1) {
            const db = ctx.rstore.acquireTileDeviceConstLinear(s.b, nt) catch return error.ExecutionFailed;
            const dc = ctx.rstore.acquireTileDeviceMutLinear(s.c, nt) catch return error.ExecutionFailed;
            defer {
                hs.releaseConst(db.token);
                hs.releaseMut(dc.token);
            }
            if (!context.storageBindingFits(ctx, db.len) or !context.storageBindingFits(ctx, dc.len)) return error.Unsupported;

            const cv = context.rowView(dc.rank, dc.shape_mem[0..@as(usize, dc.rank)], dc.strides_mem[0..@as(usize, dc.rank)]) orelse return error.Unsupported;
            if (cv.rows != m_total) return error.Unsupported;
            const n_count = cv.cols;
            if (db.shape_mem[0] < n_count) return error.Unsupported;

            // B row size in u32 words (q8_0 rows are (K/32)*34 bytes; K%64==0
            // makes that word-aligned. f32 rows are K words).
            const b_row_bytes: u32 = if (b_meta.dtype == .q8_0) (k / Q8_BLOCK_ELEMS) * Q8_BLOCK_BYTES else k * 4;
            const b_wpr = b_row_bytes / 4;
            if (@as(u64, b_row_bytes) * n_count > db.len) return error.Unsupported;

            if (m_total == 1) {
                try self.recordGemv(ctx, frame, s, da, db, dc, k, n_count, b_wpr, b_meta.dtype == .q8_0);
            } else {
                try self.recordDequantGemm(ctx, frame, s, generated, da, db, dc, av, cv, k, n_count, b_wpr, b_meta.dtype == .q8_0);
            }
        }
    }

    fn recordGemv(
        self: *MatmulNt,
        ctx: Ctx,
        frame: *Frame,
        s: executable.StepMatMulNTTiled,
        da: anytype,
        db: anytype,
        dc: anytype,
        k: u32,
        n_count: u32,
        b_wpr: u32,
        is_q8: bool,
    ) ExecuteProgramError!void {
        _ = self;
        const groups = context.ceilDiv(n_count, GEMV_RPW);
        if (groups > context.MAX_GROUPS_PER_DIM) return error.Unsupported;
        const built = try ctx.pipes.get(gemv_kernel, if (is_q8) "gemv_q8" else "gemv_f32");
        const bufs = [_]c.WGPUBuffer{
            ctx.devmem.bufferFor(da.handle).?,
            ctx.devmem.bufferFor(db.handle).?,
            ctx.devmem.bufferFor(dc.handle).?,
        };
        const sizes = [_]u64{ da.len, db.len, dc.len };
        const params: GemvParams = .{ .k = k, .n = n_count, .b_wpr = b_wpr, .alpha = s.alpha, .beta = s.beta };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups, 1, 1 });
    }

    fn recordDequantGemm(
        self: *MatmulNt,
        ctx: Ctx,
        frame: *Frame,
        s: executable.StepMatMulNTTiled,
        generated: []const Generated,
        da: anytype,
        db: anytype,
        dc: anytype,
        av: context.RowView,
        cv: context.RowView,
        k: u32,
        n_count: u32,
        b_wpr: u32,
        is_q8: bool,
    ) ExecuteProgramError!void {
        // 1) Dequant/transpose the B tile into scratch as f32 [K, n_count].
        const scratch_bytes = @as(u64, k) * n_count * 4;
        if (!context.storageBindingFits(ctx, scratch_bytes)) return error.Unsupported;
        const scratch = try self.ensureScratch(ctx, scratch_bytes);

        const count: u32 = if (is_q8) n_count * (k / 64) else n_count * k;
        const dq_built = try ctx.pipes.get(dequant_kernel, if (is_q8) "q8_nt_to_f32t" else "f32_nt_t");
        const dq_bufs = [_]c.WGPUBuffer{ ctx.devmem.bufferFor(db.handle).?, scratch };
        const dq_sizes = [_]u64{ db.len, scratch_bytes };
        const dq_params: DequantParams = .{ .n = n_count, .k = k, .src_wpr = b_wpr, .dst_row = n_count, .count = count };
        const dq_groups = @max(1, @min(context.ceilDiv(count, DEQUANT_WG), context.MAX_GROUPS_1D));
        try frame.recordCompute(dq_built, &dq_bufs, &dq_sizes, std.mem.asBytes(&dq_params), .{ dq_groups, 1, 1 });

        // 2) Run the f32 GEMM over [M, K] @ scratch[K, n_count] -> C tile.
        const a_row_bytes: isize = @intCast(@as(u64, av.row_stride) * 4);
        const scratch_row_bytes: isize = @intCast(@as(u64, n_count) * 4);
        const idx = pickGemmConfig(generated, ctx.gpu.limits, a_row_bytes, scratch_row_bytes) orelse return error.Unsupported;
        const g = generated[idx];
        const built = try ctx.pipes.get(g.desc, g.entry);

        const params: matmul_exec.MatMulParams = .{
            .m = av.rows,
            .n = n_count,
            .k = k,
            .a_row = av.row_stride,
            .b_row = n_count,
            .c_row = cv.row_stride,
            .alpha = s.alpha,
            .beta = s.beta,
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
            .{ context.ceilDiv(n_count, g.cfg.bn), context.ceilDiv(av.rows, g.cfg.bm), 1 },
        );
    }
};

/// Pick a GEMM config for the scratch-B path: bounds-checked (edge tiles are the
/// norm here) and device-eligible, preferring the shape autotune usually settles
/// on (128x128, bk16, vec4, double-buffered). No per-shape tuning — the dequant
/// pass makes re-timing every shape needlessly expensive; hook into the
/// autotune cache later if profiles say it matters.
fn pickGemmConfig(generated: []const Generated, limits: wgpu.Limits, a_row_bytes: isize, b_row_bytes: isize) ?usize {
    var fallback: ?usize = null;
    for (generated, 0..) |g, i| {
        const cfg = g.cfg;
        if (!cfg.bounds_check) continue;
        if (!matmul_exec.eligibleConfig(cfg, limits, a_row_bytes, b_row_bytes)) continue;
        if (cfg.bm == 128 and cfg.bn == 128 and cfg.bk == 16 and cfg.vec4_load and cfg.double_buffer) return i;
        if (fallback == null) fallback = i;
    }
    return fallback;
}

test "pickGemmConfig prefers the tuned 128x128 shape and falls back to scalar" {
    const cfgs = [_]Generated{
        .{ .desc = .{ .name = "a", .wgsl = "" }, .entry = "a", .cfg = .{ .bm = 128, .bn = 128, .bk = 8, .tm = 8, .tn = 8, .vec4_load = false } },
        .{ .desc = .{ .name = "b", .wgsl = "" }, .entry = "b", .cfg = .{ .bm = 128, .bn = 128, .bk = 16, .tm = 8, .tn = 8, .vec4_load = true, .double_buffer = true } },
    };
    const limits: wgpu.Limits = .{ .max_shared_bytes = 48 * 1024, .max_invocations = 256 };
    // Aligned strides: the vec4 128x128 bk16 db config wins.
    try std.testing.expectEqual(@as(?usize, 1), pickGemmConfig(&cfgs, limits, 512, 512));
    // Misaligned B stride: only the scalar config is eligible.
    try std.testing.expectEqual(@as(?usize, 0), pickGemmConfig(&cfgs, limits, 512, 500));
}
