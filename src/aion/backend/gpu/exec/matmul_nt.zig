// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! MatMulNT execution for the GPU backend:
//!   C[m, n] = alpha * sum_k A[m, k] * B[n, k]  +  beta * C[m, n]
//! with A f32 [.., K] and B [N, K] in q8_0 or f32 (same contract as the CPU
//! executor in `backend/cpu/exec/matmul_nt.zig`). A B past the binding limit is
//! chunked along N; each chunk writes its columns of the one C.
//!
//! Two regimes, split on M:
//!   - M == 1 (decode): bandwidth-bound matvec — `kernels/matmul_nt_gemv.wgsl`
//!     reads B exactly once (dequant-in-registers for q8_0).
//!   - M > 1 (prefill): compute-bound — a dequant/transpose pass materializes
//!     the B chunk as f32 [K, n] in a pooled scratch buffer, then the existing
//!     autotuned f32 GEMM pipeline runs on it. Scratch is reused across chunks;
//!     the frame's compute-pass ordering serializes dequant(i+1) after gemm(i).
//!
//! q8_0 path requires K % 64 == 0 (whole word-aligned block pairs per row —
//! see the kernel headers); other shapes fall back to `error.Unsupported`.

const std = @import("std");
const types = @import("../../types.zig");
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
/// Rows per group of a `lanes32` weight — must match L_W in the WGSL.
const LANES_W: u32 = 32;
/// Below this many workgroups, a `lanes32` GEMV splits K 32 ways instead of 8,
/// so a narrow projection still occupies the GPU (whose core count WebGPU does
/// not expose).
const LANES_WIDE_BELOW_GROUPS: u32 = 64;

/// The grouping these kernels read fastest (`types.QuantBlockOrder`), which the
/// device reports so its weights are laid out for it.
pub const block_order: types.QuantBlockOrder = .lanes32;

/// How B's bytes are laid out, which picks the kernels that read it.
const BForm = enum { q8_pairs, q8_lanes32, f32 };
const DEQUANT_WG: u32 = 64;

/// Field order matches `struct Params` in matmul_nt_gemv.wgsl.
const GemvParams = extern struct { k: u32, n: u32, b_wpr: u32, c_off: u32, alpha: f32, beta: f32 };
/// Field order matches `struct Params` in dequant.wgsl.
const DequantParams = extern struct { n: u32, k: u32, src_wpr: u32, dst_row: u32, count: u32, _p0: u32 = 0, _p1: u32 = 0, _p2: u32 = 0 };

pub const MatmulNt = struct {
    /// Pooled f32 scratch holding one dequantized/transposed B chunk [K, n]
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

    pub fn exec(self: *MatmulNt, ctx: Ctx, frame: *Frame, s: executable.StepMatMulNT, generated: []const Generated) ExecuteProgramError!void {
        const hs = ctx.store;
        const c_meta = hs.meta(s.c) catch return error.ExecutionFailed;
        const a_meta = hs.meta(s.a) catch return error.ExecutionFailed;
        const b_meta = hs.meta(s.b) catch return error.ExecutionFailed;

        if (c_meta.dtype != .f32 or a_meta.dtype != .f32) return error.Unsupported;
        if (b_meta.dtype != .q8_0 and b_meta.dtype != .f32) return error.Unsupported;
        if (b_meta.rank != 2) return error.Unsupported;

        const k = std.math.cast(u32, b_meta.shape[1]) orelse return error.Unsupported;
        if (k % 4 != 0) return error.Unsupported; // vec4 A reads in every path
        const lanes = b_meta.block_order == .lanes32;
        if (b_meta.block_order != .row_major and !lanes) return error.Unsupported;
        // Row-major q8 rows are walked in word-aligned block pairs; `lanes32`
        // segments are word-aligned whatever K is.
        if (b_meta.dtype == .q8_0 and k % (if (lanes) Q8_BLOCK_ELEMS else 64) != 0) return error.Unsupported;
        if (a_meta.chunks != 1 or c_meta.chunks != 1) return error.Unsupported;
        const a_rank: usize = a_meta.rank;
        const c_rank: usize = c_meta.rank;
        if (a_rank == 0 or c_rank == 0 or a_meta.shape[a_rank - 1] != k) return error.Unsupported;
        var m_rows: usize = 1;
        for (a_meta.shape[0 .. a_rank - 1]) |d| m_rows *= d;
        var c_rows: usize = 1;
        for (c_meta.shape[0 .. c_rank - 1]) |d| c_rows *= d;
        const n_total = b_meta.shape[0];
        if (c_rows != m_rows or c_meta.shape[c_rank - 1] != n_total) return error.Unsupported;
        const m_total = std.math.cast(u32, m_rows) orelse return error.Unsupported;
        const c_row = std.math.cast(u32, n_total) orelse return error.Unsupported;
        if (m_total == 0) return;

        const da = ctx.store.acquireConst(s.a) catch return error.ExecutionFailed;
        defer hs.releaseConst(da.token);
        const dc = ctx.store.acquireMut(s.c) catch return error.ExecutionFailed;
        defer hs.releaseMut(dc.token);
        if (!context.storageBindingFits(ctx, da.len) or !context.storageBindingFits(ctx, dc.len)) return error.Unsupported;

        var n0: usize = 0;
        var chunk: usize = 0;
        const chunks = b_meta.chunks;
        while (chunk < chunks) : (chunk += 1) {
            const db = ctx.store.acquireChunkConst(s.b, chunk) catch return error.ExecutionFailed;
            defer hs.releaseConst(db.token);
            if (!context.storageBindingFits(ctx, db.len)) return error.Unsupported;
            const n_count = std.math.cast(u32, db.rows) orelse return error.Unsupported;
            const c_off = std.math.cast(u32, n0) orelse return error.Unsupported;
            n0 += n_count;

            // B row size in u32 words (q8_0 rows are (K/32)*34 bytes; K%64==0
            // makes that word-aligned. f32 rows are K words).
            const b_row_bytes: u32 = if (b_meta.dtype == .q8_0) (k / Q8_BLOCK_ELEMS) * Q8_BLOCK_BYTES else k * 4;
            const b_wpr = b_row_bytes / 4;
            if (@as(u64, b_row_bytes) * n_count > db.len) return error.Unsupported;

            if (lanes and n_count % LANES_W != 0) return error.Unsupported;
            const form: BForm = if (b_meta.dtype != .q8_0) .f32 else if (lanes) .q8_lanes32 else .q8_pairs;
            if (m_total == 1) {
                try self.recordGemv(ctx, frame, s, da, db, dc, k, n_count, b_wpr, c_off, form);
            } else {
                try self.recordDequantGemm(ctx, frame, s, generated, da, db, dc, m_total, c_row, c_off, k, n_count, b_wpr, form);
            }
        }
        if (n0 != n_total) return error.Unsupported; // chunks did not cover B
    }

    fn recordGemv(
        self: *MatmulNt,
        ctx: Ctx,
        frame: *Frame,
        s: executable.StepMatMulNT,
        da: anytype,
        db: anytype,
        dc: anytype,
        k: u32,
        n_count: u32,
        b_wpr: u32,
        c_off: u32,
        form: BForm,
    ) ExecuteProgramError!void {
        _ = self;
        const groups = switch (form) {
            .q8_lanes32 => n_count / LANES_W,
            else => context.ceilDiv(n_count, GEMV_RPW),
        };
        if (groups > context.MAX_GROUPS_PER_DIM) return error.Unsupported;
        const entry: [:0]const u8 = switch (form) {
            .q8_pairs => "gemv_q8",
            .q8_lanes32 => if (groups < LANES_WIDE_BELOW_GROUPS) "gemv_q8_lanes32_wide" else "gemv_q8_lanes32",
            .f32 => "gemv_f32",
        };
        const built = try ctx.pipes.get(gemv_kernel, entry);
        const bufs = [_]c.WGPUBuffer{
            ctx.devmem.bufferFor(da.handle).?,
            ctx.devmem.bufferFor(db.handle).?,
            ctx.devmem.bufferFor(dc.handle).?,
        };
        const sizes = [_]u64{ da.len, db.len, dc.len };
        const params: GemvParams = .{ .k = k, .n = n_count, .b_wpr = b_wpr, .c_off = c_off, .alpha = s.alpha, .beta = s.beta };
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups, 1, 1 });
    }

    fn recordDequantGemm(
        self: *MatmulNt,
        ctx: Ctx,
        frame: *Frame,
        s: executable.StepMatMulNT,
        generated: []const Generated,
        da: anytype,
        db: anytype,
        dc: anytype,
        m: u32,
        c_row: u32,
        c_off: u32,
        k: u32,
        n_count: u32,
        b_wpr: u32,
        form: BForm,
    ) ExecuteProgramError!void {
        // 1) Dequant/transpose the B chunk into scratch as f32 [K, n_count].
        const scratch_bytes = @as(u64, k) * n_count * 4;
        if (!context.storageBindingFits(ctx, scratch_bytes)) return error.Unsupported;
        const scratch = try self.ensureScratch(ctx, scratch_bytes);

        const count: u32 = switch (form) {
            .q8_pairs => n_count * (k / 64),
            .q8_lanes32 => n_count * (k / Q8_BLOCK_ELEMS),
            .f32 => n_count * k,
        };
        const dq_built = try ctx.pipes.get(dequant_kernel, switch (form) {
            .q8_pairs => "q8_nt_to_f32t",
            .q8_lanes32 => "q8_lanes32_to_f32t",
            .f32 => "f32_nt_t",
        });
        const dq_bufs = [_]c.WGPUBuffer{ ctx.devmem.bufferFor(db.handle).?, scratch };
        const dq_sizes = [_]u64{ db.len, scratch_bytes };
        const dq_params: DequantParams = .{ .n = n_count, .k = k, .src_wpr = b_wpr, .dst_row = n_count, .count = count };
        const dq_groups = @max(1, @min(context.ceilDiv(count, DEQUANT_WG), context.MAX_GROUPS_1D));
        try frame.recordCompute(dq_built, &dq_bufs, &dq_sizes, std.mem.asBytes(&dq_params), .{ dq_groups, 1, 1 });

        // 2) Run the f32 GEMM over [M, K] @ scratch[K, n_count] -> C[:, c_off ..].
        const a_row_bytes: isize = @intCast(@as(u64, k) * 4);
        const scratch_row_bytes: isize = @intCast(@as(u64, n_count) * 4);
        const idx = pickGemmConfig(generated, ctx.gpu.limits, a_row_bytes, scratch_row_bytes) orelse return error.Unsupported;
        const g = generated[idx];
        const built = try ctx.pipes.get(g.desc, g.entry);

        const params: matmul_exec.MatMulParams = .{
            .m = m,
            .n = n_count,
            .k = k,
            .a_row = k,
            .b_row = n_count,
            .c_row = c_row,
            .alpha = s.alpha,
            .beta = s.beta,
            .c_off = c_off,
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
            .{ context.ceilDiv(n_count, g.cfg.bn), context.ceilDiv(m, g.cfg.bm), 1 },
        );
    }
};

/// Pick a GEMM config for the scratch-B path: bounds-checked (edge blocks are the
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
