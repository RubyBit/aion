// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");
const exec_utils = @import("utils.zig");

const matmul_registry = @import("../registry/matmul_registry.zig");
const quant_matmul_registry = @import("../registry/quant_matmul_registry.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const DType = types.DType;

/// q8_0 block: 2-byte f16 scale + 32 i8 values, 34 bytes total.
const Q8_0_BLOCK_ELEMS: usize = 32;
const Q8_0_BLOCK_BYTES: usize = 34;

pub const MatMulNtExecCtx = struct {
    matmul_f32: matmul_registry.F32Kernels,
    matmul_qx0: quant_matmul_registry.QuantKernels,
};

/// C[m, n] = alpha * sum_k A[m, k] * B[n, k]  +  beta * C[m, n]
///
/// Layout:
/// - A: f32, trailing axis K. Leading axes collapse to a flat M = prod(A.shape[:-1]).
/// - B: q8_0 `[N, K]` (quant_axis == 1, one row = K/32 contiguous q8_0 blocks) or f32 `[N, K]`.
/// - C: f32, trailing axis N.
///
/// Actual compute is delegated to the kernel registry:
/// - Q8_0 → `quant_matmul.matmulNtQ8_0` via `quant_matmul_registry.QuantKernels.matmul_nt_q8_0`
/// - F32  → `matmul_tuned.matmulNtF32` via `matmul_registry.F32Kernels.matmul_nt_f32`
///
/// Parallelism here is over N tiles (B's axis-0 tiling must match C's last-axis tiling,
/// enforced at compile time). Each worker handles a contiguous range of N tiles.
pub fn execMatMulNTTiled(
    ctx: *const MatMulNtExecCtx,
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepMatMulNTTiled,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const c_meta = try store.meta(s.c);
    const a_meta = try store.meta(s.a);
    const b_meta = try store.meta(s.b);

    if (b_meta.rank != 2) return BackendError.InvalidArgument;
    if (a_meta.rank != c_meta.rank) return BackendError.InvalidArgument;
    if (c_meta.dtype != .f32 or a_meta.dtype != .f32) return BackendError.InvalidArgument;
    if (b_meta.dtype != .q8_0 and b_meta.dtype != .f32) return BackendError.InvalidArgument;

    const k: usize = b_meta.shape[1];
    const n_total: usize = b_meta.shape[0];
    if (a_meta.shape[a_meta.rank - 1] != k) return BackendError.InvalidArgument;
    if (c_meta.shape[c_meta.rank - 1] != n_total) return BackendError.InvalidArgument;
    if (b_meta.dtype == .q8_0 and (k % Q8_0_BLOCK_ELEMS) != 0) return BackendError.InvalidArgument;

    // `m_total` is the flattened product of leading C/A axes.
    var m_total: usize = 1;
    var d: usize = 0;
    while (d + 1 < @as(usize, c_meta.rank)) : (d += 1) {
        if (a_meta.shape[d] != c_meta.shape[d]) return BackendError.InvalidArgument;
        m_total = std.math.mul(usize, m_total, c_meta.shape[d]) catch return BackendError.InvalidArgument;
    }

    // A is held in a single tile spanning all (batch,seq) rows × K. Acquire once.
    var a_tile = try store.acquireTileConstLinear(s.a, 0);
    defer store.releaseConst(a_tile.token);
    const a_view = a_tile.bufferView();

    // A must occupy a single tile containing all M×K f32 elements (the compile step
    // retiles to `[...leading_full, K]` which is a single tile).
    const expected_a_bytes: usize = m_total * k * @sizeOf(f32);
    if (a_view.bytes.len < expected_a_bytes) return BackendError.InvalidArgument;
    const a_ptr: [*]align(1) const f32 = @ptrCast(a_view.bytes.ptr);

    // C's last-axis tile size is aligned to B's axis-0 tile size by the compile step,
    // so B-tile `nt` covers C's N range [nt*tile_size .. min((nt+1)*tile_size, N)].
    if (b_meta.tile_counts[0] != c_meta.tile_counts[c_meta.rank - 1]) return BackendError.InvalidArgument;
    if (b_meta.tile_shape[0] != c_meta.tile_shape[c_meta.rank - 1]) return BackendError.InvalidArgument;

    const tile_total: usize = c_meta.tile_counts[c_meta.rank - 1];
    const n_tile_size: usize = c_meta.tile_shape[c_meta.rank - 1];

    const Runner = struct {
        store: tensor_store.TensorStore,
        c: tensor_store.TensorId,
        b: tensor_store.TensorId,
        a_ptr: [*]align(1) const f32,
        m_total: usize,
        k: usize,
        n_total: usize,
        n_tile_size: usize,
        alpha: f32,
        beta: f32,
        b_dtype: DType,
        matmul_nt_q8_0: quant_matmul_registry.MatMulNtQ8_0Fn,
        matmul_nt_f32: matmul_registry.MatMulNtF32Fn,

        fn runRange(self: *@This(), start: usize, end: usize) ExecuteProgramError!void {
            var nt: usize = start;
            while (nt < end) : (nt += 1) {
                var c_tile = try self.store.acquireTileMutLinear(self.c, nt);
                defer self.store.releaseMut(c_tile.token);
                const c_view = c_tile.bufferView();
                if ((c_view.bytes.len % @sizeOf(f32)) != 0) return BackendError.InvalidArgument;

                var b_tile = try self.store.acquireTileConstLinear(self.b, nt);
                defer self.store.releaseConst(b_tile.token);
                const b_view = b_tile.bufferView();

                const n_start: usize = nt * self.n_tile_size;
                const n_count: usize = @min(self.n_tile_size, self.n_total - n_start);
                const c_ptr: [*]align(1) f32 = @ptrCast(c_view.bytes.ptr);
                if (c_view.bytes.len < self.m_total * n_count * @sizeOf(f32)) return BackendError.InvalidArgument;

                switch (self.b_dtype) {
                    .q8_0 => {
                        const expected_b_tile_bytes: usize = n_count * (self.k / Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES;
                        if (b_view.bytes.len < expected_b_tile_bytes) return BackendError.InvalidArgument;
                        try self.matmul_nt_q8_0(
                            self.a_ptr,
                            b_view.bytes.ptr,
                            c_ptr,
                            self.m_total,
                            self.k,
                            self.n_total,
                            n_start,
                            n_count,
                            self.alpha,
                            self.beta,
                        );
                    },
                    .f32 => {
                        const expected_b_tile_bytes: usize = n_count * self.k * @sizeOf(f32);
                        if (b_view.bytes.len < expected_b_tile_bytes) return BackendError.InvalidArgument;
                        const b_ptr: [*]align(1) const f32 = @ptrCast(b_view.bytes.ptr);
                        try self.matmul_nt_f32(
                            self.a_ptr,
                            b_ptr,
                            c_ptr,
                            self.m_total,
                            self.k,
                            self.n_total,
                            n_start,
                            n_count,
                            self.alpha,
                            self.beta,
                        );
                    },
                    else => return BackendError.Unsupported,
                }
            }
        }
    };

    var runner: Runner = .{
        .store = store,
        .c = s.c,
        .b = s.b,
        .a_ptr = a_ptr,
        .m_total = m_total,
        .k = k,
        .n_total = n_total,
        .n_tile_size = n_tile_size,
        .alpha = s.alpha,
        .beta = s.beta,
        .b_dtype = b_meta.dtype,
        .matmul_nt_q8_0 = ctx.matmul_qx0.matmul_nt_q8_0,
        .matmul_nt_f32 = ctx.matmul_f32.matmul_nt_f32,
    };
    if (pool) |p| {
        const tile_bytes: usize = exec_utils.tileByteSize(c_meta);
        const min_total_bytes: usize = 256 * 1024;
        if (exec_utils.shouldParallelTiles(thread_count, tile_total, tile_bytes, min_total_bytes)) {
            const Task = struct {
                runner: *Runner,
                stop: std.atomic.Value(bool) = .init(false),
                err_mutex: std.Io.Mutex = .init,
                err_any: ?anyerror = null,

                fn fail(t: *@This(), err: anyerror) void {
                    if (t.stop.swap(true, .acq_rel)) return;
                    std.Io.Threaded.mutexLock(&t.err_mutex);
                    defer std.Io.Threaded.mutexUnlock(&t.err_mutex);
                    if (t.err_any == null) t.err_any = err;
                }

                fn runTiles(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
                    _ = tid;
                    const t: *@This() = @ptrCast(@alignCast(ctx_any));
                    if (start >= end) return;
                    if (t.stop.load(.acquire)) return;
                    t.runner.runRange(start, end) catch |e| {
                        t.fail(e);
                        return;
                    };
                }
            };

            var task: Task = .{ .runner = &runner };
            var grain: usize = if (tile_bytes == 0) 1 else @max(@as(usize, 1), min_total_bytes / tile_bytes);
            if (grain > tile_total) grain = tile_total;
            p.parallelForAny(@ptrCast(&task), tile_total, grain, Task.runTiles);
            if (task.err_any) |e| return @errorCast(e);
            return;
        }
    }

    try runner.runRange(0, tile_total);
}
