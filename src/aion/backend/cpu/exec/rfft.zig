// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// RFFT op execution: real FFT over the last (power-of-two) dimension.
//
// Input `x[.., n_fft]` (f32) → output `[.., n_fft+2]` packed complex (real bins
// in `[0..bins)`, imaginary in `[bins..2*bins)`, `bins = n_fft/2+1`). All
// leading dimensions are treated as an independent batch of frames. Frames are
// processed in groups of `kernels.lanes` so the kernel can vectorize across
// them.

const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");
const fft = @import("../kernels/fft.zig");
const fft_registry = @import("../registry/fft_registry.zig");
const exec_utils = @import("utils.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

const MAX_RANK: usize = 8;

pub fn execRFFT(
    allocator: std.mem.Allocator,
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    kernels: fft_registry.FftKernels,
    plan: *const fft.Plan,
    s: executable.StepRFFT,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    // v1: sequential over frame-groups; the kernel vectorizes across the lanes
    // within each group. Frame-group threading is a future optimization.
    _ = pool;
    _ = thread_count;

    const out_meta = try store.meta(s.out);
    const x_meta = try store.meta(s.x);

    if (out_meta.dtype != .f32 or x_meta.dtype != .f32) return BackendError.Unsupported;
    const rank: usize = @as(usize, x_meta.rank);
    if (rank == 0 or rank > MAX_RANK) return BackendError.InvalidArgument;
    if (@as(usize, out_meta.rank) != rank) return BackendError.InvalidArgument;

    const n_fft: usize = s.n_fft;
    if (n_fft < 4 or (n_fft & (n_fft - 1)) != 0) return BackendError.InvalidArgument;
    if (x_meta.shape[rank - 1] != n_fft) return BackendError.InvalidArgument;
    const bins: usize = n_fft / 2 + 1;
    if (out_meta.shape[rank - 1] != 2 * bins) return BackendError.InvalidArgument;

    const leading_rank: usize = rank - 1;
    var frames: usize = 1;
    var d: usize = 0;
    while (d < leading_rank) : (d += 1) {
        if (out_meta.shape[d] != x_meta.shape[d]) return BackendError.InvalidArgument;
        frames *= x_meta.shape[d];
    }
    if (frames == 0) return; // nothing to do (empty leading dim)
    if (plan.n_fft != n_fft) return BackendError.InvalidArgument;

    const lanes: usize = kernels.lanes;
    const in_buf = allocator.alloc(f32, lanes * n_fft) catch return BackendError.ExecutionFailed;
    defer allocator.free(in_buf);
    const out_buf = allocator.alloc(f32, lanes * 2 * bins) catch return BackendError.ExecutionFailed;
    defer allocator.free(out_buf);
    const scratch = allocator.alignedAlloc(u8, .@"64", fft.scratchBytes(plan, lanes)) catch return BackendError.ExecutionFailed;
    defer allocator.free(scratch);

    // Packed strides over the leading (frame) dims for linear<->coord decoding.
    var lead_strides: [MAX_RANK]usize = @splat(0);
    if (leading_rank > 0) {
        var stride: usize = 1;
        var dd: usize = leading_rank;
        while (dd > 0) : (dd -= 1) {
            lead_strides[dd - 1] = stride;
            stride *= x_meta.shape[dd - 1];
        }
    }

    var x_cache: exec_utils.TileCacheConstND = .{};
    defer x_cache.deinit(store);
    var out_cache: exec_utils.TileCacheMutND = .{};
    defer out_cache.deinit(store);

    var coords: [MAX_RANK]usize = @splat(0);

    var group_start: usize = 0;
    while (group_start < frames) : (group_start += lanes) {
        const count: usize = @min(lanes, frames - group_start);

        // Gather: windowed (here just raw) frames into the contiguous in_buf.
        var l: usize = 0;
        while (l < count) : (l += 1) {
            const f: usize = group_start + l;
            decodeLeading(f, lead_strides[0..leading_rank], coords[0..leading_rank]);
            var p: usize = 0;
            while (p < n_fft) : (p += 1) {
                coords[leading_rank] = p;
                in_buf[l * n_fft + p] = try exec_utils.readScalarF32At(store, x_meta, s.x, coords[0..rank], &x_cache);
            }
        }

        kernels.process_group(plan, in_buf, out_buf, count, scratch) catch return BackendError.ExecutionFailed;

        // Scatter packed complex bins to the output.
        l = 0;
        while (l < count) : (l += 1) {
            const f: usize = group_start + l;
            decodeLeading(f, lead_strides[0..leading_rank], coords[0..leading_rank]);
            var k: usize = 0;
            while (k < 2 * bins) : (k += 1) {
                coords[leading_rank] = k;
                try exec_utils.writeScalarFromF32At(store, out_meta, s.out, coords[0..rank], out_buf[l * 2 * bins + k], &out_cache);
            }
        }
    }
}

fn decodeLeading(linear: usize, strides: []const usize, out: []usize) void {
    var rem: usize = linear;
    for (strides, 0..) |stride, i| {
        const v: usize = rem / stride;
        out[i] = v;
        rem -= v * stride;
    }
}
