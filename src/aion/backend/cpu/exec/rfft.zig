// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// RFFT op execution: real FFT over the last (power-of-two) dimension.
//
// Input `x[.., n_fft]` (f32) → output `[.., n_fft+2]` packed complex (real bins
// in `[0..bins)`, imaginary in `[bins..2*bins)`, `bins = n_fft/2+1`). All
// leading dimensions are treated as an independent batch of frames. Frames are
// processed in groups of `kernels.lanes` so the kernel can vectorize across
// them, and groups split across the pool.

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

    var frames: usize = 1;
    for (0..rank - 1) |d| {
        if (out_meta.shape[d] != x_meta.shape[d]) return BackendError.InvalidArgument;
        frames *= x_meta.shape[d];
    }
    if (frames == 0) return;
    if (plan.n_fft != n_fft) return BackendError.InvalidArgument;

    const x = try store.acquireConst(s.x);
    defer store.releaseConst(x.token);
    const out = try store.acquireMut(s.out);
    defer store.releaseMut(out.token);

    // Frames are consecutive rows of both tensors, so a group of them is one run
    // the kernel reads and writes in place.
    const Ctx = struct {
        allocator: std.mem.Allocator,
        kernels: fft_registry.FftKernels,
        plan: *const fft.Plan,
        frames: usize,
        n_fft: usize,
        out_row: usize,
        x: []align(1) const f32,
        out: []align(1) f32,

        fn run(c: @This(), lo: usize, hi: usize, _: usize) BackendError!void {
            const lanes = c.kernels.lanes;
            const scratch = c.allocator.alignedAlloc(u8, .@"64", fft.scratchBytes(c.plan, lanes)) catch return BackendError.ExecutionFailed;
            defer c.allocator.free(scratch);
            for (lo..hi) |group| {
                const f0 = group * lanes;
                const count = @min(lanes, c.frames - f0);
                const in: []const f32 = @alignCast(c.x[f0 * c.n_fft ..][0 .. count * c.n_fft]);
                const dst: []f32 = @alignCast(c.out[f0 * c.out_row ..][0 .. count * c.out_row]);
                c.kernels.process_group(c.plan, in, dst, count, scratch) catch return BackendError.ExecutionFailed;
            }
        }
    };
    const ctx: Ctx = .{
        .allocator = allocator,
        .kernels = kernels,
        .plan = plan,
        .frames = frames,
        .n_fft = n_fft,
        .out_row = 2 * bins,
        .x = std.mem.bytesAsSlice(f32, x.bytes),
        .out = std.mem.bytesAsSlice(f32, out.bytes),
    };
    const groups = std.math.divCeil(usize, frames, kernels.lanes) catch unreachable;
    return exec_utils.parallelRange(BackendError, pool, thread_count, groups, kernels.lanes * n_fft * 4 * @sizeOf(f32), ctx, Ctx.run);
}
