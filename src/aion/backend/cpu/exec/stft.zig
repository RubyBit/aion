// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// STFT op execution: frame the signal, apply the window, run a real FFT.
//
// Inputs: signal `[batch, samples]` (f32), window `[n_fft]` (f32). Output:
// `[batch, num_frames, n_fft+2]` packed complex (same layout as RFFT). Framing
// is done on the fly into a small per-group buffer — the overlapped framed
// signal is never materialized in full.
// Frame groups are processed in parallel across the thread pool.

const std = @import("std");
const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");
const simd = @import("../kernels/simd.zig");
const fft = @import("../kernels/fft.zig");
const fft_registry = @import("../registry/fft_registry.zig");
const exec_utils = @import("utils.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

/// Reflect a (possibly out-of-range) sample index into `[0, n)` without
/// repeating the edge sample — matching torch/NeMo 'reflect' padding.
inline fn reflectIndex(i: isize, n: usize) usize {
    if (n <= 1) return 0;
    const period: isize = 2 * (@as(isize, @intCast(n)) - 1);
    var m: isize = @mod(i, period); // @mod yields [0, period)
    const ni: isize = @intCast(n);
    if (m >= ni) m = period - m;
    return @intCast(m);
}

pub fn execSTFT(
    allocator: std.mem.Allocator,
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    kernels: fft_registry.FftKernels,
    plan: *const fft.Plan,
    s: executable.StepSTFT,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    const out_meta = try store.meta(s.out);
    const sig_meta = try store.meta(s.signal);
    const win_meta = try store.meta(s.window);

    if (out_meta.dtype != .f32 or sig_meta.dtype != .f32 or win_meta.dtype != .f32) return BackendError.Unsupported;
    if (sig_meta.rank != 2 or win_meta.rank != 1 or out_meta.rank != 3) return BackendError.InvalidArgument;

    const n_fft: usize = s.n_fft;
    const hop: usize = s.hop_length;
    if (n_fft < 4 or (n_fft & (n_fft - 1)) != 0 or hop == 0) return BackendError.InvalidArgument;
    if (win_meta.shape[0] != n_fft) return BackendError.InvalidArgument;

    const bins: usize = n_fft / 2 + 1;
    const batch: usize = sig_meta.shape[0];
    const samples: usize = sig_meta.shape[1];
    const num_frames: usize = s.num_frames;

    if (out_meta.shape[0] != batch) return BackendError.InvalidArgument;
    if (out_meta.shape[1] != num_frames) return BackendError.InvalidArgument;
    if (out_meta.shape[2] != 2 * bins) return BackendError.InvalidArgument;
    if (batch == 0 or num_frames == 0) return;
    if (plan.n_fft != n_fft) return BackendError.InvalidArgument;

    const sig_view = try store.acquireConst(s.signal);
    defer store.releaseConst(sig_view.token);
    const win_view = try store.acquireConst(s.window);
    defer store.releaseConst(win_view.token);
    const out_view = try store.acquireMut(s.out);
    defer store.releaseMut(out_view.token);

    // Tensor backing is 64-byte aligned, so natural-alignment casts are safe.
    const sig: []const f32 = @alignCast(simd.bytesAsSliceConstUnaligned(f32, sig_view.bytes));
    const win: []const f32 = @alignCast(simd.bytesAsSliceConstUnaligned(f32, win_view.bytes));
    const out: []f32 = @alignCast(simd.bytesAsSliceMutUnaligned(f32, out_view.bytes));
    if (sig.len < batch * samples or win.len < n_fft or out.len < batch * num_frames * 2 * bins) return BackendError.InvalidArgument;
    return execFast(allocator, pool, thread_count, kernels, plan, s, sig, win, out, batch, samples, num_frames, n_fft, bins);
}

const FastTask = struct {
    plan: *const fft.Plan,
    kernels: fft_registry.FftKernels,
    sig: []const f32,
    win: []const f32,
    out: []f32,
    in_base: []f32,
    scratch_base: []u8,
    in_stride: usize,
    scratch_stride: usize,
    lanes: usize,
    n_fft: usize,
    bins: usize,
    hop: usize,
    pad: isize,
    samples: usize,
    num_frames: usize,
    num_groups: usize,
    center: bool,

    fn runItems(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) ExecuteProgramError!void {
        const t: *FastTask = @ptrCast(@alignCast(ctx_any));
        const n_fft = t.n_fft;
        const lanes = t.lanes;
        const two_bins = 2 * t.bins;

        const in_buf: []f32 = t.in_base[tid * t.in_stride ..][0 .. lanes * n_fft];
        const scratch: []u8 = t.scratch_base[tid * t.scratch_stride ..][0..t.scratch_stride];

        var item: usize = start;
        while (item < end) : (item += 1) {
            const b: usize = item / t.num_groups;
            const group: usize = item % t.num_groups;
            const gstart: usize = group * lanes;
            const count: usize = @min(lanes, t.num_frames - gstart);

            const sig_row: []const f32 = t.sig[b * t.samples ..][0..t.samples];

            var l: usize = 0;
            while (l < count) : (l += 1) {
                const frame: usize = gstart + l;
                const origin: isize = @as(isize, @intCast(frame * t.hop)) - t.pad;
                const dst: []f32 = in_buf[l * n_fft ..][0..n_fft];

                // Whole frame in range → tight contiguous windowed copy.
                if (origin >= 0 and origin + @as(isize, @intCast(n_fft)) <= @as(isize, @intCast(t.samples))) {
                    const base: usize = @intCast(origin);
                    var j: usize = 0;
                    while (j < n_fft) : (j += 1) dst[j] = sig_row[base + j] * t.win[j];
                } else {
                    var j: usize = 0;
                    while (j < n_fft) : (j += 1) {
                        const idx: isize = origin + @as(isize, @intCast(j));
                        var sample: f32 = 0.0;
                        if (idx >= 0 and idx < @as(isize, @intCast(t.samples))) {
                            sample = sig_row[@intCast(idx)];
                        } else if (t.center) {
                            sample = sig_row[reflectIndex(idx, t.samples)];
                        }
                        dst[j] = sample * t.win[j];
                    }
                }
            }

            const out_off: usize = (b * t.num_frames + gstart) * two_bins;
            const out_group: []f32 = t.out[out_off ..][0 .. count * two_bins];
            t.kernels.process_group(t.plan, in_buf, out_group, count, scratch) catch |err| return switch (err) { error.OutOfMemory => error.OutOfMemory, else => error.InvalidArgument };
        }
    }
};

fn execFast(
    allocator: std.mem.Allocator,
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    kernels: fft_registry.FftKernels,
    plan: *const fft.Plan,
    s: executable.StepSTFT,
    sig: []const f32,
    win: []const f32,
    out: []f32,
    batch: usize,
    samples: usize,
    num_frames: usize,
    n_fft: usize,
    bins: usize,
) ExecuteProgramError!void {
    const lanes: usize = kernels.lanes;
    const num_groups: usize = (num_frames + lanes - 1) / lanes;
    const total_items: usize = batch * num_groups;

    // Thread-dispatch (futex wake/sync) costs ~tens of microseconds; a single
    // STFT call is only a few microseconds of compute per group. Only parallelize
    // when each worker gets enough groups to amortize that — i.e. large/offline
    // buffers. Streaming-sized buffers (a handful of groups) run sequentially.
    const min_items_per_thread: usize = 4;
    const want_parallel: bool = pool != null and thread_count > 1 and
        total_items >= min_items_per_thread * thread_count;
    const n_threads: usize = if (want_parallel) @min(thread_count, total_items) else 1;

    const in_stride: usize = lanes * n_fft;
    const scratch_stride: usize = fft.scratchBytes(plan, lanes);

    const in_base: []f32 = allocator.alloc(f32, n_threads * in_stride) catch return BackendError.ExecutionFailed;
    defer allocator.free(in_base);
    const scratch_base = allocator.alignedAlloc(u8, .@"64", n_threads * scratch_stride) catch return BackendError.ExecutionFailed;
    defer allocator.free(scratch_base);

    var task: FastTask = .{
        .plan = plan,
        .kernels = kernels,
        .sig = sig,
        .win = win,
        .out = out,
        .in_base = in_base,
        .scratch_base = scratch_base,
        .in_stride = in_stride,
        .scratch_stride = scratch_stride,
        .lanes = lanes,
        .n_fft = n_fft,
        .bins = bins,
        .hop = s.hop_length,
        .pad = if (s.center) @intCast(n_fft / 2) else 0,
        .samples = samples,
        .num_frames = num_frames,
        .num_groups = num_groups,
        .center = s.center,
    };

    if (want_parallel and n_threads > 1) {
        try pool.?.parallelForFallible(ExecuteProgramError, @ptrCast(&task), total_items, 1, FastTask.runItems);
        return;
    }
    try FastTask.runItems(@ptrCast(&task), 0, total_items, 0);
}

