// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// STFT op execution: frame the signal, apply the window, run a real FFT.
//
// Inputs: signal `[batch, samples]` (f32), window `[n_fft]` (f32). Output:
// `[batch, num_frames, n_fft+2]` packed complex (same layout as RFFT). Framing
// is done on the fly into a small per-group buffer — the overlapped framed
// signal is never materialized in full.
//
// Fast path (the common case: small/single-tile, packed-contiguous tensors):
// the raw f32 tile buffers are acquired once and indexed directly, and frame
// groups are processed in parallel across the thread pool. A scalar
// tile-by-tile fallback handles arbitrary tilings.

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

/// True when every dim is a single tile and the tile is packed row-major f32,
/// so the whole tensor is one contiguous f32 buffer indexable by flat offset.
fn packedContiguousF32(meta: tensor_store.TensorMeta, dtype: types.DType, layout: types.Layout) bool {
    const rank: usize = @as(usize, meta.rank);
    if (dtype != .f32) return false;
    for (meta.tile_counts) |tc| {
        if (tc != 1) return false;
    }
    if (@as(usize, layout.rank) != rank) return false;
    var expect: isize = 4; // f32
    var d: usize = rank;
    while (d > 0) : (d -= 1) {
        if (layout.strides_bytes[d - 1] != expect) return false;
        expect *= @intCast(meta.shape[d - 1]);
    }
    return true;
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

    // Fast path: single-tile, packed-contiguous f32 buffers + thread-parallel.
    {
        const sig_tile = try store.acquireTileConstLinear(s.signal, 0);
        var release_sig = true;
        defer if (release_sig) store.releaseConst(sig_tile.token);
        const win_tile = try store.acquireTileConstLinear(s.window, 0);
        var release_win = true;
        defer if (release_win) store.releaseConst(win_tile.token);
        const out_tile = try store.acquireTileMutLinear(s.out, 0);
        var release_out = true;
        defer if (release_out) store.releaseMut(out_tile.token);

        const sig_view = sig_tile.bufferView();
        const win_view = win_tile.bufferView();
        const out_view = out_tile.bufferView();

        if (packedContiguousF32(sig_meta, sig_view.dtype, sig_view.layout) and
            packedContiguousF32(win_meta, win_view.dtype, win_view.layout) and
            packedContiguousF32(out_meta, out_view.dtype, out_view.layout))
        {
            // Tile backing is >= 64-byte aligned, so natural-alignment casts are safe.
            const sig: []const f32 = @alignCast(simd.bytesAsSliceConstUnaligned(f32, sig_view.bytes));
            const win: []const f32 = @alignCast(simd.bytesAsSliceConstUnaligned(f32, win_view.bytes));
            const out: []f32 = @alignCast(simd.bytesAsSliceMutUnaligned(f32, out_view.bytes));
            if (sig.len < batch * samples or win.len < n_fft or out.len < batch * num_frames * 2 * bins) {
                return BackendError.InvalidArgument;
            }
            try execFast(allocator, pool, thread_count, kernels, plan, s, sig, win, out, batch, samples, num_frames, n_fft, bins);
            return;
        }

        // Not contiguous; release and fall through to the scalar path.
        store.releaseConst(sig_tile.token);
        release_sig = false;
        store.releaseConst(win_tile.token);
        release_win = false;
        store.releaseMut(out_tile.token);
        release_out = false;
    }

    try execScalar(allocator, kernels, plan, s, store, sig_meta, win_meta, out_meta, batch, samples, num_frames, n_fft, bins);
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

    fn runItems(ctx_any: *anyopaque, start: usize, end: usize, tid: usize) void {
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
            t.kernels.process_group(t.plan, in_buf, out_group, count, scratch) catch {};
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
        pool.?.parallelForAny(@ptrCast(&task), total_items, 1, FastTask.runItems);
        return;
    }
    FastTask.runItems(@ptrCast(&task), 0, total_items, 0);
}

fn execScalar(
    allocator: std.mem.Allocator,
    kernels: fft_registry.FftKernels,
    plan: *const fft.Plan,
    s: executable.StepSTFT,
    store: tensor_store.TensorStore,
    sig_meta: tensor_store.TensorMeta,
    win_meta: tensor_store.TensorMeta,
    out_meta: tensor_store.TensorMeta,
    batch: usize,
    samples: usize,
    num_frames: usize,
    n_fft: usize,
    bins: usize,
) ExecuteProgramError!void {
    const lanes: usize = kernels.lanes;
    const in_buf = allocator.alloc(f32, lanes * n_fft) catch return BackendError.ExecutionFailed;
    defer allocator.free(in_buf);
    const out_buf = allocator.alloc(f32, lanes * 2 * bins) catch return BackendError.ExecutionFailed;
    defer allocator.free(out_buf);
    const win = allocator.alloc(f32, n_fft) catch return BackendError.ExecutionFailed;
    defer allocator.free(win);
    const scratch = allocator.alignedAlloc(u8, .@"64", fft.scratchBytes(plan, lanes)) catch return BackendError.ExecutionFailed;
    defer allocator.free(scratch);

    var win_cache: exec_utils.TileCacheConstND = .{};
    defer win_cache.deinit(store);
    var sig_cache: exec_utils.TileCacheConstND = .{};
    defer sig_cache.deinit(store);
    var out_cache: exec_utils.TileCacheMutND = .{};
    defer out_cache.deinit(store);

    {
        var j: usize = 0;
        while (j < n_fft) : (j += 1) {
            win[j] = try exec_utils.readScalarF32At(store, win_meta, s.window, &[_]usize{j}, &win_cache);
        }
    }

    const pad: isize = if (s.center) @intCast(n_fft / 2) else 0;

    var b: usize = 0;
    while (b < batch) : (b += 1) {
        var group_start: usize = 0;
        while (group_start < num_frames) : (group_start += lanes) {
            const count: usize = @min(lanes, num_frames - group_start);

            var l: usize = 0;
            while (l < count) : (l += 1) {
                const t: usize = group_start + l;
                const frame_origin: isize = @as(isize, @intCast(t * s.hop_length)) - pad;
                var j: usize = 0;
                while (j < n_fft) : (j += 1) {
                    const idx: isize = frame_origin + @as(isize, @intCast(j));
                    var sample: f32 = 0.0;
                    if (idx >= 0 and idx < @as(isize, @intCast(samples))) {
                        sample = try exec_utils.readScalarF32At(store, sig_meta, s.signal, &[_]usize{ b, @intCast(idx) }, &sig_cache);
                    } else if (s.center) {
                        sample = try exec_utils.readScalarF32At(store, sig_meta, s.signal, &[_]usize{ b, reflectIndex(idx, samples) }, &sig_cache);
                    }
                    in_buf[l * n_fft + j] = sample * win[j];
                }
            }

            kernels.process_group(plan, in_buf, out_buf, count, scratch) catch return BackendError.ExecutionFailed;

            l = 0;
            while (l < count) : (l += 1) {
                const t: usize = group_start + l;
                var k: usize = 0;
                while (k < 2 * bins) : (k += 1) {
                    try exec_utils.writeScalarFromF32At(store, out_meta, s.out, &[_]usize{ b, t, k }, out_buf[l * 2 * bins + k], &out_cache);
                }
            }
        }
    }
}
