// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! RFFT / STFT execution for the GPU backend (kernels/fft.wgsl, stft.wgsl):
//! naive per-bin DFT kernels, one thread per output bin. Output packing matches
//! the CPU executors (one-sided spectrum, real halves then imaginary). v1: f32,
//! single buffers. Numerics: the CPU uses an O(n log n) FFT plan, the GPU
//! a direct DFT — same math, different rounding order, so CPU-vs-GPU
//! comparisons need a tolerance scaled to the spectrum magnitude.

const std = @import("std");
const wgpu = @import("../wgpu.zig");
const pipelines = @import("../pipelines.zig");
const context = @import("../context.zig");
const backend_mod = @import("../../backend.zig");
const executable = @import("../../../runtime/executable.zig");

const c = wgpu.c;
const Ctx = context.Ctx;
const Frame = @import("../frame.zig").Frame;
const ExecuteProgramError = backend_mod.ExecuteProgramError;
const KernelDesc = pipelines.KernelDesc;

const fft_kernel: KernelDesc = .{ .name = "fft", .wgsl = @embedFile("../kernels/fft.wgsl") };
const stft_kernel: KernelDesc = .{ .name = "stft", .wgsl = @embedFile("../kernels/stft.wgsl") };

const WG_1D: u32 = 64;

/// Field order matches `struct Params` in fft.wgsl.
const RfftParams = extern struct { rows: u32, n: u32, bins: u32, total: u32 };
/// Field order matches `struct Params` in stft.wgsl.
const StftParams = extern struct {
    batch: u32,
    samples: u32,
    frames: u32,
    n_fft: u32,
    hop: u32,
    bins: u32,
    center: u32,
    total: u32,
};

fn groups1D(n: u32) u32 {
    return @max(1, @min(context.ceilDiv(n, WG_1D), context.MAX_GROUPS_1D));
}

// Cooperative Stockham FFT parameters (mirror fft.wgsl / stft.wgsl constants).
const WGC: u32 = 256;
const MAX_N_COOP: u32 = 1024;
const COOP_SHARED_BYTES: u32 = 2 * MAX_N_COOP * 8; // two vec2<f32> ping-pong buffers

/// Whether the one-workgroup-per-row cooperative FFT fits this device and grid.
/// `groups` is the number of workgroups (rows/frames) launched on the x dim.
fn fftCoopEligible(n: u32, groups: u32, limits: wgpu.Limits) bool {
    if (n > MAX_N_COOP) return false;
    if (COOP_SHARED_BYTES > limits.max_shared_bytes) return false;
    if (WGC > limits.max_invocations or WGC > limits.max_workgroup_size_x) return false;
    if (groups == 0 or groups > context.MAX_GROUPS_PER_DIM) return false;
    return true;
}

pub fn execRFFT(ctx: Ctx, frame: *Frame, s: executable.StepRFFT) ExecuteProgramError!void {
    const hs = ctx.store;
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const x_meta = hs.meta(s.x) catch return error.ExecutionFailed;

    if (out_meta.dtype != .f32 or x_meta.dtype != .f32) return error.Unsupported;
    if (x_meta.chunks != 1 or out_meta.chunks != 1) return error.Unsupported;

    const n_fft = s.n_fft;
    if (n_fft < 4 or (n_fft & (n_fft - 1)) != 0) return error.Unsupported;
    const bins = n_fft / 2 + 1;

    const dx = ctx.store.acquireConst(s.x) catch return error.ExecutionFailed;
    const dout = ctx.store.acquireMut(s.out) catch return error.ExecutionFailed;
    defer {
        hs.releaseConst(dx.token);
        hs.releaseMut(dout.token);
    }
    if (!context.storageBindingFits(ctx, dx.len) or !context.storageBindingFits(ctx, dout.len)) return error.Unsupported;

    const x_rank: usize = x_meta.rank;
    if (x_rank == 0 or x_meta.shape[x_rank - 1] != n_fft) return error.Unsupported;
    var x_n: usize = 1;
    for (x_meta.shape) |d| x_n *= d;
    var out_n: usize = 1;
    for (out_meta.shape) |d| out_n *= d;
    const rows = std.math.cast(u32, x_n / n_fft) orelse return error.Unsupported;
    if (out_n < @as(usize, rows) * 2 * bins) return error.Unsupported;

    const total = std.math.cast(u32, @as(usize, rows) * bins) orelse return error.Unsupported;
    const params: RfftParams = .{
        .rows = rows,
        .n = @intCast(n_fft),
        .bins = @intCast(bins),
        .total = total,
    };
    const bufs = [_]c.WGPUBuffer{
        ctx.devmem.bufferFor(dx.handle).?,
        ctx.devmem.bufferFor(dout.handle).?,
    };
    const sizes = [_]u64{ dx.len, dout.len };
    if (fftCoopEligible(@intCast(n_fft), rows, ctx.gpu.limits)) {
        // One workgroup per row, cooperative Stockham FFT.
        const built = try ctx.pipes.get(fft_kernel, "rfft_coop");
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ rows, 1, 1 });
    } else {
        const built = try ctx.pipes.get(fft_kernel, "rfft_dft");
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(total), 1, 1 });
    }
}

pub fn execSTFT(ctx: Ctx, frame: *Frame, s: executable.StepSTFT) ExecuteProgramError!void {
    const hs = ctx.store;
    const out_meta = hs.meta(s.out) catch return error.ExecutionFailed;
    const sig_meta = hs.meta(s.signal) catch return error.ExecutionFailed;
    const win_meta = hs.meta(s.window) catch return error.ExecutionFailed;

    if (out_meta.dtype != .f32 or sig_meta.dtype != .f32 or win_meta.dtype != .f32) return error.Unsupported;
    if (sig_meta.rank != 2 or win_meta.rank != 1 or out_meta.rank != 3) return error.Unsupported;
    if (sig_meta.chunks != 1 or win_meta.chunks != 1 or out_meta.chunks != 1) return error.Unsupported;

    const n_fft = s.n_fft;
    if (n_fft < 4 or (n_fft & (n_fft - 1)) != 0 or s.hop_length == 0) return error.Unsupported;
    const bins = n_fft / 2 + 1;
    const batch = sig_meta.shape[0];
    const samples = sig_meta.shape[1];
    const frames = s.num_frames;
    if (batch == 0 or frames == 0) return;
    if (win_meta.shape[0] != n_fft) return error.Unsupported;

    const dsig = ctx.store.acquireConst(s.signal) catch return error.ExecutionFailed;
    const dwin = ctx.store.acquireConst(s.window) catch return error.ExecutionFailed;
    const dout = ctx.store.acquireMut(s.out) catch return error.ExecutionFailed;
    defer {
        hs.releaseConst(dsig.token);
        hs.releaseConst(dwin.token);
        hs.releaseMut(dout.token);
    }
    if (!context.storageBindingFits(ctx, dsig.len) or !context.storageBindingFits(ctx, dout.len)) return error.Unsupported;

    if (out_meta.shape[0] * out_meta.shape[1] * out_meta.shape[2] < batch * frames * 2 * bins) return error.Unsupported;

    const total = std.math.cast(u32, batch * frames * bins) orelse return error.Unsupported;
    const params: StftParams = .{
        .batch = @intCast(batch),
        .samples = @intCast(samples),
        .frames = @intCast(frames),
        .n_fft = @intCast(n_fft),
        .hop = @intCast(s.hop_length),
        .bins = @intCast(bins),
        .center = @intFromBool(s.center),
        .total = total,
    };
    const bufs = [_]c.WGPUBuffer{
        ctx.devmem.bufferFor(dsig.handle).?,
        ctx.devmem.bufferFor(dwin.handle).?,
        ctx.devmem.bufferFor(dout.handle).?,
    };
    const sizes = [_]u64{ dsig.len, dwin.len, dout.len };
    const frames_u32: u32 = @intCast(frames);
    const batch_u32: u32 = @intCast(batch);
    // 2D grid (frame, batch) so neither dim overflows the per-dim cap.
    if (fftCoopEligible(@intCast(n_fft), frames_u32, ctx.gpu.limits) and batch_u32 <= context.MAX_GROUPS_PER_DIM) {
        const built = try ctx.pipes.get(stft_kernel, "stft_coop");
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ frames_u32, batch_u32, 1 });
    } else {
        const built = try ctx.pipes.get(stft_kernel, "stft_dft");
        try frame.recordCompute(built, &bufs, &sizes, std.mem.asBytes(&params), .{ groups1D(total), 1, 1 });
    }
}
