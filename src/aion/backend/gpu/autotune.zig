// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Kernel choice by timing on the device.
//!
//! Kernels that compute the same thing differ in how they tile the work, and
//! which tiling is fastest depends on how many workgroups the device runs at once
//! and how wide a block its memory system rewards: properties WebGPU does not
//! describe. An op hands its candidates to `Tuner.pick`, which times each one the
//! first time a problem is seen and remembers the fastest. One harness, so every
//! op family is timed the same way.

const std = @import("std");
const context = @import("context.zig");
const backend_mod = @import("../backend.zig");
const env = @import("../../env.zig");
const Frame = @import("frame.zig").Frame;

const Ctx = context.Ctx;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

/// The op families that tune. Part of the key: two families' problems can share
/// dims and still want different candidates.
pub const Family = enum(u8) { gemm, gemm_rows, gemm_q8, conv, attention };

/// The problem a choice was timed for: the family and the dims that decide it.
pub const Key = struct {
    family: Family,
    dims: [8]u32 = @splat(0),
    /// The candidates that can run this problem, one bit each. `pick` fills it in,
    /// so a remembered choice is only reused where it can run: dims in one size
    /// class can still differ in which tiles divide them.
    eligible: u64 = 0,

    pub fn of(family: Family, dims: anytype) Key {
        var key: Key = .{ .family = family };
        inline for (dims, 0..) |d, i| key.dims[i] = @intCast(d);
        return key;
    }
};

/// The size class a dim that follows the input's length is keyed by: the next
/// power of two. A prompt of 137 tokens and one of 138 want the same tiling, and
/// timing every length a model is run at cost more than the tuning bought (Gemma
/// prefill: 12 GEMMs re-timed per new length, ~0.8 s, against a ~50 ms run). The
/// kernel still runs at the exact size; only the remembered choice is shared.
pub fn sizeClass(n: anytype) u32 {
    const v: u32 = @intCast(n);
    if (v <= 1) return v;
    return std.math.ceilPowerOfTwo(u32, v) catch std.math.maxInt(u32);
}

/// How long a timed burst lasts, whatever the kernel costs: as many runs as fit,
/// found from a short probe. A burst is one submit, whose own cost is a few
/// hundred microseconds and varies, so a cheap kernel needs many runs before
/// candidates a few tens of microseconds apart are told apart; a kernel that
/// takes milliseconds needs only a few, and a slow candidate costs no more time
/// to rule out than a fast one.
const BURST_NS: u64 = 8 * std.time.ns_per_ms;
const PROBE_RUNS: usize = 4;
const MAX_RUNS: usize = 256;

/// Times every candidate is timed. A device raises its clock under load over a
/// few hundred milliseconds, so timed one after another the last candidate would
/// run fastest whatever it is. Instead each round times them all, a candidate
/// keeps its best round, and rounds alternate direction: the last and warmest
/// one ends on the first candidate, so what is left of the ramp favours the
/// default over a later candidate that only ties it.
const ROUNDS: usize = 4;

pub const Tuner = struct {
    choices: std.AutoHashMap(Key, usize),

    pub fn init(allocator: std.mem.Allocator) Tuner {
        return .{ .choices = .init(allocator) };
    }

    pub fn deinit(self: *Tuner) void {
        self.choices.deinit();
    }

    /// Problems tuned so far in `family`.
    pub fn count(self: *const Tuner, family: Family) usize {
        var n: usize = 0;
        var it = self.choices.keyIterator();
        while (it.next()) |key| n += @intFromBool(key.family == family);
        return n;
    }

    /// The fastest of `candidates` for `key`: remembered, or timed now. Null when
    /// none can run.
    ///
    /// `cand` describes the candidates (duck-typed):
    ///   fn eligible(self, idx: usize) bool
    ///   fn record(self, idx: usize, frame: *Frame) ExecuteProgramError!void
    /// `record` records one run of candidate `idx` into `frame`. A trial runs on
    /// the op's real operands, so it must be safe to repeat: it may overwrite the
    /// op's output, which the real run rewrites, and nothing else.
    pub fn pick(self: *Tuner, ctx: Ctx, frame: *Frame, problem: Key, candidates: usize, cand: anytype) ExecuteProgramError!?usize {
        std.debug.assert(candidates <= @bitSizeOf(u64));
        var key = problem;
        var only: ?usize = null;
        var eligible: usize = 0;
        for (0..candidates) |idx| {
            if (!cand.eligible(idx)) continue;
            only = only orelse idx;
            eligible += 1;
            key.eligible |= @as(u64, 1) << @intCast(idx);
        }
        if (self.choices.get(key)) |idx| return idx;

        var best: ?usize = only;
        if (eligible > 1) {
            // Trials run ahead of anything still pending in `frame`, and they
            // overwrite the op's output — memory the planner may share with a
            // tensor a pending step has yet to read. So everything recorded so far
            // goes first.
            frame.flushInPlace() catch return error.ExecutionFailed;
            best = try fastest(ctx, key, candidates, cand);
        }
        if (best) |idx| self.choices.put(key, idx) catch {}; // remembering is best-effort
        return best;
    }
};

/// The candidate whose run is shortest, the earliest of any that tie; null when
/// none ran.
fn fastest(ctx: Ctx, key: Key, candidates: usize, cand: anytype) ExecuteProgramError!?usize {
    // Per candidate: runs in its burst (0 = cannot run), and its best run so far.
    const runs = ctx.allocator.alloc(usize, candidates) catch return error.OutOfMemory;
    defer ctx.allocator.free(runs);
    const best_ns = ctx.allocator.alloc(u64, candidates) catch return error.OutOfMemory;
    defer ctx.allocator.free(best_ns);
    for (0..candidates) |idx| {
        runs[idx] = if (cand.eligible(idx)) burstRuns(ctx, cand, idx) catch 0 else 0;
        best_ns[idx] = std.math.maxInt(u64);
    }

    for (0..ROUNDS) |round| {
        for (0..candidates) |step| {
            const idx = if (round % 2 == 0) step else candidates - 1 - step;
            if (runs[idx] == 0) continue;
            const ns = burst(ctx, cand, idx, runs[idx]) catch {
                runs[idx] = 0;
                continue;
            };
            best_ns[idx] = @min(best_ns[idx], ns / runs[idx]);
        }
    }

    const trace = env.flagEnabled("AION_GPU_TRACE");
    var best: ?usize = null;
    for (0..candidates) |idx| {
        if (runs[idx] == 0) continue;
        if (trace) std.debug.print("[gpu] tune {s} {any}: candidate {d} runs in {d} us\n", .{ @tagName(key.family), key.dims, idx, best_ns[idx] / std.time.ns_per_us });
        if (best == null or best_ns[idx] < best_ns[best.?]) best = idx;
    }
    if (trace) std.debug.print("[gpu] tune {s} {any}: chose {?d}\n", .{ @tagName(key.family), key.dims, best });
    return best;
}

/// Runs in a burst of candidate `idx`, found from a probe after one untimed run
/// that builds its pipeline.
fn burstRuns(ctx: Ctx, cand: anytype, idx: usize) ExecuteProgramError!usize {
    _ = try burst(ctx, cand, idx, 1);
    const probe = try burst(ctx, cand, idx, PROBE_RUNS);
    return std.math.clamp(PROBE_RUNS * BURST_NS / @max(probe, 1), 1, MAX_RUNS);
}

/// Nanoseconds for `runs` runs of candidate `idx`, recorded into one frame the way
/// a program records its steps, submitted, and waited for.
fn burst(ctx: Ctx, cand: anytype, idx: usize, runs: usize) ExecuteProgramError!u64 {
    const start = nowNs();
    {
        var f = try Frame.init(ctx.allocator, ctx.gpu);
        defer f.deinit();
        for (0..runs) |_| try cand.record(idx, &f);
        f.submit();
    }
    ctx.gpu.waitIdle();
    return nowNs() - start;
}

fn nowNs() u64 {
    const ts: std.Io.Timestamp = std.Io.Clock.awake.now(std.Options.debug_io);
    const ns: i96 = ts.toNanoseconds();
    if (ns <= 0) return 0;
    return @intCast(@min(ns, @as(i96, std.math.maxInt(u64))));
}
