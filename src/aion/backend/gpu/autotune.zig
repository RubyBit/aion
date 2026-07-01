// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Generic per-shape kernel autotuner. Op-agnostic: given a set of candidate
//! kernel configs and a way to (a) test eligibility and (b) time a candidate on
//! device, it benchmarks the eligible ones once per problem shape and caches the
//! fastest. Reusable by any op family (matmul today; q8_0 matvec / attention later).
//!
//! The caller supplies a `ctx` value exposing two methods (duck-typed):
//!   fn eligible(self, idx: usize) bool   // can this config run on this device/shape?
//!   fn timeNs(self, idx: usize) ?u64     // total ns for a fixed burst; null = skip
//! `ctx` does the op-specific work (build pipeline, record dispatches, sync, time);
//! this module only owns the cache + the pick-the-min loop.

const std = @import("std");

/// Shape-key → chosen candidate index.
pub const Cache = std.AutoHashMap(u64, usize);

/// Monotonic nanoseconds (matches src/bench.zig's clock).
pub fn nowNs() u64 {
    const ts: std.Io.Timestamp = std.Io.Clock.awake.now(std.Options.debug_io);
    const ns: i96 = ts.toNanoseconds();
    if (ns <= 0) return 0;
    return @intCast(@min(ns, @as(i96, std.math.maxInt(u64))));
}

/// Pack a 3D problem/tile shape into a cache key (each dim < 2^21).
pub fn shapeKey(d0: usize, d1: usize, d2: usize) u64 {
    return (@as(u64, @intCast(d0)) << 42) | (@as(u64, @intCast(d1)) << 21) | @as(u64, @intCast(d2));
}

/// Return the cached-or-newly-tuned best candidate index for `key`, or null if no
/// candidate was eligible+timeable. On a miss, times every eligible candidate via
/// `ctx.timeNs` and caches the fastest.
pub fn pickBest(cache: *Cache, key: u64, count: usize, ctx: anytype) ?usize {
    if (cache.get(key)) |idx| return idx;

    var best: ?usize = null;
    var best_ns: u64 = std.math.maxInt(u64);
    var idx: usize = 0;
    while (idx < count) : (idx += 1) {
        if (!ctx.eligible(idx)) continue;
        const ns = ctx.timeNs(idx) orelse continue;
        if (best == null or ns < best_ns) {
            best_ns = ns;
            best = idx;
        }
    }
    if (best) |b| cache.put(key, b) catch {}; // cache is best-effort
    return best;
}
