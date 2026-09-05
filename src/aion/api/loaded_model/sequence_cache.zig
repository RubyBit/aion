// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Physical capacity for the sequence caches a model carries as recurrent state.
//!
//! Three things stay apart here. *Visibility* is which keys a query may read and
//! belongs to the attention op's `AttentionWindow`. *Retention* is how much
//! logical history the model requires to survive and is declared by the package.
//! *Capacity* is this file's concern: the rows a slot must physically hold before
//! an invocation runs, which is retention plus the width of the append itself.
//!
//! Sites are described in terms of public inputs so capacity settles BEFORE the
//! graph is specialized. That ordering is the point: capacity is part of the
//! specialization key, so a slot grown afterwards leaves the compiled program
//! describing a shape the store no longer has.

const std = @import("std");

const file = @import("../../storage/aion_file.zig");
const template_mod = @import("../../graph/template.zig");

pub const no_input: u32 = std.math.maxInt(u32);

/// Sequence axis of a rank-4 `[B, T, H, D]` cache. `SequenceAppend` states it.
pub const time_axis: usize = 1;

/// How much logical history a cache must keep across invocations.
pub const Retention = union(enum) {
    all,
    bounded: usize,
};

/// One `SequenceAppend` whose cache is a public input.
pub const Site = struct {
    /// Public input index of the cache being appended to.
    cache_input: u32,
    /// Public input whose sequence length is the append width, or `no_input`
    /// when the appended value is produced (then the `tokens` role supplies it).
    width_input: u32 = no_input,
    /// Public i32 input holding this append's per-batch start positions, or
    /// `no_input` when it is produced (then capacity cannot be settled).
    start_input: u32 = no_input,
};

/// Rows the cache must hold for one invocation that appends `append_len`
/// positions starting at logical `start`.
pub fn requiredCapacity(retention: Retention, start: usize, append_len: usize) error{InvalidArgument}!usize {
    const base: usize = switch (retention) {
        .all => start,
        .bounded => |history| @min(start, history),
    };
    return std.math.add(usize, base, append_len) catch error.InvalidArgument;
}

/// The append site that writes public input `index`, or null when the runtime
/// cannot size that cache. Both operands have to be knowable before the graph is
/// specialized: the start index must be a public input, and the width either a
/// public input or — for the produced values a real converter emits — the
/// sequence length of the `tokens` role.
pub fn settleableSite(sites: []const Site, index: usize, has_tokens_role: bool) ?Site {
    for (sites) |site| {
        if (site.cache_input != index) continue;
        if (site.start_input == no_input) return null;
        if (site.width_input == no_input and !has_tokens_role) return null;
        return site;
    }
    return null;
}

/// Every append site the runtime can size, in public-input terms. Regions are
/// walked too: a cache written inside one is still model-level state.
pub fn collect(gpa: std.mem.Allocator, t: template_mod.Template) error{OutOfMemory}![]Site {
    var out: std.ArrayList(Site) = .empty;
    errdefer out.deinit(gpa);

    for (t.nodes) |node| try note(gpa, &out, t, node);
    for (t.regions) |region| {
        for (region.nodes) |node| try note(gpa, &out, t, node);
    }
    return out.toOwnedSlice(gpa);
}

fn note(gpa: std.mem.Allocator, out: *std.ArrayList(Site), t: template_mod.Template, node: file.NodeRecord) error{OutOfMemory}!void {
    if (node.op != .SequenceAppend or node.inputs.len < 3) return;
    const cache_input = publicInputIndex(t, node.inputs[0]) orelse return;
    try out.append(gpa, .{
        .cache_input = cache_input,
        .width_input = publicInputIndex(t, node.inputs[1]) orelse no_input,
        .start_input = publicInputIndex(t, node.inputs[2]) orelse no_input,
    });
}

fn publicInputIndex(t: template_mod.Template, value: u32) ?u32 {
    if (value >= t.values.len or t.values[value].source != .public_input) return null;
    for (t.inputs, 0..) |id, idx| {
        if (id == value) return @intCast(idx);
    }
    return null;
}
