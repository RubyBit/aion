// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Graph optimization pipeline.
//!
//! Deliberately separate from tiling (`plan.TilePolicy`): tiling decides how a
//! fixed graph is laid out, whereas these passes *rewrite* the graph itself.
//! `compileGraph` runs this pipeline (after shape inference, before the
//! value→tensor map is sized) so every model — builder or loaded package —
//! benefits without per-call-site changes. Add future passes here.

const std = @import("std");

const graph_mod = @import("graph.zig");
const plan = @import("plan.zig");
const manager_mod = @import("../storage/manager.zig");
const fuse_horizontal_matmul = @import("opt/fuse_horizontal_matmul.zig");
const fuse_gpu_decode = @import("opt/fuse_gpu_decode.zig");
const lower_pointwise_conv = @import("opt/lower_pointwise_conv.zig");

const Graph = graph_mod.Graph;
const StorageManager = manager_mod.StorageManager;
const TilePolicy = plan.TilePolicy;

pub const Error = graph_mod.GraphError || manager_mod.StorageError;

/// Registry of discriminators for `StorageManager.derived_weight_cache`, so
/// distinct weight-deriving transforms don't alias over the same source tensors.
/// One value per transform; append new kinds, never renumber existing ones.
pub const DerivedWeightKind = enum(u32) {
    /// Column-concatenated weight for horizontal MatMul fusion.
    horizontal_matmul_concat = 1,
    _,
};

/// Which optimization passes run. All default-on; flip a field off to bisect a
/// regression or to A/B a pass in tests/benches.
pub const OptPolicy = struct {
    /// Lower pointwise (1x1, groups=1, unit-stride, unpadded) Conv1D to MatMul so
    /// it takes the autotuned GEMM path. See `opt/lower_pointwise_conv.zig`.
    lower_pointwise_conv: bool = true,
    /// Fuse parallel projections off a shared input (Q/K/V, gate/up, …) into one
    /// wide MatMul + slices. See `opt/fuse_horizontal_matmul.zig`.
    fuse_horizontal_matmul: bool = true,
};

/// Run the enabled passes in order, mutating `graph` in place (and allocating any
/// fused weights in `mgr`).
pub fn run(
    allocator: std.mem.Allocator,
    graph: *Graph,
    mgr: *StorageManager,
    policy: TilePolicy,
    opt: OptPolicy,
) Error!void {
    // Lower pointwise convs first so the resulting MatMuls are visible to the
    // horizontal-fusion pass.
    if (opt.lower_pointwise_conv) {
        try lower_pointwise_conv.run(allocator, graph);
    }
    if (opt.fuse_horizontal_matmul) {
        try fuse_horizontal_matmul.run(allocator, graph, mgr, policy);
    }
    // Vertical fusion example
    try fuse_gpu_decode.run(allocator, graph, policy);
}
