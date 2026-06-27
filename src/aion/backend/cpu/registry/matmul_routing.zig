// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const types = @import("../../types.zig");

const MatMulParams = types.MatMulParams;

/// Shape predicate for routing GEMM-shaped execution through specialized matvec kernels.
pub fn isMatvecShape(m: usize) bool {
    return m == 1;
}

/// Largest M routed through the direct k-major q8 matvec (which re-reads B per row
/// but skips the scalar pack-B repack). Above this, the packed gemm amortizes the
/// repack over enough rows to win. Critical for low-latency streaming, where each
/// chunk is only a few frames (small M) and pack-B otherwise dominates per-chunk.
pub const Q8_DIRECT_MAX_M: usize = 16;

/// Direct K-major q8 matvec avoids pack-B cost, but packed kernels can be faster on
/// tiny workloads. Use direct mode only once the tile is large enough to amortize its
/// less cache-friendly B traversal, and only when parallel execution is available.
pub fn shouldUseQ8DirectMatvec(params: MatMulParams, thread_count: usize) bool {
    if (params.m == 0 or params.m > Q8_DIRECT_MAX_M) return false;
    if (thread_count <= 1) return false;
    const work: usize = std.math.mul(usize, params.k, params.n) catch return false;
    return work >= (64 * 1024);
}
