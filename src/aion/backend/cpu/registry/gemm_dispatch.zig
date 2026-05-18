// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const types = @import("../../types.zig");

const MatMulParams = types.MatMulParams;

/// Shape predicate for routing GEMM-shaped execution through specialized matvec kernels.
pub fn isMatvecShape(m: usize) bool {
    // TODO: could change this to include m > 1 (at least small m), not sure 
    // if thats more efficient or not (bench), but this might come down to making a more holistic
    // benchmark (current one is a bit too "manual")
    return m == 1;
}

/// Direct K-major q8 matvec avoids pack-B cost, but packed kernels can be faster on
/// tiny workloads. Use direct mode only once the tile is large enough to amortize its
/// less cache-friendly B traversal, and only when parallel execution is available.
pub fn shouldUseQ8DirectMatvec(params: MatMulParams, thread_count: usize) bool {
    if (!isMatvecShape(params.m)) return false;
    if (thread_count <= 1) return false;
    const work: usize = std.math.mul(usize, params.k, params.n) catch return false;
    return work >= (64 * 1024);
}
