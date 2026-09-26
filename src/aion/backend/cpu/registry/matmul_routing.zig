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

/// Whether a small-M q8 matmul should read B where it lies rather than repacking it.
///
/// Packing B pays for itself only when the packed copy gets reused. At these
/// shapes it never does: a decode step visits every weight tile exactly once, so
/// the repack is pure overhead. (When storage still split B into tiles, the test
/// read every tile of a large weight as "small" and sent the whole model down the
/// packing path per token: 3.8 tok/s against 11.5.)
pub fn shouldUseQ8DirectMatvec(params: MatMulParams) bool {
    return params.m != 0 and params.m <= Q8_DIRECT_MAX_M;
}
