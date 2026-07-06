// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Aggregator for graph optimization-pass tests. `src/tests.zig` imports only
//! this file; add new per-pass test modules here.

test {
    _ = @import("test_fuse_horizontal_matmul.zig");
    _ = @import("test_lower_pointwise_conv.zig");
}
