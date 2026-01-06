//! Test root for `zig build test`.
//!
//! Zig compilation is lazy: merely re-exporting a module from `src/root.zig`
//! does not necessarily force the compiler to analyze that module, which can
//! lead to `0 tests passed` even though other files contain `test` blocks.
//!
//! This file exists to force compilation of modules that contain tests.

const std = @import("std");

comptime {
    // Force semantic analysis of these files so their `test` blocks are discovered.
    _ = @import("aion/backend/backend.zig");
    _ = @import("aion/backend/cpu/test_cpu_backend.zig");
    _ = @import("aion/storage/test_storage.zig");
    _ = @import("aion/graph/test_program.zig");
}

test "sanity: tests are running" {
    try std.testing.expect(true);
}
