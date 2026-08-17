// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

//! Test root for `zig build test`.
//!
//! Zig compilation is lazy: merely re-exporting a module from `src/root.zig`
//! does not necessarily force the compiler to analyze that module, which can
//! lead to `0 tests passed` even though other files contain `test` blocks.
//!
//! This file exists to force compilation of modules that contain tests.

const std = @import("std");

comptime {
    _ = @import("aion/backend/backend.zig");
    _ = @import("aion/runtime/thread_pool.zig");
    _ = @import("aion/runtime/device_memory.zig");
    _ = @import("aion/runtime/executable.zig");
    _ = @import("aion/api/tiling.zig");
    // Only GPU files import this, so without an explicit entry its comptime
    // guard against a host-byte accessor would go unchecked in a CPU build.
    _ = @import("aion/runtime/device_store.zig");
    _ = @import("aion/storage/storage.zig");
    _ = @import("aion/storage/test_storage.zig");
    _ = @import("aion/storage/quantize.zig");
    _ = @import("aion/backend/cpu/test_cpu_kernels.zig");
    _ = @import("aion/backend/cpu/test_cpu_backend.zig");
    _ = @import("aion/backend/cpu/kernels/fft.zig");
    _ = @import("aion/backend/cpu/kernels/matmul.zig");
    _ = @import("aion/backend/cpu/kernels/matmul_q_i8.zig");

    _ = @import("aion/graph/test_compile.zig");
    _ = @import("aion/graph/test_program_golden.zig");
    _ = @import("aion/graph/opt/test_optimize.zig");
    _ = @import("aion/api/test_api.zig");
    _ = @import("aion/api/test_nn.zig");
}

test {
    std.testing.refAllDecls(@This());
}

test "sanity: tests are running" {
    try std.testing.expect(true);
}
