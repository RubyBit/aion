// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

//! Test root for the GPU suites (`zig build gpu-test`, and part of `zig build test`
//! when `-Dgpu` is on). One artifact against the `aion` module built with
//! enable_gpu=true, so it links wgpu-native once and runs as one test process.

comptime {
    // Kernel-level: each op on the GPU backend against the CPU backend.
    _ = @import("aion/backend/gpu/test_gpu_backend.zig");
    // Public API: device selection, residency and state across runs.
    _ = @import("aion/api/test_api_gpu.zig");
}
