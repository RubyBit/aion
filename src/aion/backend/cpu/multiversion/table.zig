// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Cross-object kernel dispatch table for portable single-binary multiversioning.
//
// When the library is built with `-Dmultiversion=true`, `build.zig` compiles
// `kernels_export.zig` once per CPU feature tier (x86: v3 / v3_vnni / v4),
// each as a separate object linked into the final archive. Every tier object
// fills in one `DispatchTable` instance (specialized to that tier's SIMD lane
// width and ISA) and exports a uniquely-named accessor returning a pointer to it.
//
// The main module never sees a `@Vector` across this boundary: the table holds
// only function pointers, slices, scalars and small integer structs, all of
// which share a stable ABI across compilation units of the *same* Zig compiler
// and target architecture (only the CPU feature set differs between tiers). The
// SIMD code lives entirely inside the kernel bodies the pointers refer to.
//
// `abi_version` guards the one assumption we rely on (same-compiler layout): it
// is validated at backend init before any table pointer is used.

const matmul_registry = @import("../registry/matmul_registry.zig");
const matmul_q_registry = @import("../registry/matmul_q_registry.zig");
const matmul_nt_registry = @import("../registry/matmul_nt_registry.zig");
const matvec_registry = @import("../registry/matvec_registry.zig");
const attention_registry = @import("../registry/attention_registry.zig");
const conv1d_registry = @import("../registry/conv1d_registry.zig");
const conv2d_registry = @import("../registry/conv2d_registry.zig");
const fft_registry = @import("../registry/fft_registry.zig");

/// Bumped whenever the `DispatchTable` layout or any kernel-struct ABI changes.
pub const ABI_VERSION: u32 = 2;

/// Ordering of the tiled (packed-GEMM) kernel arrays below.
pub const TILE_SMALL = 0;
pub const TILE_MEDIUM = 1;
pub const TILE_LARGE = 2;

/// The full set of CPU kernels for a single ISA tier.
///
/// Reuses the existing per-family registry kernel structs verbatim — no separate
/// C ABI is introduced. The packed-GEMM families (`matmul`, `matmul_q`) keep
/// all three cache-blocking variants so the main module can still pick a tile by
/// runtime L2 size; every other family selects purely on lane width, so one entry
/// per tier suffices.
pub const DispatchTable = struct {
    abi_version: u32,
    lanes: u32,

    /// [small, medium, large] for this tier's lane width.
    matmul: [3]matmul_registry.F32Kernels,
    matmul_q: [3]matmul_q_registry.QuantKernels,

    matmul_nt: matmul_nt_registry.Kernels,
    matvec: matvec_registry.Kernels,
    attention: attention_registry.Kernels,
    relpos_mha: attention_registry.Kernels,
    conv1d: conv1d_registry.Kernels,
    conv2d: conv2d_registry.Kernels,
    fft: fft_registry.FftKernels,
};
