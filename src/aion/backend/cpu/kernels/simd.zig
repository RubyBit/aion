// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const builtin = @import("builtin");

/// Unaligned-safe typed view over raw bytes.
///
/// NOTE: This truncates any remainder bytes that are not a whole number of T.
pub fn bytesAsSliceConstUnaligned(comptime T: type, bytes: []const u8) []align(1) const T {
    const n: usize = bytes.len / @sizeOf(T);
    return @as([*]align(1) const T, @ptrCast(bytes.ptr))[0..n];
}

/// Unaligned-safe typed view over raw bytes.
///
/// NOTE: This truncates any remainder bytes that are not a whole number of T.
pub fn bytesAsSliceMutUnaligned(comptime T: type, bytes: []u8) []align(1) T {
    const n: usize = bytes.len / @sizeOf(T);
    return @as([*]align(1) T, @ptrCast(bytes.ptr))[0..n];
}

/// Lane count for f32 vectorization.
///
/// v0 policy: pick a reasonable default per-arch and rely on the compiler to
/// scalarize/split if the target ISA doesn't support that width.
pub fn lanesF32() usize {
    return switch (builtin.cpu.arch) {
        // 8-wide often maps well to AVX2; on SSE-only targets the compiler can split/scalarize.
        .x86_64 => 8,
        // NEON is typically 4-wide for f32.
        .aarch64 => 4,
        else => 4,
    };
}
