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

// --- Vectors in memory ---------------------------------------------------------
//
// A vector has no defined byte layout, so `@ptrCast` between memory and a vector
// is Illegal Behavior (langref, "Relationship with Arrays"): the self-hosted x86_64
// backend pads a vector narrower than 16 bytes to 16, and a store through such a
// pointer writes past the elements. Memory is read and written as an array, which
// does have a defined layout and coerces to and from the vector. LLVM emits the
// same vector loads and stores either way, with one exception: under the x86-64-v4
// (AVX-512) tuning, which prefers 256-bit accesses, a 512-bit store is split in two.

/// Load a `V` from its elements at `ptr`, which may have any alignment.
pub inline fn load(comptime V: type, ptr: anytype) V {
    return loadAligned(V, ptr, 1);
}

/// Load a `V` from its elements at `ptr`, which is aligned to `alignment` bytes
/// (checked in safe builds).
pub inline fn loadAligned(comptime V: type, ptr: anytype, comptime alignment: comptime_int) V {
    const elems: *align(alignment) const Elems(V) = @ptrCast(@alignCast(ptr));
    return elems.*;
}

/// Store `value`'s elements at `ptr`, which may have any alignment.
pub inline fn store(comptime V: type, ptr: anytype, value: V) void {
    storeAligned(V, ptr, 1, value);
}

/// Store `value`'s elements at `ptr`, which is aligned to `alignment` bytes
/// (checked in safe builds).
pub inline fn storeAligned(comptime V: type, ptr: anytype, comptime alignment: comptime_int, value: V) void {
    const elems: *align(alignment) Elems(V) = @ptrCast(@alignCast(ptr));
    elems.* = value;
}

/// The array a vector type's elements occupy in memory.
fn Elems(comptime V: type) type {
    const info = @typeInfo(V).vector;
    return [info.len]info.child;
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
