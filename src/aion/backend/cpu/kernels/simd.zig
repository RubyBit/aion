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

/// Write `nr` rows of `rows` (row `first` on, each `ldb` elements apart) into a GEMM
/// B panel `width` columns wide, as its first `kc` panel rows: element `(k, j)` lands
/// at `panel[k * width + j]`. The transpose goes a 4x4 block at a time through
/// registers, so each row is read, and each panel row written, a whole vector at a
/// time. Columns `nr .. width` are zeroed, as the column packers leave them.
pub fn transposeRowsIntoPanel(comptime width: usize, panel: []f32, kc: usize, nr: usize, rows: []align(1) const f32, ldb: usize, first: usize) void {
    const V = @Vector(4, f32);
    var j: usize = 0;
    while (j + 4 <= nr) : (j += 4) {
        const base = (first + j) * ldb;
        var k: usize = 0;
        while (k + 4 <= kc) : (k += 4) {
            const cols = transpose4(.{
                load(V, rows[base + k ..].ptr),
                load(V, rows[base + ldb + k ..].ptr),
                load(V, rows[base + 2 * ldb + k ..].ptr),
                load(V, rows[base + 3 * ldb + k ..].ptr),
            });
            inline for (cols, 0..) |col, q| store(V, panel[(k + q) * width + j ..].ptr, col);
        }
        while (k < kc) : (k += 1) {
            inline for (0..4) |l| panel[k * width + j + l] = rows[base + l * ldb + k];
        }
    }
    while (j < nr) : (j += 1) {
        for (0..kc) |k| panel[k * width + j] = rows[(first + j) * ldb + k];
    }
    if (nr < width) {
        for (0..kc) |k| @memset(panel[k * width + nr .. (k + 1) * width], 0.0);
    }
}

/// Four rows of four as four columns of four.
inline fn transpose4(r: [4]@Vector(4, f32)) [4]@Vector(4, f32) {
    const lo01 = @shuffle(f32, r[0], r[1], [4]i32{ 0, -1, 1, -2 });
    const hi01 = @shuffle(f32, r[0], r[1], [4]i32{ 2, -3, 3, -4 });
    const lo23 = @shuffle(f32, r[2], r[3], [4]i32{ 0, -1, 1, -2 });
    const hi23 = @shuffle(f32, r[2], r[3], [4]i32{ 2, -3, 3, -4 });
    return .{
        @shuffle(f32, lo01, lo23, [4]i32{ 0, 1, -1, -2 }),
        @shuffle(f32, lo01, lo23, [4]i32{ 2, 3, -3, -4 }),
        @shuffle(f32, hi01, hi23, [4]i32{ 0, 1, -1, -2 }),
        @shuffle(f32, hi01, hi23, [4]i32{ 2, 3, -3, -4 }),
    };
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
