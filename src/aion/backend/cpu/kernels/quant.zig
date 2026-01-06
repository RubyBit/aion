const types = @import("../../types.zig");
const simd = @import("simd.zig");

const BackendError = types.BackendError;
const MatMulParams = types.MatMulParams;

/// q4_0 block: 32 elements stored as 2-byte f16 scale + 16 bytes (32 nibbles)
pub const Q4_0_BLOCK_ELEMS: usize = 32;
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// q8_0 block: 32 elements stored as 2-byte f16 scale + 32 bytes (int8s)
pub const Q8_0_BLOCK_ELEMS: usize = 32;
pub const Q8_0_BLOCK_BYTES: usize = 34;

/// Dequantize a q4_0 block to f32 (32 floats).
pub fn dequantQ4_0Block(block: *const [Q4_0_BLOCK_BYTES]u8, out: *[Q4_0_BLOCK_ELEMS]f32) void {
    // First 2 bytes: f16 scale
    const scale_bytes: *const [2]u8 = block[0..2];
    const scale: f32 = @as(f32, @as(f16, @bitCast(scale_bytes.*)));

    // Next 16 bytes: 32 4-bit values (nibbles), each shifted by -8
    const nibbles: *const [16]u8 = block[2..18];

    for (0..16) |i| {
        const byte: u8 = nibbles[i];
        // Low nibble first, then high nibble (ggml convention)
        const lo: i8 = @as(i8, @intCast(byte & 0x0F)) - 8;
        const hi: i8 = @as(i8, @intCast(byte >> 4)) - 8;
        out[i * 2] = scale * @as(f32, @floatFromInt(lo));
        out[i * 2 + 1] = scale * @as(f32, @floatFromInt(hi));
    }
}

/// Dequantize a q8_0 block to f32 (32 floats).
pub fn dequantQ8_0Block(block: *const [Q8_0_BLOCK_BYTES]u8, out: *[Q8_0_BLOCK_ELEMS]f32) void {
    // First 2 bytes: f16 scale
    const scale_bytes: *const [2]u8 = block[0..2];
    const scale: f32 = @as(f32, @as(f16, @bitCast(scale_bytes.*)));

    // Next 32 bytes: 32 int8 values
    const quants: *const [32]i8 = @ptrCast(block[2..34]);

    for (0..32) |i| {
        out[i] = scale * @as(f32, @floatFromInt(quants[i]));
    }
}

fn dotQ8_0BlockF32(a_ptr: [*]align(1) const f32, block: *const [Q8_0_BLOCK_BYTES]u8) f32 {
    const scale_bytes: *const [2]u8 = block[0..2];
    const scale: f32 = @as(f32, @as(f16, @bitCast(scale_bytes.*)));

    // 32 int8 values immediately after scale
    const q_ptr: [*]align(1) const i8 = @as([*]align(1) const i8, @ptrCast(block.ptr + 2));

    const lanes: usize = comptime simd.lanesF32();
    const VecF = @Vector(lanes, f32);
    const scale_v: VecF = @splat(scale);

    var acc: f32 = 0.0;
    var i: usize = 0;

    // Vectorize in chunks of `lanes`.
    while (i + lanes <= Q8_0_BLOCK_ELEMS) : (i += lanes) {
        const a_v: VecF = @as(*align(1) const VecF, @ptrCast(a_ptr + i)).*;
        const q_i8: @Vector(lanes, i8) = @as(*align(1) const @Vector(lanes, i8), @ptrCast(q_ptr + i)).*;
        const q_f: VecF = @floatFromInt(q_i8);
        const prod: VecF = a_v * (scale_v * q_f);
        acc += @reduce(.Add, prod);
    }

    // Scalar tail (should be none for lanes that divide 32, but keep it robust)
    while (i < Q8_0_BLOCK_ELEMS) : (i += 1) {
        acc += a_ptr[i] * (scale * @as(f32, @floatFromInt(q_ptr[i])));
    }

    return acc;
}

/// C[M,N] (f32) = A[M,K] (f32) @ B[K,N] (q4_0)
///
/// B is stored as a row-major array of blocks with shape [k_blocks, n].
pub fn matmulQ4_0(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
    const m: usize = params.m;
    const n: usize = params.n;
    const k: usize = params.k;
    const alpha: f32 = params.alpha;
    const beta: f32 = params.beta;

    if (k % Q4_0_BLOCK_ELEMS != 0) return BackendError.InvalidArgument;
    const k_blocks: usize = k / Q4_0_BLOCK_ELEMS;

    const c: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
    const a: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, a_bytes);

    if (c.len < m * n or a.len < m * k) return BackendError.InvalidArgument;
    if (b_bytes.len < k_blocks * n * Q4_0_BLOCK_BYTES) return BackendError.InvalidArgument;

    var dequant_buf: [Q4_0_BLOCK_ELEMS]f32 = undefined;

    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f32 = 0.0;

            for (0..k_blocks) |kb| {
                const block_offset: usize = (kb * n + j) * Q4_0_BLOCK_BYTES;
                const block_ptr: *const [Q4_0_BLOCK_BYTES]u8 = @ptrCast(b_bytes[block_offset..][0..Q4_0_BLOCK_BYTES]);
                dequantQ4_0Block(block_ptr, &dequant_buf);

                const a_offset: usize = i * k + kb * Q4_0_BLOCK_ELEMS;
                for (0..Q4_0_BLOCK_ELEMS) |q| {
                    acc += a[a_offset + q] * dequant_buf[q];
                }
            }

            const c_idx: usize = i * n + j;
            c[c_idx] = alpha * acc + beta * c[c_idx];
        }
    }
}

/// C[M,N] (f32) = A[M,K] (f32) @ B[K,N] (q8_0)
///
/// B is stored as a row-major array of blocks with shape [k_blocks, n].
pub fn matmulQ8_0(params: MatMulParams, c_bytes: []u8, a_bytes: []const u8, b_bytes: []const u8) BackendError!void {
    const m: usize = params.m;
    const n: usize = params.n;
    const k: usize = params.k;
    const alpha: f32 = params.alpha;
    const beta: f32 = params.beta;

    if (k % Q8_0_BLOCK_ELEMS != 0) return BackendError.InvalidArgument;
    const k_blocks: usize = k / Q8_0_BLOCK_ELEMS;

    const c: []align(1) f32 = simd.bytesAsSliceMutUnaligned(f32, c_bytes);
    const a: []align(1) const f32 = simd.bytesAsSliceConstUnaligned(f32, a_bytes);

    if (c.len < m * n or a.len < m * k) return BackendError.InvalidArgument;
    if (b_bytes.len < k_blocks * n * Q8_0_BLOCK_BYTES) return BackendError.InvalidArgument;

    // Fused dequant+dot: avoids temporary dequant buffer and uses @Vector for q8.
    for (0..m) |i| {
        for (0..n) |j| {
            var acc: f32 = 0.0;

            for (0..k_blocks) |kb| {
                const block_offset: usize = (kb * n + j) * Q8_0_BLOCK_BYTES;
                const block_ptr: *const [Q8_0_BLOCK_BYTES]u8 = @ptrCast(b_bytes[block_offset..][0..Q8_0_BLOCK_BYTES]);

                const a_offset: usize = i * k + kb * Q8_0_BLOCK_ELEMS;
                acc += dotQ8_0BlockF32(a.ptr + a_offset, block_ptr);
            }

            const c_idx: usize = i * n + j;
            c[c_idx] = alpha * acc + beta * c[c_idx];
        }
    }
}
