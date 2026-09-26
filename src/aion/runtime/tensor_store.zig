// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const types = @import("../backend/types.zig");

pub const TensorId = u32;

pub const StoreError = error{ InvalidArgument, OutOfMemory };

pub const SequenceCachePolicyKind = enum(u8) {
    none = 0,
    growable = 1,
    rolling = 2,
};

pub const SequenceCachePolicyInfo = struct {
    kind: SequenceCachePolicyKind = .none,
    rolling_history_tokens: usize = 0,
};

pub const QuantBlockOrder = types.QuantBlockOrder;

pub const TensorMeta = struct {
    dtype: types.DType,
    rank: u8,
    shape: []const usize,
    /// Buffers the tensor's bytes are split into along dim 0: always 1 on the
    /// host, more only for a device tensor past the binding limit (`storage/layout.zig`).
    chunks: usize = 1,
    /// Only ever grouped on a tensor a layout pass derived; never serialized.
    block_order: QuantBlockOrder = .row_major,
};

pub const INLINE_RANK: usize = 8;

/// A tensor's host bytes (the whole packed row-major layout) and its layout.
pub const ViewConst = struct {
    bytes: []const u8,
    dtype: types.DType,
    rank: u8,
    shape_mem: [INLINE_RANK]usize,
    strides_mem: [INLINE_RANK]isize,
    token: usize = 0,

    pub fn bufferView(self: *const ViewConst) types.BufferViewConst {
        const r: usize = @as(usize, self.rank);
        return .{
            .bytes = self.bytes,
            .dtype = self.dtype,
            .layout = .{
                .rank = self.rank,
                .shape = self.shape_mem[0..r],
                .strides_bytes = self.strides_mem[0..r],
            },
        };
    }
};

pub const ViewMut = struct {
    bytes: []u8,
    dtype: types.DType,
    rank: u8,
    shape_mem: [INLINE_RANK]usize,
    strides_mem: [INLINE_RANK]isize,
    token: usize = 0,

    pub fn bufferView(self: *const ViewMut) types.BufferViewMut {
        const r: usize = @as(usize, self.rank);
        return .{
            .bytes = self.bytes,
            .dtype = self.dtype,
            .layout = .{
                .rank = self.rank,
                .shape = self.shape_mem[0..r],
                .strides_bytes = self.strides_mem[0..r],
            },
        };
    }
};

/// One device buffer of a device-resident tensor (a tensor migrated via `moveTensor`
/// under move semantics). `handle` is a `DeviceMemory.DeviceHandle`, kept as `u64` so
/// this generic seam names no GPU API.
pub const DeviceChunkRef = struct {
    handle: u64,
    len: usize,
    /// Rows of dim 0 this chunk holds.
    rows: usize,
};

/// Packed row-major byte strides of `shape` (zero for a quantized dtype, whose
/// elements are not individually addressable).
pub fn packedStrides(dtype: types.DType, shape: []const usize, out: []isize) void {
    const info = dtype.info();
    var stride: usize = if (info.is_quantized) 0 else info.block_bytes;
    var d: usize = shape.len;
    while (d > 0) {
        d -= 1;
        out[d] = @intCast(stride);
        stride *= shape[d];
    }
}

/// Backend-facing storage interface.
///
/// Acquiring a tensor returns a temporary lease on its bytes, and the returned token
/// must be released when the caller is done. Today tokens are always `0` and release
/// is a no-op (the owning `StorageManager` guarantees lifetime); the discipline is
/// what lets a store turn them into real pins later.
///
/// Two contracts callers MUST uphold:
///   - Lease discipline: every `acquire*` is paired with exactly one matching
///     `release*` (no leak, no double-release).
///   - Read/write intent: `acquireConst` signals read-only access, `acquireMut` a
///     write. Do not acquire `Mut` for read-only use.
pub const TensorStore = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        meta: *const fn (ctx: *anyopaque, id: TensorId) StoreError!TensorMeta,
        acquireConst: *const fn (ctx: *anyopaque, id: TensorId) StoreError!ViewConst,
        acquireMut: *const fn (ctx: *anyopaque, id: TensorId) StoreError!ViewMut,
        /// Release a previously acquired read-only lease.
        releaseConst: *const fn (ctx: *anyopaque, token: usize) void,
        /// Release a previously acquired mutable lease.
        releaseMut: *const fn (ctx: *anyopaque, token: usize) void,

        /// Optional runtime hint for cache policy bound to a tensor id.
        ///
        /// If null, callers must assume `.none` semantics.
        sequenceCachePolicyInfo: ?*const fn (ctx: *anyopaque, id: TensorId) SequenceCachePolicyInfo = null,

        /// Optional logical->physical time-index mapper for KV cache tensors.
        ///
        /// If null, mapping defaults to identity with strict bounds checks.
        mapSequenceStep: ?*const fn (ctx: *anyopaque, id: TensorId, logical_t: usize, physical_capacity_tokens: usize) StoreError!usize = null,

        /// Prefetch hint. Non-blocking; a CPU cache hint.
        prefetch: ?*const fn (ctx: *anyopaque, id: TensorId) void = null,

        /// Optional no-copy exchange of two tensor backing buffers.
        ///
        /// This is intended for loop-carried SSA-style state where the loop body
        /// writes `next_state` into a scratch tensor, then the runtime makes that
        /// scratch storage become the carried state for the next iteration.
        /// Implementations must reject tensors with incompatible dtype/shape.
        swapTensors: ?*const fn (ctx: *anyopaque, a: TensorId, b: TensorId) StoreError!void = null,

        /// Optional: if the tensor's bytes live in device-owned buffers (move
        /// semantics), dim-0 chunk `chunk` of them. Null for a host-resident tensor,
        /// or when unset, which treats every tensor as host-resident.
        deviceChunk: ?*const fn (ctx: *anyopaque, id: TensorId, chunk: usize) StoreError!?DeviceChunkRef = null,
    };

    pub fn meta(self: TensorStore, id: TensorId) StoreError!TensorMeta {
        return self.vtable.meta(self.ctx, id);
    }

    pub fn acquireConst(self: TensorStore, id: TensorId) StoreError!ViewConst {
        return self.vtable.acquireConst(self.ctx, id);
    }

    pub fn acquireMut(self: TensorStore, id: TensorId) StoreError!ViewMut {
        return self.vtable.acquireMut(self.ctx, id);
    }

    pub fn releaseConst(self: TensorStore, token: usize) void {
        return self.vtable.releaseConst(self.ctx, token);
    }

    pub fn releaseMut(self: TensorStore, token: usize) void {
        return self.vtable.releaseMut(self.ctx, token);
    }

    pub fn sequenceCachePolicyInfo(self: TensorStore, id: TensorId) SequenceCachePolicyInfo {
        if (self.vtable.sequenceCachePolicyInfo) |cb| return cb(self.ctx, id);
        return .{};
    }

    pub fn mapSequenceStep(self: TensorStore, id: TensorId, logical_t: usize, physical_capacity_tokens: usize) StoreError!usize {
        if (physical_capacity_tokens == 0) return StoreError.InvalidArgument;
        if (self.vtable.mapSequenceStep) |cb| return cb(self.ctx, id, logical_t, physical_capacity_tokens);
        if (logical_t >= physical_capacity_tokens) return StoreError.InvalidArgument;
        return logical_t;
    }

    pub fn prefetch(self: TensorStore, id: TensorId) void {
        if (self.vtable.prefetch) |p| p(self.ctx, id);
    }

    pub fn swapTensors(self: TensorStore, a: TensorId, b: TensorId) StoreError!void {
        if (self.vtable.swapTensors) |swap| return swap(self.ctx, a, b);
        return StoreError.InvalidArgument;
    }

    /// Null when the store has no device-residency notion or the tensor is
    /// host-resident; otherwise the device buffer of dim-0 chunk `chunk`.
    pub fn deviceChunk(self: TensorStore, id: TensorId, chunk: usize) StoreError!?DeviceChunkRef {
        if (self.vtable.deviceChunk) |cb| return cb(self.ctx, id, chunk);
        return null;
    }
};

/// Rank ceiling for inline coordinate buffers, matching `graph.MAX_RANK`.
pub const max_rank: usize = 8;
