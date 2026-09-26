// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const types = @import("../backend/types.zig");
const utils = @import("../backend/utils.zig");
const small_vec = @import("../small_vec.zig");
// Device-memory vtable + handle type. This module is std-only (no wgpu), so
// importing it keeps `storage.zig` GPU-API-agnostic and introduces no import
// cycle: the dependency edge is storage -> runtime/device_memory -> std.
const dm = @import("../runtime/device_memory.zig");

const SmallVec = small_vec.SmallVec;
const INLINE_RANK: usize = 8;

pub const DType = types.DType;

/// Errors for in-memory storage.
///
/// This module is intentionally backend-agnostic.
pub const StorageError = error{
    InvalidArgument,
    OutOfMemory,
};

/// Which device a tensor's bytes currently live on. Move semantics: a tensor is
/// resident on exactly one device at a time. `.gpu` uses `index` to name which GPU
/// in the Context's device registry (multi-GPU); `.cpu` ignores `index`.
pub const DeviceRef = packed struct(u16) {
    kind: enum(u8) { cpu = 0, gpu = 1 } = .cpu,
    index: u8 = 0,

    pub fn isCpu(self: DeviceRef) bool {
        return self.kind == .cpu;
    }
    pub fn eql(a: DeviceRef, b: DeviceRef) bool {
        return a.kind == b.kind and a.index == b.index;
    }
};

/// A file mapping tensors borrow their bytes from, alive while any of them does.
///
/// A loaded model's weights are its mapped `.aion` payloads, used in place: the
/// in-memory layout is the file layout, so a weight needs no allocation and no
/// copy, and its pages are the page cache's to drop and fault back. The mapping is
/// read-only; a tensor about to be written takes a private copy first
/// (`Tensor.ensureWritable`), and the last tensor to let go unmaps the file.
pub const Mapping = struct {
    gpa: std.mem.Allocator,
    map: std.Io.File.MemoryMap,
    refs: std.atomic.Value(usize),

    /// Take `map` over, holding one reference for the caller.
    pub fn adopt(gpa: std.mem.Allocator, map: std.Io.File.MemoryMap) StorageError!*Mapping {
        const self = gpa.create(Mapping) catch return StorageError.OutOfMemory;
        self.* = .{ .gpa = gpa, .map = map, .refs = .init(1) };
        return self;
    }

    pub fn retain(self: *Mapping) void {
        _ = self.refs.fetchAdd(1, .monotonic);
    }

    pub fn release(self: *Mapping) void {
        if (self.refs.fetchSub(1, .acq_rel) != 1) return;
        var io_backend: std.Io.Threaded = .init_single_threaded;
        io_backend.allocator = self.gpa;
        self.map.destroy(io_backend.io());
        self.gpa.destroy(self);
    }
};

/// Byte range of one dim-0 chunk inside the packed layout.
pub const Chunk = struct { offset: usize, len: usize, rows: usize };

/// A tensor: one row-major buffer.
///
/// Quantized layout: a quantized tensor has one *block axis* (`quant_axis`); along
/// it every `block_elems` consecutive elements form one `block_bytes` block, and
/// the bytes are row-major over the block-space shape (the shape with
/// `shape[quant_axis]` divided by `block_elems`) — exactly the `.aion` file layout.
///   - `quant_axis = rank-2` for matmul-B weights `[..., K, N]` → blocks along K.
///   - `quant_axis = last` for embedding tables `[V, D]` → one row of the table is a
///     contiguous run of `D / block_elems` blocks.
///
/// On a device the bytes may be split along dim 0 into chunks of `chunk_rows` rows,
/// each its own device buffer (`storage/layout.zig` says when). Every chunk is a
/// contiguous run of the packed layout, so a chunked tensor needs no translation to
/// or from the host: chunk `i` is bytes `chunk(i).offset ..` of the packed buffer.
pub const Tensor = struct {
    allocator: std.mem.Allocator,

    dtype: DType,
    rank: u8,
    /// Block axis for quantized tensors. Ignored for scalar dtypes.
    quant_axis: u8 = 0,
    /// Block order of a quantized weight. In-memory only: a layout pass sets it on
    /// the weight it derives, and nothing writes it to a package.
    block_order: types.QuantBlockOrder = .row_major,
    shape: []const usize,
    shape_storage: SmallVec(usize, INLINE_RANK),
    /// Rows of dim 0 in each device chunk: `shape[0]` for a tensor in one buffer,
    /// which is every host tensor.
    chunk_rows: usize,

    /// Host bytes (packed layout). Empty on a device or when released.
    data: []align(64) u8,
    owns_data: bool = true,
    /// The file mapping `data` is a view of, for a tensor borrowing its bytes from
    /// one; `data` is then read-only (see `ensureWritable`).
    mapping: ?*Mapping = null,

    // --- Device residency (move semantics) ---
    // A tensor lives on exactly one device. On `.cpu`, bytes are in `data` and
    // `chunk_handles` is empty. On a `.gpu` device (after `StorageManager.moveTensor`),
    // `data` is freed and the bytes live in per-chunk device buffers named by
    // `chunk_handles`, allocated from `dev`.
    // Disambiguation: `data.len == 0 && device.kind == .cpu` means released/dead;
    // `data.len == 0 && device.kind == .gpu` means live on the device.
    device: DeviceRef = .{},
    /// Owned: freed (and each handle released via `dev`) in `deinit`.
    chunk_handles: []dm.DeviceHandle = &[_]dm.DeviceHandle{},
    /// Borrowed (non-owning) device-memory interface for `chunk_handles`.
    dev: ?dm.DeviceMemory = null,
    /// Workspace-only physical alias. Logical layout remains on this tensor;
    /// bytes and device handles are resolved through the canonical owner.
    backing_owner: ?u32 = null,
    /// Bytes reserved by this canonical physical backing.
    backing_bytes: usize = 0,
    workspace_owned: bool = false,
    /// Live compiled programs naming this tensor. Maintained by
    /// `StorageManager.retainTensor`/`releaseTensor`; see those for why the count
    /// lives on the tensor rather than in the model that compiled the program.
    program_refs: u32 = 0,
    /// Holders outside compiled programs — API handles, a builder binding it, a model
    /// with it bound as an input — of a tensor the public API created. Null for one a
    /// model or a pass made, which its maker frees.
    holders: ?u32 = null,

    const Self = @This();

    pub const InitOptions = struct {
        /// Block axis for quantized tensors. Ignored for scalar dtypes.
        /// Must be < rank when the dtype is quantized.
        quant_axis: u8 = 0,
        block_order: types.QuantBlockOrder = .row_major,
        /// Allocate zeroed host bytes. Off, the tensor starts without backing, for one
        /// whose bytes live elsewhere — a workspace slot, or chunks on a device.
        host_data: bool = true,
        /// Rows of dim 0 per device chunk; null keeps the tensor in one buffer.
        chunk_rows: ?usize = null,
        /// Zero the host bytes. Off for a tensor its creator overwrites whole, where
        /// the fill is a full extra pass over memory that is thrown away.
        zero_fill: bool = true,
    };

    pub fn init(self: *Self, allocator: std.mem.Allocator, dtype: DType, shape_in: []const usize, opts: InitOptions) StorageError!void {
        if (shape_in.len == 0 or shape_in.len > INLINE_RANK) return StorageError.InvalidArgument;
        for (shape_in) |d| if (d == 0) return StorageError.InvalidArgument;
        const di = dtype.info();
        if (di.is_quantized) {
            if (opts.quant_axis >= shape_in.len) return StorageError.InvalidArgument;
            if (shape_in[opts.quant_axis] % di.block_elems != 0) return StorageError.InvalidArgument;
        }
        const chunk_rows = opts.chunk_rows orelse shape_in[0];
        if (chunk_rows == 0 or chunk_rows > shape_in[0]) return StorageError.InvalidArgument;
        // A chunk boundary must fall on a whole block when dim 0 is the block axis.
        if (di.is_quantized and opts.quant_axis == 0 and chunk_rows % di.block_elems != 0 and chunk_rows != shape_in[0]) return StorageError.InvalidArgument;

        const shape_storage = SmallVec(usize, INLINE_RANK).initFromSlice(allocator, shape_in) catch return StorageError.OutOfMemory;
        self.* = .{
            .allocator = allocator,
            .dtype = dtype,
            .rank = @intCast(shape_in.len),
            .quant_axis = opts.quant_axis,
            .block_order = opts.block_order,
            .shape = &[_]usize{},
            .shape_storage = shape_storage,
            .chunk_rows = chunk_rows,
            .data = &[_]u8{},
        };
        self.shape = self.shape_storage.constSlice();
        errdefer self.deinit();

        const bytes = try self.byteLen();
        if (!opts.host_data) return;
        // Allocate backing buffer aligned for SIMD-friendly accesses.
        const data: []align(64) u8 = allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), bytes) catch return StorageError.OutOfMemory;
        if (opts.zero_fill) @memset(data, 0);
        self.data = data;
        self.backing_bytes = data.len;
    }

    /// A host tensor whose bytes are `bytes` of `mapping`, used in place. `bytes`
    /// must be the tensor's exact packed length and 64-byte aligned.
    pub fn initMapped(self: *Self, allocator: std.mem.Allocator, dtype: DType, shape_in: []const usize, bytes: []const u8, mapping: *Mapping, opts: InitOptions) StorageError!void {
        var no_bytes = opts;
        no_bytes.host_data = false;
        no_bytes.chunk_rows = null;
        try self.init(allocator, dtype, shape_in, no_bytes);
        errdefer self.deinit();
        if (bytes.len != try self.byteLen() or !std.mem.isAligned(@intFromPtr(bytes.ptr), 64)) return StorageError.InvalidArgument;
        self.data = @alignCast(@constCast(bytes));
        self.owns_data = false;
        self.backing_bytes = bytes.len;
        self.mapping = mapping;
        mapping.retain();
    }

    /// Let go of the host bytes: free them if owned, drop the mapping if borrowed.
    fn dropHostBytes(self: *Self) void {
        if (self.data.len != 0 and self.owns_data) self.allocator.free(self.data);
        if (self.mapping) |m| m.release();
        self.mapping = null;
        self.data = &[_]u8{};
        self.owns_data = true;
    }

    /// Make the host bytes safe to write: a view of a read-only mapping becomes a
    /// private copy. A no-op for owned bytes.
    pub fn ensureWritable(self: *Self) StorageError!void {
        if (self.mapping != null) _ = try self.promoteToOwned();
    }

    /// Free this tensor's bytes, keeping metadata so the id stays valid for
    /// metadata-only uses (e.g. external-binding shape/dtype validation). Used to
    /// reclaim a weight that an optimization pass has derived away; executing
    /// against it afterward is a bug. Idempotent.
    pub fn releaseData(self: *Self) void {
        if (self.backing_owner != null) return;
        self.freeDeviceChunks();
        self.device = .{};
        self.dropHostBytes();
        self.backing_bytes = 0;
    }

    pub fn deinit(self: *Self) void {
        self.freeDeviceChunks();
        self.device = .{};
        self.dropHostBytes();
        self.backing_owner = null;
        self.backing_bytes = 0;
        self.shape_storage.deinit();
        self.shape = &[_]usize{};
        self.rank = 0;
    }

    /// Release the device buffers (through the borrowed `dev`) and the handle slice.
    fn freeDeviceChunks(self: *Self) void {
        if (self.chunk_handles.len != 0) {
            if (self.dev) |d| {
                for (self.chunk_handles) |h| d.free(h);
            }
            self.allocator.free(self.chunk_handles);
            self.chunk_handles = &[_]dm.DeviceHandle{};
        }
        self.dev = null;
    }

    pub fn isOwned(self: *const Self) bool {
        return self.owns_data;
    }

    /// True when the tensor's bytes live on a non-cpu device (host `data` freed).
    pub fn onDevice(self: *const Self) bool {
        return self.device.kind != .cpu;
    }

    pub fn elemCount(self: *const Self) usize {
        var n: usize = 1;
        for (self.shape) |d| n *= d;
        return n;
    }

    /// Bytes of the packed layout (row-major; block space for quant).
    pub fn byteLen(self: *const Self) StorageError!usize {
        return utils.requiredBytesForElems(self.dtype, self.elemCount()) catch StorageError.InvalidArgument;
    }

    pub fn chunkCount(self: *const Self) usize {
        return (self.shape[0] + self.chunk_rows - 1) / self.chunk_rows;
    }

    /// Where chunk `i` sits in the packed layout.
    pub fn chunk(self: *const Self, i: usize) Chunk {
        const row_elems = self.elemCount() / self.shape[0];
        const first = i * self.chunk_rows;
        const rows = @min(self.chunk_rows, self.shape[0] - first);
        // Chunks hold whole blocks (checked at init), so these divide evenly.
        const offset = utils.requiredBytesForElems(self.dtype, first * row_elems) catch unreachable;
        const len = utils.requiredBytesForElems(self.dtype, rows * row_elems) catch unreachable;
        return .{ .offset = offset, .len = len, .rows = rows };
    }

    pub fn promoteToOwned(self: *Self) StorageError!bool {
        // Nothing to promote when the bytes live on a device (no host buffer).
        if (self.onDevice()) return false;
        if (self.owns_data) return false;

        const owned_copy: []align(64) u8 = self.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), self.data.len) catch return StorageError.OutOfMemory;
        @memcpy(owned_copy, self.data);
        self.dropHostBytes();
        self.data = owned_copy;
        self.owns_data = true;
        return true;
    }

    /// Grow one axis of a host tensor, keeping every existing element where it is in
    /// logical coordinates; new positions are zero. Scalar dtypes only.
    pub fn growAxisPreserveScalar(self: *Self, axis: usize, new_size: usize) StorageError!void {
        const rank: usize = self.rank;
        if (rank == 0 or axis >= rank) return StorageError.InvalidArgument;
        if (self.dtype.info().is_quantized) return StorageError.InvalidArgument;
        // Host-bytes only. A device tensor grows through
        // `StorageManager.ensureTensorAxisCapacity`, which stays on the device.
        if (self.onDevice()) return StorageError.InvalidArgument;

        const old_size: usize = self.shape[axis];
        if (new_size < old_size) return StorageError.InvalidArgument;
        if (new_size == old_size) return;

        // [outer, axis, inner]: each outer index is one contiguous run of `axis * inner`.
        const eb: usize = self.dtype.info().block_bytes;
        var outer: usize = 1;
        for (self.shape[0..axis]) |d| outer *= d;
        var inner: usize = 1;
        for (self.shape[axis + 1 ..]) |d| inner *= d;
        const old_run = old_size * inner * eb;
        const new_run = new_size * inner * eb;
        const new_bytes = std.math.mul(usize, outer, new_run) catch return StorageError.InvalidArgument;

        const data: []align(64) u8 = self.allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), new_bytes) catch return StorageError.OutOfMemory;
        @memset(data, 0);
        for (0..outer) |o| @memcpy(data[o * new_run ..][0..old_run], self.data[o * old_run ..][0..old_run]);

        self.dropHostBytes();
        self.data = data;
        self.owns_data = true;
        self.backing_bytes = new_bytes;
        self.shape_storage.slice()[axis] = new_size;
        self.shape = self.shape_storage.constSlice();
        self.chunk_rows = self.shape[0];
    }

    /// The host bytes, as `data` names them: the whole packed layout.
    fn hostBytes(self: *const Self, need: usize) StorageError![]u8 {
        if (self.onDevice()) return StorageError.InvalidArgument; // host bytes freed; migrate with .to(.cpu) first
        if (self.data.len < need) return StorageError.InvalidArgument;
        return self.data[0..need];
    }

    /// Writes a packed row-major scalar tensor.
    pub fn writeFromPackedScalar(self: *Self, packed_bytes: []const u8) StorageError!void {
        if (self.dtype.info().is_quantized) return StorageError.InvalidArgument;
        return self.writeBytes(0, packed_bytes, true);
    }

    /// Reads the tensor back as packed row-major scalars.
    pub fn readToPackedScalar(self: *const Self, out: []u8) StorageError!void {
        if (self.dtype.info().is_quantized) return StorageError.InvalidArgument;
        return self.readBytes(0, out, true);
    }

    /// Read elements `[first_elem, first_elem + out.len / elem_bytes)` of the packed
    /// row-major layout; the ranged counterpart of `readToPackedScalar`.
    pub fn readScalarRange(self: *const Self, first_elem: usize, out: []u8) StorageError!void {
        const di = self.dtype.info();
        if (di.is_quantized) return StorageError.InvalidArgument;
        return self.readBytes(first_elem * di.block_bytes, out, false);
    }

    /// Write elements `[first_elem, first_elem + bytes.len / elem_bytes)` of the packed
    /// row-major layout; the ranged counterpart of `writeFromPackedScalar`.
    pub fn writeScalarRange(self: *Self, first_elem: usize, packed_bytes: []const u8) StorageError!void {
        const di = self.dtype.info();
        if (di.is_quantized) return StorageError.InvalidArgument;
        return self.writeBytes(first_elem * di.block_bytes, packed_bytes, false);
    }

    /// Writes a packed quant tensor: row-major over the block-space shape, each
    /// element one block.
    pub fn writeFromPackedQuant(self: *Self, packed_bytes: []const u8) StorageError!void {
        if (!self.dtype.info().is_quantized) return StorageError.InvalidArgument;
        return self.writeBytes(0, packed_bytes, true);
    }

    /// Reads the tensor back in the packed quant convention.
    pub fn readToPackedQuant(self: *const Self, out: []u8) StorageError!void {
        if (!self.dtype.info().is_quantized) return StorageError.InvalidArgument;
        return self.readBytes(0, out, true);
    }

    /// Read packed quant blocks `[first_block, first_block + out.len / block_bytes)`;
    /// the ranged counterpart of `readToPackedQuant`.
    pub fn readQuantBlocks(self: *const Self, first_block: usize, out: []u8) StorageError!void {
        const di = self.dtype.info();
        if (!di.is_quantized or out.len % di.block_bytes != 0) return StorageError.InvalidArgument;
        return self.readBytes(first_block * di.block_bytes, out, false);
    }

    /// Copy `src` into the packed layout at byte `at`. `whole` takes exactly the
    /// tensor's bytes from the front of `src` (which may be longer).
    fn writeBytes(self: *Self, at: usize, src: []const u8, whole: bool) StorageError!void {
        const total = try self.byteLen();
        try self.ensureWritable();
        const bytes = try self.hostBytes(total);
        const n = if (whole) total else src.len;
        if (src.len < n or at + n > total) return StorageError.InvalidArgument;
        @memcpy(bytes[at..][0..n], src[0..n]);
    }

    fn readBytes(self: *const Self, at: usize, out: []u8, whole: bool) StorageError!void {
        const total = try self.byteLen();
        const bytes = try self.hostBytes(total);
        const n = if (whole) total else out.len;
        if (out.len < n or at + n > total) return StorageError.InvalidArgument;
        @memcpy(out[0..n], bytes[at..][0..n]);
    }
};

test "a chunked tensor's chunks are consecutive runs of its packed layout" {
    var t: Tensor = undefined;
    try t.init(std.testing.allocator, .f32, &.{ 10, 3 }, .{ .host_data = false, .chunk_rows = 4 });
    defer t.deinit();
    try std.testing.expectEqual(@as(usize, 3), t.chunkCount());
    try std.testing.expectEqual(Chunk{ .offset = 0, .len = 48, .rows = 4 }, t.chunk(0));
    try std.testing.expectEqual(Chunk{ .offset = 96, .len = 24, .rows = 2 }, t.chunk(2));
}

test "growing an axis keeps every element at its logical position" {
    var t: Tensor = undefined;
    try t.init(std.testing.allocator, .f32, &.{ 2, 2, 3 }, .{});
    defer t.deinit();
    const vals = [_]f32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    try t.writeFromPackedScalar(std.mem.sliceAsBytes(&vals));
    try t.growAxisPreserveScalar(1, 3);
    var out: [18]f32 = undefined;
    try t.readToPackedScalar(std.mem.sliceAsBytes(&out));
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 2, 3, 4, 5, 0, 0, 0, 6, 7, 8, 9, 10, 11, 0, 0, 0 }, &out);
}
