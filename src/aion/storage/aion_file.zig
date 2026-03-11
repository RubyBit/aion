const std = @import("std");
const builtin = @import("builtin");

const storage_mod = @import("storage.zig");
const types = @import("../backend/types.zig");

const windows = std.os.windows;

pub const DType = types.DType;
pub const TiledTensor = storage_mod.TiledTensor;

pub const FileError = error{
    InvalidArgument,
    InvalidFormat,
    UnsupportedVersion,
    OutOfMemory,
    IoFailure,
    CorruptData,
};

pub const magic_bytes: [4]u8 = .{ 'A', 'I', 'O', 'N' };
pub const current_version: u32 = 1;
pub const header_size: usize = 64;
pub const descriptor_size: usize = 192;
pub const kv_start_offset: usize = header_size;

pub const HeaderFlags = struct {
    pub const little_endian: u64 = 1 << 0;
    pub const crc32_present: u64 = 1 << 1;
};

pub const ValueType = enum(u8) {
    i32 = 0,
    f32 = 1,
    string = 2,
    json_blob = 3,
};

pub const MetadataEntry = struct {
    key: []const u8,
    value_type: ValueType,
    value: []const u8,
};

pub const MetadataSource = struct {
    key: []const u8,
    value_type: ValueType,
    value: []const u8,

    pub fn string(key: []const u8, value: []const u8) MetadataSource {
        return .{ .key = key, .value_type = .string, .value = value };
    }
};

pub const TensorSource = struct {
    name: []const u8,
    tensor: *const TiledTensor,
};

pub const Header = struct {
    version: u32,
    flags: u64,
    kv_count: u64,
    tensor_count: u64,
    registry_offset: u64,
    data_offset: u64,
};

pub const TensorDescriptor = struct {
    name: []const u8,
    dtype: DType,
    rank: u8,
    flags: u16,
    shape_mem: [8]u64,
    tile_shape_mem: [8]u64,
    tensor_data_offset: u64,
    tensor_data_size: u64,
    tile_offsets_offset: u64,
    tile_lens_offset: u64,
    tile_count: u64,
    crc32: u32,

    pub fn shape(self: *const TensorDescriptor) []const u64 {
        return self.shape_mem[0..@as(usize, self.rank)];
    }

    pub fn tileShape(self: *const TensorDescriptor) []const u64 {
        return self.tile_shape_mem[0..@as(usize, self.rank)];
    }
};

pub const View = struct {
    bytes: []const u8,
    header: Header,

    pub fn metadata(self: View, index: usize) FileError!MetadataEntry {
        const count: usize = castU64ToUsize(self.header.kv_count) orelse return FileError.InvalidFormat;
        if (index >= count) return FileError.InvalidArgument;

        var cursor: usize = kv_start_offset;
        var i: usize = 0;
        while (i < count) : (i += 1) {
            const entry: MetadataEntry = try parseMetadataEntryAt(self.bytes, &cursor);
            if (i == index) return entry;
        }
        return FileError.InvalidFormat;
    }

    pub fn tensor(self: View, index: usize) FileError!TensorDescriptor {
        const count: usize = castU64ToUsize(self.header.tensor_count) orelse return FileError.InvalidFormat;
        if (index >= count) return FileError.InvalidArgument;

        const registry_offset: usize = castU64ToUsize(self.header.registry_offset) orelse return FileError.InvalidFormat;
        const entry_off: usize = registry_offset + index * descriptor_size;
        return parseTensorDescriptor(self.bytes, self.header, entry_off);
    }

    pub fn findTensor(self: View, name: []const u8) FileError!?TensorDescriptor {
        const count: usize = castU64ToUsize(self.header.tensor_count) orelse return FileError.InvalidFormat;
        var i: usize = 0;
        while (i < count) : (i += 1) {
            const desc: TensorDescriptor = try self.tensor(i);
            if (std.mem.eql(u8, desc.name, name)) return desc;
        }
        return null;
    }

    pub fn tensorDataBytes(self: View, desc: TensorDescriptor) FileError![]const u8 {
        const start: usize = castU64ToUsize(desc.tensor_data_offset) orelse return FileError.InvalidFormat;
        const len: usize = castU64ToUsize(desc.tensor_data_size) orelse return FileError.InvalidFormat;
        return sliceAt(self.bytes, start, len);
    }

    pub fn tileOffsetAt(self: View, desc: TensorDescriptor, index: usize) FileError!u64 {
        if (index >= desc.tile_count) return FileError.InvalidArgument;
        const table_off: usize = castU64ToUsize(desc.tile_offsets_offset) orelse return FileError.InvalidFormat;
        return readIntAt(self.bytes, table_off + index * @sizeOf(u64), u64);
    }

    pub fn tileLenAt(self: View, desc: TensorDescriptor, index: usize) FileError!u64 {
        if (index >= desc.tile_count) return FileError.InvalidArgument;
        const table_off: usize = castU64ToUsize(desc.tile_lens_offset) orelse return FileError.InvalidFormat;
        return readIntAt(self.bytes, table_off + index * @sizeOf(u64), u64);
    }

    pub fn tileBytes(self: View, desc: TensorDescriptor, index: usize) FileError![]const u8 {
        const rel_off: u64 = try self.tileOffsetAt(desc, index);
        const len_u64: u64 = try self.tileLenAt(desc, index);
        const base: u64 = desc.tensor_data_offset;
        const start_u64: u64 = std.math.add(u64, base, rel_off) catch return FileError.InvalidFormat;
        const start: usize = castU64ToUsize(start_u64) orelse return FileError.InvalidFormat;
        const len: usize = castU64ToUsize(len_u64) orelse return FileError.InvalidFormat;
        const data = try self.tensorDataBytes(desc);
        const tensor_start: usize = castU64ToUsize(base) orelse return FileError.InvalidFormat;
        if (start < tensor_start) return FileError.InvalidFormat;
        const rel: usize = start - tensor_start;
        if (rel > data.len or len > data.len - rel) return FileError.InvalidFormat;
        return data[rel .. rel + len];
    }

    pub fn validateTensorCrc32(self: View, desc: TensorDescriptor) FileError!void {
        if ((self.header.flags & HeaderFlags.crc32_present) == 0) return;
        const data = try self.tensorDataBytes(desc);
        const actual: u32 = std.hash.Crc32.hash(data);
        if (actual != desc.crc32) return FileError.CorruptData;
    }
};

pub const WriteOptions = struct {
    data_alignment: usize = 64,
};

pub const MappedFile = struct {
    bytes: []align(std.heap.pageSize()) const u8,
    impl: Impl,

    const Impl = if (builtin.os.tag == .windows)
        struct {
            section_handle: windows.HANDLE,
        }
    else
        void;

    pub fn mapReadOnly(file: std.fs.File) FileError!MappedFile {
        const end_pos: u64 = file.getEndPos() catch return FileError.IoFailure;
        const size: usize = castU64ToUsize(end_pos) orelse return FileError.InvalidFormat;
        if (size < header_size) return FileError.InvalidFormat;

        if (builtin.os.tag == .windows) {
            var section_handle: windows.HANDLE = undefined;
            const create_section_rc = windows.ntdll.NtCreateSection(
                &section_handle,
                windows.STANDARD_RIGHTS_REQUIRED | windows.SECTION_QUERY | windows.SECTION_MAP_READ,
                null,
                null,
                windows.PAGE_READONLY,
                windows.SEC_COMMIT,
                file.handle,
            );
            if (create_section_rc != .SUCCESS) return FileError.IoFailure;
            errdefer windows.CloseHandle(section_handle);

            var view_len: usize = 0;
            var view_ptr: ?[*]const u8 = null;
            const map_section_rc = windows.ntdll.NtMapViewOfSection(
                section_handle,
                windows.GetCurrentProcess(),
                @ptrCast(&view_ptr),
                null,
                0,
                null,
                &view_len,
                .ViewUnmap,
                0,
                windows.PAGE_READONLY,
            );
            if (map_section_rc != .SUCCESS) return FileError.IoFailure;
            errdefer _ = windows.ntdll.NtUnmapViewOfSection(windows.GetCurrentProcess(), @constCast(view_ptr.?));

            const aligned_ptr: [*]align(std.heap.pageSize()) const u8 = @alignCast(view_ptr.?);
            return .{
                .bytes = aligned_ptr[0..view_len],
                .impl = .{ .section_handle = section_handle },
            };
        }

        const page_size: usize = std.heap.pageSize();
        const map_len: usize = std.mem.alignForward(usize, size, page_size);
        const mapped = std.posix.mmap(
            null,
            map_len,
            std.posix.PROT.READ,
            .{ .TYPE = .PRIVATE },
            file.handle,
            0,
        ) catch |err| switch (err) {
            error.AccessDenied, error.PermissionDenied => return FileError.IoFailure,
            error.OutOfMemory, error.SystemFdQuotaExceeded, error.ProcessFdQuotaExceeded, error.LockedMemoryLimitExceeded => return FileError.OutOfMemory,
            error.MemoryMappingNotSupported, error.MappingAlreadyExists => return FileError.IoFailure,
            else => return FileError.IoFailure,
        };

        return .{ .bytes = mapped, .impl = {} };
    }

    pub fn logicalBytes(self: *const MappedFile, file_size: usize) FileError![]align(std.heap.pageSize()) const u8 {
        if (file_size > self.bytes.len) return FileError.InvalidFormat;
        return self.bytes[0..file_size];
    }

    pub fn deinit(self: *MappedFile) void {
        if (builtin.os.tag == .windows) {
            _ = windows.ntdll.NtUnmapViewOfSection(windows.GetCurrentProcess(), @constCast(self.bytes.ptr));
            windows.CloseHandle(self.impl.section_handle);
        } else {
            std.posix.munmap(self.bytes);
        }
        self.* = undefined;
    }
};

pub fn parse(bytes: []const u8) FileError!View {
    if (bytes.len < header_size) return FileError.InvalidFormat;

    var cursor: usize = 0;
    const magic = try readBytes(bytes, &cursor, 4);
    if (!std.mem.eql(u8, magic, &magic_bytes)) return FileError.InvalidFormat;

    const header: Header = .{
        .version = try readIntCursor(bytes, &cursor, u32),
        .flags = try readIntCursor(bytes, &cursor, u64),
        .kv_count = try readIntCursor(bytes, &cursor, u64),
        .tensor_count = try readIntCursor(bytes, &cursor, u64),
        .registry_offset = try readIntCursor(bytes, &cursor, u64),
        .data_offset = try readIntCursor(bytes, &cursor, u64),
    };
    _ = try readBytes(bytes, &cursor, 16);

    if (header.version != current_version) return FileError.UnsupportedVersion;
    if ((header.flags & HeaderFlags.little_endian) == 0) return FileError.InvalidFormat;

    const registry_offset: usize = castU64ToUsize(header.registry_offset) orelse return FileError.InvalidFormat;
    const data_offset: usize = castU64ToUsize(header.data_offset) orelse return FileError.InvalidFormat;
    const tensor_count: usize = castU64ToUsize(header.tensor_count) orelse return FileError.InvalidFormat;

    if (registry_offset < header_size or registry_offset > bytes.len) return FileError.InvalidFormat;
    if (data_offset < registry_offset or data_offset > bytes.len) return FileError.InvalidFormat;

    var kv_cursor: usize = kv_start_offset;
    var kv_index: usize = 0;
    while (kv_index < header.kv_count) : (kv_index += 1) {
        _ = try parseMetadataEntryAt(bytes, &kv_cursor);
    }
    if (kv_cursor != registry_offset) return FileError.InvalidFormat;

    const registry_bytes: usize = tensor_count * descriptor_size;
    if (registry_offset > data_offset or registry_bytes > data_offset - registry_offset) return FileError.InvalidFormat;

    var i: usize = 0;
    while (i < tensor_count) : (i += 1) {
        const entry_off: usize = registry_offset + i * descriptor_size;
        const desc: TensorDescriptor = try parseTensorDescriptor(bytes, header, entry_off);
        try validateTensorDescriptor(bytes, header, desc);
    }

    return .{ .bytes = bytes, .header = header };
}

pub fn readAlloc(allocator: std.mem.Allocator, file: std.fs.File) FileError![]align(64) u8 {
    const end_pos: u64 = file.getEndPos() catch return FileError.IoFailure;
    const size: usize = castU64ToUsize(end_pos) orelse return FileError.InvalidFormat;
    const buf: []align(64) u8 = allocator.alignedAlloc(u8, std.mem.Alignment.fromByteUnits(64), size) catch return FileError.OutOfMemory;
    errdefer allocator.free(buf);

    file.seekTo(0) catch return FileError.IoFailure;
    var io_backend: std.Io.Threaded = .init_single_threaded;
    const io: std.Io = io_backend.io();
    var read_buffer: [4096]u8 = undefined;
    var reader = file.reader(io, &read_buffer);
    reader.interface.readSliceAll(buf) catch return FileError.IoFailure;
    return buf;
}

pub fn writeFile(file: std.fs.File, metadata: []const MetadataSource, tensors: []const TensorSource, opts: WriteOptions) FileError!void {
    if (opts.data_alignment == 0 or !std.math.isPowerOfTwo(opts.data_alignment)) return FileError.InvalidArgument;

    var layouts_mem: [32]TensorLayout = undefined;
    if (tensors.len > layouts_mem.len) return FileError.InvalidArgument;
    var layouts: []TensorLayout = layouts_mem[0..tensors.len];

    var cursor: u64 = header_size;
    for (metadata) |entry| {
        validateMetadataSource(entry) catch |e| return e;
        cursor = std.math.add(u64, cursor, metadataEntrySize(entry)) catch return FileError.InvalidArgument;
    }

    const registry_offset: u64 = cursor;
    cursor = std.math.add(u64, cursor, @as(u64, @intCast(tensors.len * descriptor_size))) catch return FileError.InvalidArgument;

    for (tensors, 0..) |src, i| {
        validateTensorSource(src, opts) catch |e| return e;
        layouts[i] = .{
            .name_offset = cursor,
            .tile_offsets_offset = 0,
            .tile_lens_offset = 0,
            .tensor_data_offset = 0,
            .tensor_data_size = @intCast(src.tensor.data.len),
            .tile_count = @intCast(src.tensor.tile_offsets.len),
            .crc32 = std.hash.Crc32.hash(src.tensor.data),
        };
        cursor = std.math.add(u64, cursor, @as(u64, @intCast(src.name.len))) catch return FileError.InvalidArgument;
    }

    for (layouts) |*layout| {
        layout.tile_offsets_offset = cursor;
        cursor = std.math.add(u64, cursor, layout.tile_count * @sizeOf(u64)) catch return FileError.InvalidArgument;
    }
    for (layouts) |*layout| {
        layout.tile_lens_offset = cursor;
        cursor = std.math.add(u64, cursor, layout.tile_count * @sizeOf(u64)) catch return FileError.InvalidArgument;
    }

    const data_offset: u64 = alignForwardU64(cursor, opts.data_alignment) catch return FileError.InvalidArgument;
    cursor = data_offset;

    for (layouts, tensors) |*layout, src| {
        cursor = alignForwardU64(cursor, opts.data_alignment) catch return FileError.InvalidArgument;
        layout.tensor_data_offset = cursor;
        cursor = std.math.add(u64, cursor, @as(u64, @intCast(src.tensor.data.len))) catch return FileError.InvalidArgument;
    }

    file.seekTo(0) catch return FileError.IoFailure;
    const writer = file;

    writeAll(writer, &magic_bytes) catch return FileError.IoFailure;
    try writeInt(writer, u32, current_version);
    try writeInt(writer, u64, HeaderFlags.little_endian | HeaderFlags.crc32_present);
    try writeInt(writer, u64, @intCast(metadata.len));
    try writeInt(writer, u64, @intCast(tensors.len));
    try writeInt(writer, u64, registry_offset);
    try writeInt(writer, u64, data_offset);
    try writeZeroes(writer, 16);

    for (metadata) |entry| {
        try writeMetadataEntry(writer, entry);
    }

    for (tensors, layouts) |src, layout| {
        try writeTensorDescriptor(writer, src, layout);
    }

    for (tensors) |src| {
        writeAll(writer, src.name) catch return FileError.IoFailure;
    }

    for (tensors) |src| {
        for (src.tensor.tile_offsets) |off| {
            try writeInt(writer, u64, @intCast(off));
        }
    }

    for (tensors) |src| {
        for (src.tensor.tile_lens) |len| {
            try writeInt(writer, u64, @intCast(len));
        }
    }

    var written: u64 = cursorAfterMetadata(metadata, tensors) catch return FileError.InvalidArgument;
    if (written > data_offset) return FileError.InvalidFormat;
    try writeZeroes(writer, @intCast(data_offset - written));
    written = data_offset;

    for (tensors, layouts) |src, layout| {
        if (written > layout.tensor_data_offset) return FileError.InvalidFormat;
        try writeZeroes(writer, @intCast(layout.tensor_data_offset - written));
        writeAll(writer, src.tensor.data) catch return FileError.IoFailure;
        written = layout.tensor_data_offset + @as(u64, @intCast(src.tensor.data.len));
    }

    file.setEndPos(written) catch return FileError.IoFailure;
}

const TensorLayout = struct {
    name_offset: u64,
    tile_offsets_offset: u64,
    tile_lens_offset: u64,
    tensor_data_offset: u64,
    tensor_data_size: u64,
    tile_count: u64,
    crc32: u32,
};

fn validateMetadataSource(entry: MetadataSource) FileError!void {
    if (entry.key.len == 0) return FileError.InvalidArgument;
    switch (entry.value_type) {
        .i32, .f32 => if (entry.value.len != 4) return FileError.InvalidArgument,
        .string, .json_blob => {},
    }
}

fn metadataEntrySize(entry: MetadataSource) u64 {
    return 4 + entry.key.len + 1 + 4 + entry.value.len;
}

fn cursorAfterMetadata(metadata: []const MetadataSource, tensors: []const TensorSource) FileError!u64 {
    var cursor: u64 = header_size;
    for (metadata) |entry| {
        cursor = std.math.add(u64, cursor, metadataEntrySize(entry)) catch return FileError.InvalidArgument;
    }
    cursor = std.math.add(u64, cursor, @as(u64, @intCast(tensors.len * descriptor_size))) catch return FileError.InvalidArgument;
    for (tensors) |src| {
        cursor = std.math.add(u64, cursor, @as(u64, @intCast(src.name.len))) catch return FileError.InvalidArgument;
    }
    for (tensors) |src| {
        cursor = std.math.add(u64, cursor, @as(u64, @intCast(src.tensor.tile_offsets.len * @sizeOf(u64)))) catch return FileError.InvalidArgument;
    }
    for (tensors) |src| {
        cursor = std.math.add(u64, cursor, @as(u64, @intCast(src.tensor.tile_lens.len * @sizeOf(u64)))) catch return FileError.InvalidArgument;
    }
    return cursor;
}

fn validateTensorSource(src: TensorSource, opts: WriteOptions) FileError!void {
    const tensor: *const TiledTensor = src.tensor;
    if (src.name.len == 0) return FileError.InvalidArgument;
    if (tensor.rank == 0 or tensor.rank > 8) return FileError.InvalidArgument;
    if (tensor.shape.len != tensor.rank) return FileError.InvalidArgument;
    if (tensor.tile_shape.len != tensor.rank) return FileError.InvalidArgument;
    if (tensor.tile_offsets.len != tensor.tile_lens.len) return FileError.InvalidArgument;
    if (tensor.tile_alignment == 0 or !std.math.isPowerOfTwo(tensor.tile_alignment)) return FileError.InvalidArgument;
    if (opts.data_alignment < tensor.tile_alignment) return FileError.InvalidArgument;
    for (tensor.tile_offsets) |off| {
        if (off % tensor.tile_alignment != 0) return FileError.InvalidArgument;
        if (off > tensor.data.len) return FileError.InvalidArgument;
    }
    for (tensor.tile_offsets, tensor.tile_lens) |off, len| {
        if (off > tensor.data.len or len > tensor.data.len - off) return FileError.InvalidArgument;
    }
}

fn parseMetadataEntryAt(bytes: []const u8, cursor: *usize) FileError!MetadataEntry {
    const key_len_u32: u32 = try readIntCursor(bytes, cursor, u32);
    const key_len: usize = castU64ToUsize(key_len_u32) orelse return FileError.InvalidFormat;
    const key = try readBytes(bytes, cursor, key_len);

    const value_type_raw: u8 = try readIntCursor(bytes, cursor, u8);
    const value_type: ValueType = std.meta.intToEnum(ValueType, value_type_raw) catch return FileError.InvalidFormat;

    const value_len_u32: u32 = try readIntCursor(bytes, cursor, u32);
    const value_len: usize = castU64ToUsize(value_len_u32) orelse return FileError.InvalidFormat;
    const value = try readBytes(bytes, cursor, value_len);

    switch (value_type) {
        .i32, .f32 => if (value.len != 4) return FileError.InvalidFormat,
        .string, .json_blob => {},
    }

    return .{ .key = key, .value_type = value_type, .value = value };
}

fn parseTensorDescriptor(bytes: []const u8, header: Header, entry_off: usize) FileError!TensorDescriptor {
    var cursor: usize = entry_off;
    _ = try sliceAt(bytes, entry_off, descriptor_size);

    const name_offset: u64 = try readIntCursor(bytes, &cursor, u64);
    const name_len_u32: u32 = try readIntCursor(bytes, &cursor, u32);
    const dtype_raw: u8 = try readIntCursor(bytes, &cursor, u8);
    const rank: u8 = try readIntCursor(bytes, &cursor, u8);
    const flags: u16 = try readIntCursor(bytes, &cursor, u16);

    if (rank == 0 or rank > 8) return FileError.InvalidFormat;
    const dtype: DType = std.meta.intToEnum(DType, dtype_raw) catch return FileError.InvalidFormat;

    var shape_mem: [8]u64 = .{0} ** 8;
    var tile_shape_mem: [8]u64 = .{0} ** 8;

    var i: usize = 0;
    while (i < 8) : (i += 1) shape_mem[i] = try readIntCursor(bytes, &cursor, u64);
    i = 0;
    while (i < 8) : (i += 1) tile_shape_mem[i] = try readIntCursor(bytes, &cursor, u64);

    const tensor_data_offset: u64 = try readIntCursor(bytes, &cursor, u64);
    const tensor_data_size: u64 = try readIntCursor(bytes, &cursor, u64);
    const tile_offsets_offset: u64 = try readIntCursor(bytes, &cursor, u64);
    const tile_lens_offset: u64 = try readIntCursor(bytes, &cursor, u64);
    const tile_count: u64 = try readIntCursor(bytes, &cursor, u64);
    const crc32: u32 = try readIntCursor(bytes, &cursor, u32);
    _ = try readIntCursor(bytes, &cursor, u32);

    const name_len: usize = castU64ToUsize(name_len_u32) orelse return FileError.InvalidFormat;
    const name = try sliceAt(bytes, castU64ToUsize(name_offset) orelse return FileError.InvalidFormat, name_len);

    _ = header;
    return .{
        .name = name,
        .dtype = dtype,
        .rank = rank,
        .flags = flags,
        .shape_mem = shape_mem,
        .tile_shape_mem = tile_shape_mem,
        .tensor_data_offset = tensor_data_offset,
        .tensor_data_size = tensor_data_size,
        .tile_offsets_offset = tile_offsets_offset,
        .tile_lens_offset = tile_lens_offset,
        .tile_count = tile_count,
        .crc32 = crc32,
    };
}

fn validateTensorDescriptor(bytes: []const u8, header: Header, desc: TensorDescriptor) FileError!void {
    const data_offset: usize = castU64ToUsize(header.data_offset) orelse return FileError.InvalidFormat;
    const tensor_data_off: usize = castU64ToUsize(desc.tensor_data_offset) orelse return FileError.InvalidFormat;
    const tensor_data_len: usize = castU64ToUsize(desc.tensor_data_size) orelse return FileError.InvalidFormat;
    const tile_offsets_off: usize = castU64ToUsize(desc.tile_offsets_offset) orelse return FileError.InvalidFormat;
    const tile_lens_off: usize = castU64ToUsize(desc.tile_lens_offset) orelse return FileError.InvalidFormat;
    const tile_count: usize = castU64ToUsize(desc.tile_count) orelse return FileError.InvalidFormat;

    if (tensor_data_off < data_offset or tensor_data_off > bytes.len) return FileError.InvalidFormat;
    _ = try sliceAt(bytes, tensor_data_off, tensor_data_len);

    const offsets_bytes: usize = tile_count * @sizeOf(u64);
    const lens_bytes: usize = tile_count * @sizeOf(u64);
    if (tile_offsets_off < header.registry_offset or tile_offsets_off > data_offset) return FileError.InvalidFormat;
    if (tile_lens_off < header.registry_offset or tile_lens_off > data_offset) return FileError.InvalidFormat;
    _ = try sliceAt(bytes, tile_offsets_off, offsets_bytes);
    _ = try sliceAt(bytes, tile_lens_off, lens_bytes);

    var dims_tile_count: usize = 1;
    var i: usize = 0;
    while (i < desc.rank) : (i += 1) {
        const dim: usize = castU64ToUsize(desc.shape_mem[i]) orelse return FileError.InvalidFormat;
        const tile_dim: usize = castU64ToUsize(desc.tile_shape_mem[i]) orelse return FileError.InvalidFormat;
        if (dim == 0 or tile_dim == 0) return FileError.InvalidFormat;
        const count: usize = (dim + tile_dim - 1) / tile_dim;
        dims_tile_count = std.math.mul(usize, dims_tile_count, count) catch return FileError.InvalidFormat;
    }
    if (dims_tile_count != tile_count) return FileError.InvalidFormat;

    i = 0;
    while (i < tile_count) : (i += 1) {
        const rel_off: usize = castU64ToUsize(try readIntAt(bytes, tile_offsets_off + i * @sizeOf(u64), u64)) orelse return FileError.InvalidFormat;
        const len: usize = castU64ToUsize(try readIntAt(bytes, tile_lens_off + i * @sizeOf(u64), u64)) orelse return FileError.InvalidFormat;
        if (rel_off > tensor_data_len or len > tensor_data_len - rel_off) return FileError.InvalidFormat;
    }
}

fn writeMetadataEntry(writer: anytype, entry: MetadataSource) FileError!void {
    try writeInt(writer, u32, @intCast(entry.key.len));
    writer.writeAll(entry.key) catch return FileError.IoFailure;
    try writeInt(writer, u8, @intFromEnum(entry.value_type));
    try writeInt(writer, u32, @intCast(entry.value.len));
    writer.writeAll(entry.value) catch return FileError.IoFailure;
}

fn writeTensorDescriptor(writer: anytype, src: TensorSource, layout: TensorLayout) FileError!void {
    const tensor: *const TiledTensor = src.tensor;
    try writeInt(writer, u64, layout.name_offset);
    try writeInt(writer, u32, @intCast(src.name.len));
    try writeInt(writer, u8, @intFromEnum(tensor.dtype));
    try writeInt(writer, u8, tensor.rank);
    try writeInt(writer, u16, 0);

    var i: usize = 0;
    while (i < 8) : (i += 1) {
        const value: u64 = if (i < tensor.shape.len) @intCast(tensor.shape[i]) else 0;
        try writeInt(writer, u64, value);
    }
    i = 0;
    while (i < 8) : (i += 1) {
        const value: u64 = if (i < tensor.tile_shape.len) @intCast(tensor.tile_shape[i]) else 0;
        try writeInt(writer, u64, value);
    }

    try writeInt(writer, u64, layout.tensor_data_offset);
    try writeInt(writer, u64, layout.tensor_data_size);
    try writeInt(writer, u64, layout.tile_offsets_offset);
    try writeInt(writer, u64, layout.tile_lens_offset);
    try writeInt(writer, u64, layout.tile_count);
    try writeInt(writer, u32, layout.crc32);
    try writeInt(writer, u32, 0);
}

fn writeInt(writer: anytype, comptime T: type, value: T) FileError!void {
    var buf: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &buf, value, .little);
    writeAll(writer, &buf) catch return FileError.IoFailure;
}

fn writeZeroes(writer: anytype, count: usize) FileError!void {
    var zeros: [256]u8 = .{0} ** 256;
    var remaining: usize = count;
    while (remaining != 0) {
        const n: usize = @min(remaining, zeros.len);
        writeAll(writer, zeros[0..n]) catch return FileError.IoFailure;
        remaining -= n;
    }
}

fn writeAll(writer: anytype, bytes: []const u8) @TypeOf(writer.writeAll(bytes)) {
    return writer.writeAll(bytes);
}

fn readBytes(bytes: []const u8, cursor: *usize, len: usize) FileError![]const u8 {
    const out = try sliceAt(bytes, cursor.*, len);
    cursor.* += len;
    return out;
}

fn readIntCursor(bytes: []const u8, cursor: *usize, comptime T: type) FileError!T {
    const slice = try readBytes(bytes, cursor, @sizeOf(T));
    return std.mem.readInt(T, slice[0..@sizeOf(T)], .little);
}

fn readIntAt(bytes: []const u8, start: usize, comptime T: type) FileError!T {
    const slice = try sliceAt(bytes, start, @sizeOf(T));
    return std.mem.readInt(T, slice[0..@sizeOf(T)], .little);
}

fn sliceAt(bytes: []const u8, start: usize, len: usize) FileError![]const u8 {
    if (start > bytes.len or len > bytes.len - start) return FileError.InvalidFormat;
    return bytes[start .. start + len];
}

fn castU64ToUsize(value: anytype) ?usize {
    return std.math.cast(usize, value);
}

fn alignForwardU64(value: u64, alignment: usize) FileError!u64 {
    const a_u64: u64 = @intCast(alignment);
    const rem: u64 = value % a_u64;
    if (rem == 0) return value;
    return std.math.add(u64, value, a_u64 - rem) catch FileError.InvalidArgument;
}
