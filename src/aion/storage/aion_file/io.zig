const std = @import("std");
const types = @import("types.zig");
const parse_mod = @import("parse.zig");

pub const PackageError = types.PackageError;
pub const Package = types.Package;

pub fn readAlloc(allocator: std.mem.Allocator, file: std.Io.File) PackageError![]u8 {
    var io_backend: std.Io.Threaded = .init_single_threaded;
    const io = io_backend.io();
    const end_pos: u64 = file.length(io) catch return PackageError.IoFailure;
    const size: usize = std.math.cast(usize, end_pos) orelse return PackageError.InvalidFormat;
    const buf = allocator.alloc(u8, size) catch return PackageError.OutOfMemory;
    errdefer allocator.free(buf);
    const read_len = file.readPositionalAll(io, buf, 0) catch return PackageError.IoFailure;
    if (read_len != buf.len) return PackageError.IoFailure;
    return buf;
}

pub fn readPackageFile(allocator: std.mem.Allocator, file: std.Io.File) PackageError!Package {
    const bytes = try readAlloc(allocator, file);
    defer allocator.free(bytes);
    return parse_mod.parse(allocator, bytes);
}
