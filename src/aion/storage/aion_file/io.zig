// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const builtin = @import("builtin");
const types = @import("types.zig");
const parse_mod = @import("parse.zig");

pub const PackageError = types.PackageError;
pub const Package = types.Package;

/// Most one positional read or write moves: macOS refuses one past `INT_MAX`
/// bytes, and a tensor of a large model passes that.
const max_io: usize = 1 << 30;

/// Fill `buf` from `file` at `offset`, in calls no larger than `max_io`.
pub fn readAt(file: std.Io.File, buf: []u8, offset: u64) PackageError!void {
    var io_backend: std.Io.Threaded = .init_single_threaded;
    const io = io_backend.io();
    var at: usize = 0;
    while (at < buf.len) {
        const part = buf[at..@min(buf.len, at + max_io)];
        const got = file.readPositionalAll(io, part, offset + at) catch return PackageError.IoFailure;
        if (got != part.len) return PackageError.IoFailure;
        at += part.len;
    }
}

/// Write `bytes` to `file` at `offset`, in calls no larger than `max_io`.
pub fn writeAt(file: std.Io.File, bytes: []const u8, offset: u64) PackageError!void {
    var io_backend: std.Io.Threaded = .init_single_threaded;
    const io = io_backend.io();
    var at: usize = 0;
    while (at < bytes.len) {
        const part = bytes[at..@min(bytes.len, at + max_io)];
        file.writePositionalAll(io, part, offset + at) catch return PackageError.IoFailure;
        at += part.len;
    }
}

pub fn readAlloc(allocator: std.mem.Allocator, file: std.Io.File) PackageError![]u8 {
    var io_backend: std.Io.Threaded = .init_single_threaded;
    const io = io_backend.io();
    const end_pos: u64 = file.length(io) catch return PackageError.IoFailure;
    const size: usize = std.math.cast(usize, end_pos) orelse return PackageError.InvalidFormat;
    const buf = allocator.alloc(u8, size) catch return PackageError.OutOfMemory;
    errdefer allocator.free(buf);
    try readAt(file, buf, 0);
    return buf;
}

/// A package file mapped read-only, and the package parsed from it. The package's
/// tensor payloads are views of the mapping, so they are read straight from the page
/// cache — clean pages the OS can drop — rather than from a copy of the file.
pub const MappedPackage = struct {
    /// Null once `unmap` has run.
    map: ?std.Io.File.MemoryMap,
    package: Package,

    pub fn open(gpa: std.mem.Allocator, file: std.Io.File) PackageError!MappedPackage {
        var io_backend = fileIo(gpa);
        const io = io_backend.io();
        const len = std.math.cast(usize, file.length(io) catch return PackageError.IoFailure) orelse return PackageError.InvalidFormat;
        if (len < types.header_size) return PackageError.InvalidFormat;
        var map = file.createMemoryMap(io, .{
            .len = len,
            .protection = .{ .read = true, .write = false },
            // Pages come in as tensors are copied out, not all up front. Windows is
            // the exception in name only: std maps `populate` to `SEC_COMMIT`, which
            // a file-backed section requires (without it NtCreateSection fails with
            // INVALID_PARAMETER_6 and std silently reads the whole file into memory),
            // and which still faults pages in lazily. Drop the special case once
            // https://codeberg.org/ziglang/zig/issues/36960 is fixed.
            .populate = builtin.os.tag == .windows,
        }) catch |e| return switch (e) {
            error.OutOfMemory => PackageError.OutOfMemory,
            else => PackageError.IoFailure,
        };
        errdefer map.destroy(io);
        return .{ .map = map, .package = try parse_mod.parse(gpa, map.memory[0..len]) };
    }

    /// Whether the OS actually mapped the file. When it cannot, std falls back to
    /// reading the whole file into memory -- correct, but it silently costs the
    /// file's size in private memory, so this is what the platform test pins.
    pub fn isMapped(self: *const MappedPackage) bool {
        const map = self.map orelse return false;
        return map.section != null;
    }

    /// The whole file, as mapped; only valid before `unmap`.
    pub fn bytes(self: *const MappedPackage) []const u8 {
        return self.map.?.memory;
    }

    /// Release the mapping once the payloads have been copied out; the package keeps
    /// what it owns and is still the caller's to `deinit`. Idempotent.
    pub fn unmap(self: *MappedPackage) void {
        var map = self.map orelse return;
        self.package.dropPayloads();
        var io_backend = fileIo(self.package.allocator);
        map.destroy(io_backend.io());
        self.map = null;
    }

    pub fn deinit(self: *MappedPackage) void {
        self.unmap();
        self.package.deinit();
    }
};

/// Single-threaded I/O whose allocator is `gpa`: `std.Io.File.MemoryMap` falls back
/// to reading a file into memory it allocates where the OS cannot map it.
fn fileIo(gpa: std.mem.Allocator) std.Io.Threaded {
    var t: std.Io.Threaded = .init_single_threaded;
    t.allocator = gpa;
    return t;
}
