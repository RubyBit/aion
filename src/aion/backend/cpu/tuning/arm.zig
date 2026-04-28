// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const builtin = @import("builtin");
const cpuid_root = @import("cpuid.zig");

pub fn detect() cpuid_root.CpuInfo {
    var info: cpuid_root.CpuInfo = .{
        .arch = .aarch64,
        .features = .{ .neon = true },
    };

    if (builtin.os.tag == .linux or builtin.os.tag == .android) {
        const AT_HWCAP = 16;
        const AT_HWCAP2 = 26;
        const hwcaps = std.os.linux.getauxval(AT_HWCAP);
        const hwcaps2 = std.os.linux.getauxval(AT_HWCAP2);

        const HWCAP_ASIMDDP = 1 << 20;
        const HWCAP2_I8MM = 1 << 13;
        const HWCAP2_SME = 1 << 23;

        info.features.dotprod = (hwcaps & HWCAP_ASIMDDP) != 0;
        info.features.i8mm = (hwcaps2 & HWCAP2_I8MM) != 0;
        info.features.sme = (hwcaps2 & HWCAP2_SME) != 0;
    } else if (builtin.os.tag == .windows) {
        const kernel32 = struct {
            extern "kernel32" fn IsProcessorFeaturePresent(feature: u32) callconv(std.os.windows.WINAPI) std.os.windows.BOOL;
        };
        info.features.dotprod = kernel32.IsProcessorFeaturePresent(43) != 0;
    } else if (builtin.os.tag.isDarwin()) {
        info.features.dotprod = sysctlBool("hw.optional.arm.FEAT_DotProd");
        info.features.i8mm = sysctlBool("hw.optional.arm.FEAT_I8MM");
        info.features.sme = sysctlBool("hw.optional.arm.FEAT_SME");
    }

    info.caches = detectCaches();
    return info;
}

fn sysctlBool(name: [:0]const u8) bool {
    if (!builtin.os.tag.isDarwin()) return false;
    var val: u32 = 0;
    var size: usize = @sizeOf(u32);
    const rc = std.c.sysctlbyname(name, &val, &size, null, 0);
    return rc == 0 and val != 0;
}

fn sysctlU64(name: [:0]const u8) u64 {
    if (!builtin.os.tag.isDarwin()) return 0;
    var val: u64 = 0;
    var size: usize = @sizeOf(u64);
    const rc = std.c.sysctlbyname(name, &val, &size, null, 0);
    return if (rc == 0) val else 0;
}

fn parseSizeToBytes(s: []const u8) usize {
    if (s.len == 0) return 0;
    var end = s.len;
    while (end > 0 and (s[end - 1] == '\n' or s[end - 1] == '\r' or s[end - 1] == ' ' or s[end - 1] == '\t')) : (end -= 1) {}
    if (end == 0) return 0;

    const last = s[end - 1];
    var mult: usize = 1;
    var num_end: usize = end;

    if (last == 'K' or last == 'k') {
        mult = 1024;
        num_end = end - 1;
    } else if (last == 'M' or last == 'm') {
        mult = 1024 * 1024;
        num_end = end - 1;
    } else if (last == 'G' or last == 'g') {
        mult = 1024 * 1024 * 1024;
        num_end = end - 1;
    }

    const num_str = std.mem.trim(u8, s[0..num_end], " \t\r\n");
    const n = std.fmt.parseInt(usize, num_str, 10) catch return 0;
    return n * mult;
}

fn readFileTrimAlloc(alloc: std.mem.Allocator, dir: std.Io.Dir, path: []const u8) ![]u8 {
    var io_backend: std.Io.Threaded = .init_single_threaded;
    const io = io_backend.io();
    const f = try dir.openFile(io, path, .{});
    defer f.close(io);

    const len_u64 = try f.length(io);
    const len: usize = std.math.cast(usize, len_u64) orelse return error.FileTooBig;
    const data = try alloc.alloc(u8, len);
    errdefer alloc.free(data);

    const read_len = try f.readPositionalAll(io, data, 0);
    const trimmed = std.mem.trimRight(u8, data[0..read_len], "\r\n\t ");
    if (trimmed.len == data.len) return data;

    const out = try alloc.dupe(u8, trimmed);
    alloc.free(data);
    return out;
}

fn detectCachesArmLinux(alloc: std.mem.Allocator) cpuid_root.Caches {
    var out: cpuid_root.Caches = .{};
    const base_path = "/sys/devices/system/cpu";
    var io_backend: std.Io.Threaded = .init_single_threaded;
    const io = io_backend.io();
    var cpu_dir = std.Io.Dir.openDirAbsolute(io, base_path, .{ .iterate = true }) catch return out;
    defer cpu_dir.close(io);

    var it = cpu_dir.iterate();
    while (it.next(io) catch null) |e| {
        if (e.kind != .directory) continue;
        if (!std.mem.startsWith(u8, e.name, "cpu")) continue;
        if (e.name.len <= 3) continue;
        _ = std.fmt.parseInt(u32, e.name[3..], 10) catch continue;

        const cache_path = std.fmt.allocPrint(alloc, "{s}/cache", .{e.name}) catch continue;
        defer alloc.free(cache_path);
        var cache_dir = cpu_dir.openDir(io, cache_path, .{ .iterate = true }) catch continue;
        defer cache_dir.close(io);

        var it2 = cache_dir.iterate();
        while (it2.next(io) catch null) |idx| {
            if (idx.kind != .directory) continue;
            if (!std.mem.startsWith(u8, idx.name, "index")) continue;

            const idx_dir = cache_dir.openDir(io, idx.name, .{}) catch continue;
            defer idx_dir.close(io);

            const level_s = readFileTrimAlloc(alloc, idx_dir, "level") catch continue;
            defer alloc.free(level_s);
            const type_s = readFileTrimAlloc(alloc, idx_dir, "type") catch continue;
            defer alloc.free(type_s);
            const size_s = readFileTrimAlloc(alloc, idx_dir, "size") catch continue;
            defer alloc.free(size_s);

            const level = std.fmt.parseInt(u32, level_s, 10) catch continue;
            const size_bytes = parseSizeToBytes(size_s);

            const is_dataish = std.mem.eql(u8, type_s, "Data") or std.mem.eql(u8, type_s, "Unified");

            switch (level) {
                1 => if (is_dataish) {
                    out.l1d_bytes = @max(out.l1d_bytes, size_bytes);
                },
                2 => if (is_dataish) {
                    out.l2_bytes = @max(out.l2_bytes, size_bytes);
                },
                3 => if (is_dataish) {
                    out.l3_bytes = @max(out.l3_bytes, size_bytes);
                },
                else => {},
            }
        }
    }
    return out;
}

fn detectCaches() cpuid_root.Caches {
    if (builtin.os.tag == .linux or builtin.os.tag == .android) {
        var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        defer _ = gpa.deinit();
        return detectCachesArmLinux(gpa.allocator());
    }
    if (builtin.os.tag.isDarwin()) {
        return .{
            .l1d_bytes = @intCast(sysctlU64("hw.l1dcachesize")),
            .l2_bytes = @intCast(sysctlU64("hw.l2cachesize")),
            .l3_bytes = @intCast(sysctlU64("hw.l3cachesize")),
        };
    }
    return .{};
}
