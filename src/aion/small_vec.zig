// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

pub fn SmallVec(comptime T: type, comptime inline_cap: usize) type {
    return struct {
        len: usize = 0,
        inline_buf: [inline_cap]T = undefined,
        heap_buf: []T = &[_]T{},
        using_heap: bool = false,
        allocator: ?std.mem.Allocator = null,

        const Self = @This();

        pub fn initEmpty() Self {
            return .{};
        }

        pub fn initWithLen(allocator: std.mem.Allocator, len: usize) !Self {
            var self: Self = .{};
            if (len <= inline_cap) {
                self.len = len;
                return self;
            }
            const buf: []T = try allocator.alloc(T, len);
            self.len = len;
            self.heap_buf = buf;
            self.using_heap = true;
            self.allocator = allocator;
            return self;
        }

        pub fn initFromSlice(allocator: std.mem.Allocator, items: []const T) !Self {
            var self: Self = try Self.initWithLen(allocator, items.len);
            if (items.len == 0) return self;

            if (self.using_heap) {
                @memcpy(self.heap_buf[0..items.len], items);
            } else {
                @memcpy(self.inline_buf[0..items.len], items);
            }
            return self;
        }

        pub fn deinit(self: *Self) void {
            if (self.using_heap) {
                if (self.allocator) |a| a.free(self.heap_buf);
            }
            self.* = .{};
        }

        pub fn slice(self: *Self) []T {
            if (self.using_heap) return self.heap_buf[0..self.len];
            return self.inline_buf[0..self.len];
        }

        pub fn constSlice(self: *const Self) []const T {
            if (self.using_heap) return self.heap_buf[0..self.len];
            return self.inline_buf[0..self.len];
        }
    };
}
