const std = @import("std");

/// I/O layer for SSD/NVMe-backed storage.
///
/// v0: not implemented yet; this module defines deterministic error surfaces and
/// will later host the prefetch worker and explicit pread implementation.
pub const IoError = error{
    IoFailure,
    IoStall,
    CorruptData,
};

pub const IoConfig = struct {
    /// Maximum in-flight reads.
    max_in_flight: usize = 8,
};

pub const IoScheduler = struct {
    allocator: std.mem.Allocator,
    cfg: IoConfig,

    pub fn init(allocator: std.mem.Allocator, cfg: IoConfig) IoScheduler {
        return .{ .allocator = allocator, .cfg = cfg };
    }

    pub fn deinit(self: *IoScheduler) void {
        _ = self;
    }
};
