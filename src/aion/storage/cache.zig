const std = @import("std");

/// Placeholder for the tiered RAM cache / LRU / pinning layer.
///
/// v0: `TiledTensor` is RAM-resident and effectively always pinned.
///
/// This file exists so higher layers can start depending on stable error sets and
/// interfaces without committing to a full out-of-core implementation yet.
pub const CacheError = error{
    CacheLimitExceeded,
    OutOfMemoryRam,
};

pub const CacheConfig = struct {
    /// Soft upper bound for RAM used by cached tiles.
    /// Enforced strictly once we implement eviction.
    ram_budget_bytes: usize,
};

pub const Cache = struct {
    allocator: std.mem.Allocator,
    cfg: CacheConfig,

    pub fn init(allocator: std.mem.Allocator, cfg: CacheConfig) Cache {
        return .{ .allocator = allocator, .cfg = cfg };
    }

    pub fn deinit(self: *Cache) void {
        _ = self;
    }
};
