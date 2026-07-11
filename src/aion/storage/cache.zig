// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

/// Cache-layer errors used by runtime policy and lease bookkeeping.
pub const CacheError = error{
    CacheLimitExceeded,
    OutOfMemoryRam,
    InvalidArgument,
};

pub const CachePolicy = enum(u8) {
    none = 0,
    growable = 1,
    ring = 2,
};

pub const GrowablePolicy = struct {
    /// Initial logical token capacity tracked by the policy state.
    ///
    /// `0` means "start from physical tensor capacity".
    initial_capacity_tokens: usize = 0,

    /// Geometric growth factor as a rational number.
    growth_numerator: usize = 3,
    growth_denominator: usize = 2,

    /// Hard ceiling on logical token capacity — the growth bound / "max sequence
    /// length" knob. `0` means unbounded (grow as far as needed). A step past it
    /// fails (`CacheLimitExceeded`) rather than growing further, and growth never
    /// overshoots it. Lets callers allocate a small initial cache and cap how far
    /// the runtime may grow it.
    max_capacity_tokens: usize = 0,
};

pub const RingPolicy = struct {
    /// Logical window size in tokens.
    ///
    /// `0` means "use full physical T capacity".
    window_tokens: usize = 0,
};

pub const SequenceCachePolicy = union(CachePolicy) {
    none: void,
    growable: GrowablePolicy,
    ring: RingPolicy,
};

pub const CacheConfig = struct {
    /// Soft upper bound for RAM used by cached tiles.
    ///
    /// v1 notes:
    /// - we do not yet implement out-of-core eviction,
    /// - this is validated to be non-zero and is reserved for future cache
    ///   accounting policy.
    ram_budget_bytes: usize,

    /// Optional hard cap on simultaneously outstanding tile leases.
    ///
    /// `0` means unlimited.
    max_live_leases: usize = 0,
};

pub const SequenceCachePolicyKind = enum(u8) {
    none = 0,
    growable = 1,
    ring = 2,
};

pub const SequenceCachePolicyInfo = struct {
    kind: SequenceCachePolicyKind = .none,
    ring_window_tokens: usize = 0,
};

pub const Cache = struct {
    allocator: std.mem.Allocator,
    cfg: CacheConfig,

    tensor_policies: std.ArrayListUnmanaged(TensorPolicyRecord) = .empty,
    live_leases: std.AutoHashMapUnmanaged(usize, LeaseRecord) = .empty,
    next_lease_token: usize = 1,

    const Self = @This();

    const GrowableState = struct {
        logical_capacity_tokens: usize = 0,
        max_seen_end_tokens: usize = 0,
    };

    const RingState = struct {
        window_tokens: usize = 0,
        write_head_tokens: usize = 0,
    };

    const PolicyState = union(CachePolicy) {
        none: void,
        growable: GrowableState,
        ring: RingState,
    };

    const TensorPolicyRecord = struct {
        policy: SequenceCachePolicy = .{ .none = {} },
        state: PolicyState = .{ .none = {} },
    };

    const LeaseRecord = struct {
        tensor_id: u32,
        tile_index: usize,
        is_mut: bool,
    };

    fn makePolicyRecord(policy: SequenceCachePolicy) CacheError!TensorPolicyRecord {
        return switch (policy) {
            .none => .{ .policy = .{ .none = {} }, .state = .{ .none = {} } },
            .growable => |g| blk: {
                if (g.growth_numerator == 0 or g.growth_denominator == 0) {
                    return CacheError.InvalidArgument;
                }
                break :blk .{
                    .policy = .{ .growable = g },
                    .state = .{ .growable = .{ .logical_capacity_tokens = g.initial_capacity_tokens } },
                };
            },
            .ring => |r| .{
                .policy = .{ .ring = r },
                .state = .{ .ring = .{ .window_tokens = r.window_tokens, .write_head_tokens = 0 } },
            },
        };
    }

    fn ensureTensorSlots(self: *Self, tensor_id: u32) CacheError!void {
        const need_len: usize = @as(usize, @intCast(tensor_id)) + 1;
        while (self.tensor_policies.items.len < need_len) {
            self.tensor_policies.append(self.allocator, .{}) catch return CacheError.OutOfMemoryRam;
        }
    }

    fn policyRecordPtr(self: *Self, tensor_id: u32) ?*TensorPolicyRecord {
        const idx: usize = @as(usize, @intCast(tensor_id));
        if (idx >= self.tensor_policies.items.len) return null;
        return &self.tensor_policies.items[idx];
    }

    fn policyRecordConst(self: *const Self, tensor_id: u32) ?*const TensorPolicyRecord {
        const idx: usize = @as(usize, @intCast(tensor_id));
        if (idx >= self.tensor_policies.items.len) return null;
        return &self.tensor_policies.items[idx];
    }

    fn growCapacity(current: usize, num: usize, den: usize) CacheError!usize {
        if (den == 0 or num == 0) return CacheError.InvalidArgument;
        const prod: usize = std.math.mul(usize, current, num) catch return CacheError.CacheLimitExceeded;
        const rounded: usize = std.math.add(usize, prod, den - 1) catch return CacheError.CacheLimitExceeded;
        const next: usize = rounded / den;
        if (next <= current) {
            return std.math.add(usize, current, 1) catch return CacheError.CacheLimitExceeded;
        }
        return next;
    }

    pub fn init(allocator: std.mem.Allocator, cfg: CacheConfig) CacheError!Cache {
        if (cfg.ram_budget_bytes == 0) return CacheError.InvalidArgument;
        return .{ .allocator = allocator, .cfg = cfg };
    }

    pub fn registerTensorPolicy(self: *Self, tensor_id: u32, policy: SequenceCachePolicy) CacheError!void {
        try self.ensureTensorSlots(tensor_id);
        const rec: TensorPolicyRecord = try makePolicyRecord(policy);
        self.tensor_policies.items[@as(usize, @intCast(tensor_id))] = rec;
    }

    pub fn tensorPolicy(self: *const Self, tensor_id: u32) SequenceCachePolicy {
        const rec_opt: ?*const TensorPolicyRecord = self.policyRecordConst(tensor_id);
        if (rec_opt) |rec| return rec.policy;
        return .{ .none = {} };
    }

    pub fn tensorPolicyInfo(self: *const Self, tensor_id: u32) SequenceCachePolicyInfo {
        return switch (self.tensorPolicy(tensor_id)) {
            .none => .{ .kind = .none, .ring_window_tokens = 0 },
            .growable => .{ .kind = .growable, .ring_window_tokens = 0 },
            .ring => |r| .{ .kind = .ring, .ring_window_tokens = r.window_tokens },
        };
    }

    /// Map a logical token position to physical cache time index.
    ///
    /// This is deterministic and policy-specific:
    /// - `.none` / `.growable`: identity mapping with strict bounds checks.
    /// - `.ring`: modulo mapping using window or full physical capacity.
    pub fn mapLogicalTime(self: *Self, tensor_id: u32, logical_t: usize, physical_capacity_tokens: usize) CacheError!usize {
        if (physical_capacity_tokens == 0) return CacheError.InvalidArgument;

        const rec_opt: ?*TensorPolicyRecord = self.policyRecordPtr(tensor_id);
        if (rec_opt == null) {
            if (logical_t >= physical_capacity_tokens) return CacheError.CacheLimitExceeded;
            return logical_t;
        }

        const rec: *TensorPolicyRecord = rec_opt.?;
        switch (rec.policy) {
            .none => {
                if (logical_t >= physical_capacity_tokens) return CacheError.CacheLimitExceeded;
                return logical_t;
            },
            .growable => |g| {
                if (rec.state != .growable) return CacheError.InvalidArgument;
                var st: *GrowableState = &rec.state.growable;

                const need: usize = std.math.add(usize, logical_t, 1) catch return CacheError.CacheLimitExceeded;
                if (g.max_capacity_tokens != 0 and need > g.max_capacity_tokens) return CacheError.CacheLimitExceeded;
                if (st.logical_capacity_tokens == 0) {
                    st.logical_capacity_tokens = if (g.initial_capacity_tokens == 0) physical_capacity_tokens else g.initial_capacity_tokens;
                }
                while (st.logical_capacity_tokens < need) {
                    st.logical_capacity_tokens = try growCapacity(st.logical_capacity_tokens, g.growth_numerator, g.growth_denominator);
                }
                if (g.max_capacity_tokens != 0 and st.logical_capacity_tokens > g.max_capacity_tokens) {
                    st.logical_capacity_tokens = g.max_capacity_tokens;
                }
                if (need > st.max_seen_end_tokens) st.max_seen_end_tokens = need;
                return logical_t;
            },
            .ring => {
                if (rec.state != .ring) return CacheError.InvalidArgument;
                var st: *RingState = &rec.state.ring;
                const configured_window: usize = if (st.window_tokens == 0) physical_capacity_tokens else @min(st.window_tokens, physical_capacity_tokens);
                if (configured_window == 0) return CacheError.InvalidArgument;
                st.write_head_tokens = logical_t;
                return logical_t % configured_window;
            },
        }
    }

    pub fn acquireLease(self: *Self, tensor_id: u32, tile_index: usize, is_mut: bool) CacheError!usize {
        if (self.cfg.max_live_leases != 0 and self.live_leases.count() >= self.cfg.max_live_leases) {
            return CacheError.CacheLimitExceeded;
        }

        if (self.next_lease_token == 0) return CacheError.CacheLimitExceeded;
        const token: usize = self.next_lease_token;
        self.next_lease_token = std.math.add(usize, self.next_lease_token, 1) catch return CacheError.CacheLimitExceeded;

        self.live_leases.put(self.allocator, token, .{
            .tensor_id = tensor_id,
            .tile_index = tile_index,
            .is_mut = is_mut,
        }) catch return CacheError.OutOfMemoryRam;

        return token;
    }

    pub fn releaseLease(self: *Self, token: usize) void {
        if (token == 0) return;
        _ = self.live_leases.remove(token);
    }

    pub fn liveLeaseCount(self: *const Self) usize {
        return self.live_leases.count();
    }

    pub fn hasLiveLease(self: *const Self, token: usize) bool {
        return self.live_leases.contains(token);
    }

    pub fn deinit(self: *Cache) void {
        self.live_leases.deinit(self.allocator);
        self.tensor_policies.deinit(self.allocator);
        self.* = undefined;
    }
};
