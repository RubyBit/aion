// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const env_util = @import("../env.zig");

const backend_mod = @import("../backend/backend.zig");
const graph_mod = @import("../graph/graph.zig");
const program_mod = @import("../graph/program.zig");
const package_file = @import("../storage/aion_file.zig");

const api_errors = @import("errors.zig");

const lease = @import("../graph/program/lease.zig");
const template_mod = @import("../graph/template.zig");
const api_package_export = @import("package_export.zig");

const types_mod = @import("loaded_model/types.zig");
const signatures = @import("loaded_model/signatures.zig");
const retarget = @import("loaded_model/retarget.zig");
const initializers = @import("loaded_model/initializers.zig");
const params_mod = @import("loaded_model/params.zig");

fn traceEnabled() bool {
    return env_util.flagEnabled("AION_TRACE");
}

pub const Tensor = types_mod.Tensor;
pub const StorageManager = types_mod.StorageManager;
pub const TensorId = types_mod.TensorId;
pub const DeviceRef = types_mod.DeviceRef;
pub const SequenceCachePolicy = types_mod.SequenceCachePolicy;
pub const DType = types_mod.DType;
pub const TilePolicy = types_mod.TilePolicy;
pub const Target = types_mod.Target;
pub const Params = params_mod.Params;
pub const no_param = params_mod.invalid;
pub const Package = types_mod.Package;
pub const SignatureInfo = types_mod.SignatureInfo;
pub const IoAliasInfo = types_mod.IoAliasInfo;
pub const LoadModelOptions = types_mod.LoadModelOptions;
pub const CacheOptions = types_mod.CacheOptions;
pub const CacheGrowth = types_mod.CacheGrowth;

/// Owns the records borrowed by a model template: either a parsed package or a compiled
/// snapshot. Downstream specialization uses the template uniformly.
pub const TemplateOwner = union(enum) {
    package: Package,
    parts: api_package_export.Parts,

    pub fn template(self: *const TemplateOwner) template_mod.Template {
        return switch (self.*) {
            .package => |*p| .{
                .values = p.values,
                .nodes = p.nodes,
                .regions = p.regions,
                .inputs = &.{}, // filled by `init` from the signatures
                .outputs = &.{},
                .dim_exprs = p.dim_exprs,
                .dim_symbol_count = p.dim_symbols.len,
            },
            .parts => |*parts| parts.view(),
        };
    }

    fn values(self: *const TemplateOwner) []const package_file.ValueRecord {
        return switch (self.*) {
            .package => |*p| p.values,
            .parts => |*parts| parts.values,
        };
    }

    fn namedInputs(self: *const TemplateOwner) []const package_file.NamedValue {
        return switch (self.*) {
            .package => |*p| p.inputs,
            .parts => |*parts| parts.inputs,
        };
    }

    fn namedOutputs(self: *const TemplateOwner) []const package_file.NamedValue {
        return switch (self.*) {
            .package => |*p| p.outputs,
            .parts => |*parts| parts.outputs,
        };
    }

    fn debugNames(self: *const TemplateOwner) []const package_file.DebugName {
        return switch (self.*) {
            .package => |*p| p.debug_names,
            .parts => |*parts| parts.debug_names,
        };
    }

    fn ioAliases(self: *const TemplateOwner) []const package_file.IoAlias {
        return switch (self.*) {
            .package => |*p| p.io_aliases,
            .parts => |*parts| parts.io_aliases,
        };
    }

    fn inputRoles(self: *const TemplateOwner) []const package_file.InputRole {
        return switch (self.*) {
            .package => |*p| p.input_roles,
            .parts => |*parts| parts.input_roles,
        };
    }

    fn deinit(self: *TemplateOwner, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .package => |*p| p.deinit(),
            .parts => |*parts| parts.deinit(allocator),
        }
    }
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    backend: backend_mod.Backend,
    store: *StorageManager,
    /// Execution session bound to `store`. Owns the backend's per-store state
    /// (device residency on GPU) for this model's lifetime, so weights stay
    /// device-resident across `run` calls. Released in `deinit`.
    session: backend_mod.Session,
    /// Compilation and execution device, tiling, and optimization policy.
    /// Discrete GPUs may also keep recurrent-state slots device-exclusive.
    target: Target,
    /// The model's graph before its free dims are known, and whatever owns its records.
    template: template_mod.Template,
    template_owner: TemplateOwner,
    /// This model's weights, keyed by graph value. One representation for both sources
    /// (see `loaded_model/params.zig`).
    params: Params,
    input_signatures: []SignatureInfo,
    output_signatures: []SignatureInfo,
    output_aliases: []IoAliasInfo,
    input_alias_output_indices: []u32,
    output_alias_input_indices: []u32,
    /// Lazy host mirror per device-exclusive output; `invalid_tensor_id` until read.
    /// `resolveOutputTensor` gathers into this mirror for host access.
    output_host_mirror_tids: []TensorId,
    bound_inputs: []?Tensor,
    aliased_input_bind_versions: []u64,
    /// Persistent store tensors auto-allocated for inputs the caller never binds
    /// (zero-initialized; aliased ones carry state across runs). One slot per input;
    /// `invalid_tensor_id` until first auto-init. See `ensureAutoInputs`.
    auto_input_tids: []TensorId,
    auto_init_inputs: bool,
    /// Shared backing per io-aliased state input, allowing recurrent state to carry
    /// across cache entries with different shapes. Invalid for non-aliased/unbuilt inputs.
    aliased_state_tids: []TensorId,
    /// Optional per-input sequence-cache policy (grow-on-demand / ring), set via
    /// `setStateInputPolicy` and applied to the recurrent-state slot when it is
    /// created. `null` = fixed capacity (the slot never grows). One slot per input.
    state_input_policies: []?SequenceCachePolicy,
    /// Latest bind version seeded into each shared state slot.
    /// Model-level tracking prevents a new cache entry from overwriting carried state.
    aliased_state_synced_versions: []u64,
    /// Per-input package-declared role (null = none). Records copied at init.
    input_roles: []?package_file.InputRole,
    /// Cached input indices of the singleton control roles (null when absent).
    role_tokens_index: ?usize,
    role_positions_index: ?usize,
    role_write_index_index: ?usize,
    role_visible_end_index: ?usize,
    /// True when position auto-management is active (roles present + opt-in).
    auto_positions_enabled: bool,
    /// Tokens consumed so far by auto position tracking. `resetState` resets to 0.
    position_tokens: u64 = 0,
    /// Advance applied to `position_tokens` when the current `run()` succeeds.
    pending_position_advance: u64 = 0,
    /// Persistent i32 tensors backing auto-fed control inputs (per input;
    /// `invalid_tensor_id` until first use).
    role_input_tids: []TensorId,
    /// Whether `bound_inputs[i]` is the model's own auto position tensor (vs a
    /// caller bind, which permanently reclaims the input).
    role_auto_bound: []bool,
    /// Per-dim-symbol default used only to size unbound inputs in
    /// `ensureAutoInputs` (e.g. a free cache-capacity symbol from
    /// `LoadModelOptions.cache`). null = no default.
    symbol_defaults: []?u64,
    run_symbol_bindings: []?u64,
    run_symbol_values: []u64,
    run_input_shapes: []usize,
    run_direct_input_ids: []TensorId,
    cache_entries: std.ArrayList(CacheEntry) = .empty,
    cache_workspace_budget_bytes: usize,
    cache_workspace_bytes: usize = 0,
    cache_clock: u64 = 0,
    cache_build_count: u64 = 0,
    cache_eviction_count: u64 = 0,
    last_run_cache_index: ?usize = null,
    package_hash: u64,
    trace_runs: bool = false,

    const Self = @This();

    fn debugDumpBoundInputs(self: *const Self, trace: bool) void {
        if (!trace) return;
        std.debug.print("[aion][run] bound inputs (name -> dtype/shape/tile):\n", .{});
        for (self.input_signatures, 0..) |sig, idx| {
            const t_opt: ?Tensor = self.bound_inputs[idx];
            if (t_opt == null) {
                std.debug.print("  - {s}: <unbound>\n", .{sig.name});
                continue;
            }
            const t: Tensor = t_opt.?;
            const meta = self.store.getConst(t.id) catch {
                std.debug.print("  - {s}: <invalid tensor id {d}>\n", .{ sig.name, t.id });
                continue;
            };
            std.debug.print(
                "  - {s}: id={d} dtype={s} rank={d} shape={any} tile_shape={any} tile_counts={any} quant_axis={d} aliased_input={any}\n",
                .{ sig.name, t.id, @tagName(meta.dtype), meta.rank, meta.shape, meta.tile_shape, meta.tile_counts, meta.quant_axis, (self.inputAliasOutputIndex(idx) != null) },
            );
        }
    }

    pub fn deinit(self: *Self) void {
        self.session.retireResources();
        for (self.cache_entries.items) |*entry| self.destroyCacheEntry(entry);
        self.cache_entries.deinit(self.allocator);
        self.session.deinit();
        self.params.deinit(self.allocator);
        self.allocator.free(self.input_signatures);
        self.allocator.free(self.output_signatures);
        self.allocator.free(self.output_aliases);
        self.allocator.free(self.input_alias_output_indices);
        self.allocator.free(self.output_alias_input_indices);
        self.allocator.free(self.output_host_mirror_tids);
        self.allocator.free(self.bound_inputs);
        self.allocator.free(self.aliased_input_bind_versions);
        self.allocator.free(self.auto_input_tids);
        self.allocator.free(self.aliased_state_tids);
        self.allocator.free(self.aliased_state_synced_versions);
        self.allocator.free(self.state_input_policies);
        self.allocator.free(self.input_roles);
        self.allocator.free(self.role_input_tids);
        self.allocator.free(self.role_auto_bound);
        self.allocator.free(self.symbol_defaults);
        self.allocator.free(self.run_symbol_bindings);
        self.allocator.free(self.run_symbol_values);
        self.allocator.free(self.run_input_shapes);
        self.allocator.free(self.run_direct_input_ids);
        self.allocator.free(@constCast(self.template.inputs));
        self.allocator.free(@constCast(self.template.outputs));
        self.template_owner.deinit(self.allocator);
        self.* = undefined;
    }

    pub fn inputNames(self: *const Self) []const SignatureInfo {
        return self.input_signatures;
    }

    pub fn outputNames(self: *const Self) []const SignatureInfo {
        return self.output_signatures;
    }

    pub fn outputAliases(self: *const Self) []const IoAliasInfo {
        return self.output_aliases;
    }

    /// Return author-provided value names from either the loaded package or compiled
    /// template owner.
    pub fn debugNames(self: *const Self) []const package_file.DebugName {
        return self.template_owner.debugNames();
    }

    /// Return the value index for a debug name, or null if not present.
    pub fn findValueByDebugName(self: *const Self, name: []const u8) ?u32 {
        for (self.template_owner.debugNames()) |entry| {
            if (std.mem.eql(u8, entry.name, name)) return entry.value;
        }
        return null;
    }

    /// Return the store tensor backing parameter `value_index` for either model source.
    fn paramTid(self: *Self, value_index: u32) api_errors.ApiError!TensorId {
        return self.params.get(value_index) orelse api_errors.ApiError.InvalidArgument;
    }

    /// Return an owned, mutable initializer tensor handle.
    /// Fused weights require `overwriteInitializerByValue` because they lack standalone
    /// storage.
    pub fn initializerTensorByValue(self: *Self, value_index: u32) api_errors.ApiError!Tensor {
        const tid: TensorId = try self.paramTid(value_index);
        if (self.store.derivedLocate(tid) != null) return api_errors.ApiError.InvalidArgument;
        const meta = try self.store.getConst(tid);
        return .{ .store = self.store, .id = tid, .dtype = meta.dtype, .shape = meta.shape };
    }

    /// Like `initializerTensorByValue`, but looks up the initializer via a
    /// persisted debug name.
    pub fn initializerTensorByDebugName(self: *Self, debug_name: []const u8) api_errors.ApiError!Tensor {
        const value_index: u32 = self.findValueByDebugName(debug_name) orelse return api_errors.ApiError.InvalidArgument;
        return self.initializerTensorByValue(value_index);
    }

    /// Copy compatible bytes over an initializer without changing tensor IDs or
    /// retargeting compiled programs. Different tilings use pack/unpack.
    pub fn overwriteInitializerByValue(self: *Self, value_index: u32, src: Tensor) api_errors.ApiError!void {
        if (src.store != self.store) return api_errors.ApiError.InvalidArgument;
        const tid: TensorId = try self.paramTid(value_index);

        // If this weight was fused into a combined tensor, the fused tensor is the
        // canonical store — write through to its sub-region (the original buffer may
        // have been reclaimed). Takes effect next run with no recompile.
        if (self.store.derivedLocate(tid) != null) {
            return self.store.writeDerivedSource(tid, src.id) catch api_errors.ApiError.InvalidArgument;
        }

        if (tid == src.id) return;
        try self.copyTensorInto(tid, src);
    }

    pub fn overwriteInitializerByDebugName(self: *Self, debug_name: []const u8, src: Tensor) api_errors.ApiError!void {
        const value_index: u32 = self.findValueByDebugName(debug_name) orelse return api_errors.ApiError.InvalidArgument;
        return self.overwriteInitializerByValue(value_index, src);
    }

    /// Read an initializer into a caller-owned, dtype/shape-compatible `dst`.
    /// Fused weights are materialized from their combined tensor.
    pub fn readInitializerByValue(self: *Self, value_index: u32, dst: Tensor) api_errors.ApiError!void {
        if (dst.store != self.store) return api_errors.ApiError.InvalidArgument;
        const tid: TensorId = try self.paramTid(value_index);

        if (self.store.derivedLocate(tid) != null) {
            return self.store.readDerivedSource(tid, dst.id) catch api_errors.ApiError.InvalidArgument;
        }

        if (tid == dst.id) return;
        const meta = try self.store.getConst(tid);
        const src: Tensor = .{ .store = self.store, .id = tid, .dtype = meta.dtype, .shape = meta.shape };
        try dst.copyFrom(self.allocator, src);
    }

    pub fn readInitializerByDebugName(self: *Self, debug_name: []const u8, dst: Tensor) api_errors.ApiError!void {
        const value_index: u32 = self.findValueByDebugName(debug_name) orelse return api_errors.ApiError.InvalidArgument;
        return self.readInitializerByValue(value_index, dst);
    }

    /// Retarget an initializer in future entries and existing cached programs.
    /// `new_tensor` must use the same context and have a compatible dtype, shape, and
    /// tile layout.
    pub fn retargetInitializerByValue(self: *Self, value_index: u32, new_tensor: Tensor) api_errors.ApiError!void {
        if (new_tensor.store != self.store) return api_errors.ApiError.InvalidArgument;
        const old_tid: TensorId = try self.paramTid(value_index);
        const new_tid: TensorId = new_tensor.id;
        if (old_tid == new_tid) return;
        if (self.store.tensorIsWorkspace(new_tid) catch true) return api_errors.ApiError.InvalidArgument;

        // A fused weight isn't referenced by the program (the combined tensor is), so
        // retargeting its id would patch nothing. Write the new contents through to
        // the fused weight's sub-region instead; the alias mapping stays put.
        if (self.store.derivedLocate(old_tid) != null) {
            return self.store.writeDerivedSource(old_tid, new_tid) catch api_errors.ApiError.InvalidArgument;
        }

        const ok = try signatures.tensorsHaveCompatibleLayout(self.store, old_tid, new_tid);
        if (!ok) return api_errors.ApiError.InvalidArgument;

        self.params.set(value_index, new_tid);
        for (self.cache_entries.items) |*entry| {
            retarget.retargetProgramTensorIds(&entry.program, old_tid, new_tid);
            // The program now names `new_tid` where it named `old_tid`, and reclaim
            // reads those counts, so they move with the ids.
            self.store.releaseTensor(old_tid);
            self.store.retainTensor(new_tid);
        }
        // The compiled programs now name a tensor the caller supplied, and their
        // placement table still says the weight is device-placed. Honor that
        // table rather than leaving a host tensor wired into a device program.
        self.store.placeTensor(new_tid, self.target.device) catch |err| switch (err) {
            error.OutOfMemory => return api_errors.ApiError.OutOfMemory,
            else => return api_errors.ApiError.InvalidArgument,
        };
    }

    pub fn retargetInitializerByDebugName(self: *Self, debug_name: []const u8, new_tensor: Tensor) api_errors.ApiError!void {
        const value_index: u32 = self.findValueByDebugName(debug_name) orelse return api_errors.ApiError.InvalidArgument;
        return self.retargetInitializerByValue(value_index, new_tensor);
    }

    pub fn cacheEntryCount(self: *const Self) usize {
        return self.cache_entries.items.len;
    }

    pub const PlanCacheStats = struct {
        entries: usize,
        workspace_bytes: usize,
        budget_bytes: usize,
        builds: u64,
        evictions: u64,
    };

    pub fn planCacheStats(self: *const Self) PlanCacheStats {
        return .{
            .entries = self.cache_entries.items.len,
            .workspace_bytes = self.cache_workspace_bytes,
            .budget_bytes = self.cache_workspace_budget_bytes,
            .builds = self.cache_build_count,
            .evictions = self.cache_eviction_count,
        };
    }

    /// Whether an io-aliased input's state slot is device-exclusive.
    /// False for non-aliased, unallocated, or CPU-backed inputs.
    pub fn stateInputOnDevice(self: *const Self, name: []const u8) bool {
        const index = self.findInputIndex(name) orelse return false;
        const tid = self.aliased_state_tids[index];
        if (tid == types_mod.invalid_tensor_id) return false;
        const ref = self.store.tensorDevice(tid) catch return false;
        return ref.kind != .cpu;
    }

    pub fn bindInput(self: *Self, name: []const u8, tensor: Tensor) api_errors.ApiError!void {
        const index = self.findInputIndex(name) orelse return api_errors.ApiError.InvalidArgument;
        const sig = self.input_signatures[index];
        if (tensor.store != self.store) return api_errors.ApiError.InvalidArgument;
        if (tensor.dtype != sig.dtype) return api_errors.ApiError.InvalidArgument;
        if (tensor.shape.len != sig.rank) return api_errors.ApiError.InvalidArgument;
        self.bound_inputs[index] = tensor;
        // A manual bind on a role-driven control input permanently reclaims it from
        // position auto-management (the escape hatch for custom schedules).
        self.role_auto_bound[index] = false;
        if (self.inputAliasOutputIndex(index) != null) {
            self.aliased_input_bind_versions[index] +%= 1;
            if (self.aliased_input_bind_versions[index] == 0) self.aliased_input_bind_versions[index] = 1;
        }
    }

    /// Zero all shared io-aliased recurrent-state slots; a no-op before first run.
    /// Declare grow-on-demand or ring policy for a named io-aliased state input.
    /// Prefer calling before first run so the policy shapes the initial slot.
    pub fn setStateInputPolicy(self: *Self, name: []const u8, policy: SequenceCachePolicy) api_errors.ApiError!void {
        const index = self.findInputIndex(name) orelse return api_errors.ApiError.InvalidArgument;
        // Growth/ring bookkeeping lives in the store's cache; ensure one exists.
        // The RAM budget is a reserved (not-yet-enforced) soft bound, so a sentinel
        // max is fine — only tensors that carry a policy ever lease against it.
        if (!self.store.hasCache()) {
            self.store.configureCache(.{ .ram_budget_bytes = std.math.maxInt(usize) }) catch return api_errors.ApiError.OutOfMemory;
        }
        self.state_input_policies[index] = policy;
        // If the slot already exists (policy set between runs), register it now too.
        const tid = self.aliased_state_tids[index];
        if (tid != types_mod.invalid_tensor_id) {
            self.store.registerSequenceCachePolicy(tid, policy) catch return api_errors.ApiError.InvalidArgument;
        }
    }

    pub fn resetState(self: *Self) api_errors.ExecuteError!void {
        for (self.aliased_state_tids) |tid| {
            if (tid == types_mod.invalid_tensor_id) continue;
            // Residency-aware: zeros the device buffer for a device-exclusive slot,
            // the host bytes otherwise (see `StorageManager.zeroTensorData`).
            self.store.zeroTensorData(tid) catch return error.InvalidArgument;
        }
        self.position_tokens = 0;
        self.pending_position_advance = 0;
    }

    /// Tokens consumed so far by position auto-management (0 when disabled or
    /// before the first run). Advances by the tokens-role sequence length after
    /// each successful `run()`.
    pub fn currentPosition(self: *const Self) u64 {
        return self.position_tokens;
    }

    /// Overwrite the auto-tracked position (escape hatch: session restore,
    /// speculative-decode rollback, or re-syncing after manual index binds).
    pub fn setPosition(self: *Self, tokens: u64) void {
        self.position_tokens = tokens;
        self.pending_position_advance = 0;
    }

    pub fn run(self: *Self) api_errors.ExecuteError!void {
        const trace: bool = self.trace_runs;
        if (trace) std.debug.print("[aion][run] begin\n", .{});

        // Feed role-declared control inputs (cache write index / visible end /
        // positions) from the auto-tracked sequence position.
        try self.ensureAutoPositions(trace);

        // Fill any input the caller never bound with a persistent, zero-initialized
        // slot (recurrent/aliased ones carry their contents across runs). After this,
        // every input is bound and the rest of `run()` proceeds unchanged.
        try self.ensureAutoInputs(trace);

        if (trace) self.debugDumpBoundInputs(true);

        const resolved = self.resolveBindings() catch |e| {
            if (trace) {
                std.debug.print("[aion][run] resolveBindings failed: {s}\n", .{@errorName(e)});
            }
            self.debugDumpBoundInputs(trace);
            return e;
        };

        const static_idx_opt: ?usize = self.findStaticCacheEntry(resolved.direct_input_ids) catch |e| {
            if (trace) {
                std.debug.print("[aion][run] findStaticCacheEntry failed: {s}\n", .{@errorName(e)});
            }
            self.debugDumpBoundInputs(trace);
            return e;
        };

        const cache_index: usize = if (static_idx_opt) |idx|
            idx
        else
            self.ensureCacheEntry(resolved.symbol_values, resolved.input_shapes, resolved.direct_input_ids) catch |e| {
                if (trace) {
                    std.debug.print("[aion][run] ensureCacheEntry failed: {s}\n", .{@errorName(e)});
                }
                self.debugDumpBoundInputs(trace);
                return e;
            };
        var entry: *CacheEntry = &self.cache_entries.items[cache_index];

        if (trace) {
            std.debug.print(
                "[aion][run] using cache_index={d} steps={d} outputs={d}\n",
                .{ cache_index, entry.program.steps.len, entry.program.outputs.len },
            );
        }

        var i: usize = 0;
        while (i < self.bound_inputs.len) : (i += 1) {
            const src_opt: ?Tensor = self.bound_inputs[i];
            if (src_opt == null) {
                if (trace) {
                    std.debug.print("[aion][run] missing bound input at index={d}\n", .{i});
                }
                return error.InvalidArgument;
            }
            const src: Tensor = src_opt.?;
            const dst_tid = entry.input_slots[i];
            const dst_meta = self.store.getConst(dst_tid) catch |e| {
                if (trace) {
                    std.debug.print(
                        "[aion][run] getConst failed for input slot {d} (name={s} tid={d}): {s}\n",
                        .{ i, self.input_signatures[i].name, dst_tid, @errorName(e) },
                    );
                }
                return e;
            };
            const dst = Tensor{ .store = self.store, .id = dst_tid, .dtype = dst_meta.dtype, .shape = dst_meta.shape };
            if (self.inputAliasOutputIndex(i) != null) {
                const bind_version = self.aliased_input_bind_versions[i];
                if (bind_version != 0 and self.aliased_state_synced_versions[i] != bind_version) {
                    if (src.id != dst.id) self.copyTensorInto(dst.id, src) catch |e| {
                        if (trace) {
                            std.debug.print(
                                "[aion][run] seed copy failed for aliased input {s} (src_id={d} dst_id={d}): {s}\n",
                                .{ self.input_signatures[i].name, src.id, dst.id, @errorName(e) },
                            );
                        }
                        return e;
                    };
                    self.aliased_state_synced_versions[i] = bind_version;
                }
                continue;
            }
            if (src.id != dst.id) self.copyTensorInto(dst.id, src) catch |e| {
                if (trace) {
                    std.debug.print(
                        "[aion][run] copyFrom failed for input {s} (src_id={d} dst_id={d}): {s}\n",
                        .{ self.input_signatures[i].name, src.id, dst.id, @errorName(e) },
                    );
                }
                return e;
            };
        }

        // After input seeding (which settles the append index) and before the
        // frame is recorded — the only window in which capacity can change
        // without synchronizing against device work. See `prepareGrowableCaches`.
        try self.prepareGrowableCaches(entry);

        if (trace) {
            std.debug.print("[aion][run] executing\n", .{});
        }

        self.session.execute(&entry.program) catch |e| {
            if (trace) {
                std.debug.print("[aion][run] session.execute failed: {s}\n", .{@errorName(e)});
            }
            return e;
        };

        if (trace) {
            std.debug.print("[aion][run] executed ok, syncing io_aliases={d}\n", .{self.output_aliases.len});
        }

        for (self.output_aliases) |alias| {
            const dst_tid = entry.input_slots[alias.input_index];
            const src_tid = entry.program.outputs[alias.output_index];
            if (dst_tid == src_tid) continue;

            // GPU model outputs are read back lazily. A non-in-place alias is an
            // internal consumer, so materialize a staged source before copying
            // it into the persistent state slot.
            const dst_meta = self.store.getConst(dst_tid) catch |e| {
                if (trace) {
                    std.debug.print(
                        "[aion][run] getConst failed for io_alias dst (input={s} tid={d}): {s}\n",
                        .{ alias.input_name, dst_tid, @errorName(e) },
                    );
                }
                return e;
            };
            const src_meta = self.store.getConst(src_tid) catch |e| {
                if (trace) {
                    std.debug.print(
                        "[aion][run] getConst failed for io_alias src (output={s} tid={d}): {s}\n",
                        .{ alias.output_name, src_tid, @errorName(e) },
                    );
                }
                return e;
            };
            const dst = Tensor{ .store = self.store, .id = dst_tid, .dtype = dst_meta.dtype, .shape = dst_meta.shape };
            const src = Tensor{ .store = self.store, .id = src_tid, .dtype = src_meta.dtype, .shape = src_meta.shape };
            // Copy non-in-place recurrent outputs back through the residency-aware
            // path; in-place aliases were skipped above and need no copy.
            self.copyTensorInto(dst.id, src) catch |e| {
                if (trace) {
                    std.debug.print(
                        "[aion][run] output alias state copy failed (input={s} output={s}): {s}\n",
                        .{ alias.input_name, alias.output_name, @errorName(e) },
                    );
                }
                return e;
            };
        }

        if (trace) {
            std.debug.print("[aion][run] done\n", .{});
        }

        if (self.auto_positions_enabled) {
            self.position_tokens += self.pending_position_advance;
            self.pending_position_advance = 0;
        }

        self.last_run_cache_index = cache_index;
    }

    /// Grow device sequence caches from placement requests using host-known write
    /// positions. In-graph positions are excluded at load time to avoid synchronization.
    fn prepareGrowableCaches(self: *Self, entry: *const CacheEntry) api_errors.ExecuteError!void {
        if (self.target.device.kind == .cpu) return;

        for (entry.program.growth_requests) |request| {
            if (self.store.sequenceCachePolicyInfo(request.cache).kind != .growable) continue;

            const cache_meta = self.store.getConst(request.cache) catch return error.InvalidArgument;
            const new_meta = self.store.getConst(request.new_kv) catch return error.InvalidArgument;
            if (cache_meta.rank != 4 or new_meta.rank != 4 or cache_meta.shape[0] == 0) return error.InvalidArgument;
            const new_len = new_meta.shape[1];
            if (new_len == 0) continue;

            var input_index: ?usize = null;
            for (entry.input_slots, 0..) |tid, i| {
                if (tid == request.end_index) {
                    input_index = i;
                    break;
                }
            }
            // A growable cache's position is an external input by construction
            // (see `appendPositionsAreExternal`), so a miss here is a bug.
            const idx = input_index orelse return error.InvalidArgument;
            const required: usize = if (self.role_write_index_index == idx and self.role_auto_bound[idx])
                std.math.add(usize, @intCast(self.position_tokens), new_len) catch return error.InvalidArgument
            else
                try self.appendEndFromBoundIndex(idx, cache_meta.shape[0], new_len);

            if (required <= cache_meta.shape[1]) continue;
            _ = self.store.tensorStore().mapSequenceStep(request.cache, required - 1, cache_meta.shape[1]) catch return error.InvalidArgument;
        }
    }

    /// Furthest append end implied by the caller-bound start-index input, read
    /// off the host. Stack-buffered for realistic batch counts so a growable
    /// cache costs no allocation per token.
    fn appendEndFromBoundIndex(
        self: *Self,
        input_index: usize,
        batch: usize,
        new_len: usize,
    ) api_errors.ExecuteError!usize {
        const src = self.bound_inputs[input_index] orelse return error.InvalidArgument;
        if (src.dtype != .i32) return error.InvalidArgument;
        const total = src.elemCount() catch return error.InvalidArgument;
        if (total < batch) return error.InvalidArgument;

        var stack_starts: [64]i32 = undefined;
        var heap_starts: ?[]i32 = null;
        defer if (heap_starts) |buf| self.allocator.free(buf);

        const starts: []i32 = if (total <= stack_starts.len) blk: {
            const buf = stack_starts[0..total];
            src.read(buf) catch return error.InvalidArgument;
            break :blk buf;
        } else blk: {
            const buf = src.readAlloc(self.allocator, i32) catch return error.InvalidArgument;
            heap_starts = buf;
            break :blk buf;
        };

        var end_max: usize = 0;
        for (starts[0..batch]) |raw| {
            if (raw < 0) return error.InvalidArgument;
            const end = std.math.add(usize, @intCast(raw), new_len) catch return error.InvalidArgument;
            end_max = @max(end_max, end);
        }
        return end_max;
    }

    pub fn outputTensor(self: *Self, name: []const u8) api_errors.ApiError!Tensor {
        const cache_index = self.last_run_cache_index orelse return api_errors.ApiError.InvalidArgument;
        const output_index = self.findOutputIndex(name) orelse return api_errors.ApiError.InvalidArgument;
        return self.resolveOutputTensor(cache_index, output_index);
    }

    pub fn outputCount(self: *const Self) usize {
        return self.output_signatures.len;
    }

    /// Whether output `index` is io-aliased recurrent state (a `next_*` carry the
    /// runtime already writes back into its input slot). Lets bindings skip
    /// copying carried state out by default.
    pub fn outputIsAliasedState(self: *const Self, index: usize) bool {
        if (index >= self.output_alias_input_indices.len) return false;
        return self.output_alias_input_indices[index] != types_mod.invalid_alias_index;
    }

    /// Fetch the most recent run's output by position (in declared output order).
    pub fn outputTensorAt(self: *Self, index: usize) api_errors.ApiError!Tensor {
        const cache_index = self.last_run_cache_index orelse return api_errors.ApiError.InvalidArgument;
        if (index >= self.output_signatures.len) return api_errors.ApiError.InvalidArgument;
        return self.resolveOutputTensor(cache_index, index);
    }

    /// Resolve an output to a host-readable tensor. Device outputs are copied
    /// explicitly into a lazy host mirror; CPU outputs are returned directly.
    fn resolveOutputTensor(self: *Self, cache_index: usize, output_index: usize) api_errors.ApiError!Tensor {
        const entry: *const CacheEntry = &self.cache_entries.items[cache_index];
        const tid = entry.program.outputs[output_index];
        const on_device = (self.store.tensorDevice(tid) catch DeviceRef{}).kind != .cpu;
        const evictable = self.store.tensorIsWorkspace(tid) catch false;
        if (on_device or evictable) return self.mirrorOutputToHost(output_index, tid);
        const t = try self.store.getConst(tid);
        return .{ .store = self.store, .id = tid, .dtype = t.dtype, .shape = t.shape };
    }

    /// Gather a device-exclusive output into a lazy, reusable host mirror.
    /// Recreate the mirror if the source shape changed.
    fn mirrorOutputToHost(self: *Self, output_index: usize, src_tid: TensorId) api_errors.ApiError!Tensor {
        const src_meta = try self.store.getConst(src_tid);
        const need_new = blk: {
            const cur = self.output_host_mirror_tids[output_index];
            if (cur == types_mod.invalid_tensor_id) break :blk true;
            const m = self.store.getConst(cur) catch break :blk true;
            break :blk m.dtype != src_meta.dtype or !signatures.sameUsize(m.shape, src_meta.shape);
        };
        if (need_new) {
            const old = self.output_host_mirror_tids[output_index];
            if (old != types_mod.invalid_tensor_id) self.store.releaseTensorData(old) catch {};
            self.output_host_mirror_tids[output_index] =
                try initializers.createTensorSingleTile(self.store, self.target.tiles, src_meta.dtype, src_meta.shape);
        }
        const mirror_tid = self.output_host_mirror_tids[output_index];
        try self.store.copyTensorData(mirror_tid, src_tid);
        const m = try self.store.getConst(mirror_tid);
        return .{ .store = self.store, .id = mirror_tid, .dtype = m.dtype, .shape = m.shape };
    }

    /// Execute, then return output `index` as a Tensor handle (tensor-first API).
    pub fn runOutputTensor(self: *Self, index: usize) api_errors.ExecuteError!Tensor {
        try self.run();
        return self.outputTensorAt(index) catch error.InvalidArgument;
    }

    /// Execute, then return output `index` as a newly allocated f32 slice (caller owns).
    pub fn runOutputF32Alloc(self: *Self, allocator: std.mem.Allocator, index: usize) api_errors.ExecuteError![]f32 {
        try self.run();
        const t: Tensor = self.outputTensorAt(index) catch return error.InvalidArgument;
        return t.readF32Alloc(allocator);
    }

    /// Execute, then return output `index` as a newly allocated f16 slice (caller owns).
    pub fn runOutputF16Alloc(self: *Self, allocator: std.mem.Allocator, index: usize) api_errors.ExecuteError![]f16 {
        try self.run();
        const t: Tensor = self.outputTensorAt(index) catch return error.InvalidArgument;
        return t.readF16Alloc(allocator);
    }

    /// Execute, then return scalar output `index` as f32 (requires f32, exactly 1 element).
    pub fn runScalarF32(self: *Self, index: usize) api_errors.ExecuteError!f32 {
        try self.run();
        const t: Tensor = self.outputTensorAt(index) catch return error.InvalidArgument;
        const n: usize = try t.elemCount();
        if (n != 1) return api_errors.ExecuteError.InvalidArgument;
        var tmp: [1]f32 = .{0.0};
        try t.readF32(tmp[0..1]);
        return tmp[0];
    }

    /// Whether every `SequenceAppend` position is externally available before execution.
    /// In-graph positions require preallocated caches because lazy growth cannot size them.
    fn appendPositionsAreExternal(t: template_mod.Template) bool {
        for (t.nodes) |node| {
            if (!appendPositionIsExternal(t, node)) return false;
        }
        for (t.regions) |region| {
            for (region.nodes) |node| {
                if (!appendPositionIsExternal(t, node)) return false;
            }
        }
        return true;
    }

    fn appendPositionIsExternal(t: template_mod.Template, node: package_file.NodeRecord) bool {
        if (node.op != .SequenceAppend) return true;
        if (node.inputs.len < 3) return true;
        return t.values[node.inputs[2]].source != .produced;
    }

    /// Build a model uniformly from a loaded or compiled template, taking `owner` and
    /// `params` on success.
    pub fn init(
        allocator: std.mem.Allocator,
        backend: backend_mod.Backend,
        store: *StorageManager,
        target: Target,
        owner: TemplateOwner,
        params: Params,
        package_hash: u64,
        opts: LoadModelOptions,
    ) api_errors.LoadError!Self {
        const input_signatures = try signatures.buildSignatures(allocator, owner.values(), owner.namedInputs());
        errdefer allocator.free(input_signatures);
        const output_signatures = try signatures.buildSignatures(allocator, owner.values(), owner.namedOutputs());
        errdefer allocator.free(output_signatures);

        // The template needs its public inputs and outputs as value ids; the signatures
        // are that list, named.
        const template_inputs = try allocator.alloc(u32, input_signatures.len);
        errdefer allocator.free(template_inputs);
        for (input_signatures, 0..) |sig, i| template_inputs[i] = sig.value;
        const template_outputs = try allocator.alloc(u32, output_signatures.len);
        errdefer allocator.free(template_outputs);
        for (output_signatures, 0..) |sig, i| template_outputs[i] = sig.value;

        var template = owner.template();
        template.inputs = template_inputs;
        template.outputs = template_outputs;

        const io_aliases = owner.ioAliases();
        const input_roles_src = owner.inputRoles();
        const dim_symbol_count = template.dim_symbol_count;
        // `io_aliases` are an optimization hint (copy outputs back into input slots).
        // Filter incompatible ones so executing never fails on an alias dtype/rank mismatch.
        var valid_alias_count: usize = 0;
        for (io_aliases) |alias| {
            const ii: usize = alias.input;
            const oi: usize = alias.output;
            if (ii >= input_signatures.len or oi >= output_signatures.len) continue;
            if (input_signatures[ii].dtype != output_signatures[oi].dtype) continue;
            if (input_signatures[ii].rank != output_signatures[oi].rank) continue;
            valid_alias_count += 1;
        }

        const output_aliases = try allocator.alloc(IoAliasInfo, valid_alias_count);
        errdefer allocator.free(output_aliases);

        const input_alias_output_indices = try allocator.alloc(u32, input_signatures.len);
        errdefer allocator.free(input_alias_output_indices);
        @memset(input_alias_output_indices, types_mod.invalid_alias_index);

        const output_alias_input_indices = try allocator.alloc(u32, output_signatures.len);
        errdefer allocator.free(output_alias_input_indices);
        @memset(output_alias_input_indices, types_mod.invalid_alias_index);

        const output_host_mirror_tids = try allocator.alloc(TensorId, output_signatures.len);
        errdefer allocator.free(output_host_mirror_tids);
        @memset(output_host_mirror_tids, types_mod.invalid_tensor_id);

        var alias_cursor: usize = 0;
        for (io_aliases) |alias| {
            const ii: usize = alias.input;
            const oi: usize = alias.output;
            if (ii >= input_signatures.len or oi >= output_signatures.len) continue;
            if (input_signatures[ii].dtype != output_signatures[oi].dtype) continue;
            if (input_signatures[ii].rank != output_signatures[oi].rank) continue;

            if (input_alias_output_indices[ii] != types_mod.invalid_alias_index) return error.InvalidArgument;
            if (output_alias_input_indices[oi] != types_mod.invalid_alias_index) return error.InvalidArgument;
            input_alias_output_indices[ii] = @intCast(oi);
            output_alias_input_indices[oi] = @intCast(ii);

            output_aliases[alias_cursor] = .{
                .input_name = input_signatures[ii].name,
                .output_name = output_signatures[oi].name,
                .input_index = ii,
                .output_index = oi,
            };
            alias_cursor += 1;
        }
        std.debug.assert(alias_cursor == output_aliases.len);

        const bound_inputs = try allocator.alloc(?Tensor, input_signatures.len);
        errdefer allocator.free(bound_inputs);
        @memset(bound_inputs, null);

        const aliased_input_bind_versions = try allocator.alloc(u64, input_signatures.len);
        errdefer allocator.free(aliased_input_bind_versions);
        @memset(aliased_input_bind_versions, 0);

        const auto_input_tids = try allocator.alloc(TensorId, input_signatures.len);
        errdefer allocator.free(auto_input_tids);
        @memset(auto_input_tids, types_mod.invalid_tensor_id);

        const aliased_state_tids = try allocator.alloc(TensorId, input_signatures.len);
        errdefer allocator.free(aliased_state_tids);
        @memset(aliased_state_tids, types_mod.invalid_tensor_id);

        const aliased_state_synced_versions = try allocator.alloc(u64, input_signatures.len);
        errdefer allocator.free(aliased_state_synced_versions);
        @memset(aliased_state_synced_versions, 0);

        const state_input_policies = try allocator.alloc(?SequenceCachePolicy, input_signatures.len);
        errdefer allocator.free(state_input_policies);
        @memset(state_input_policies, null);

        const input_roles = try allocator.alloc(?package_file.InputRole, input_signatures.len);
        errdefer allocator.free(input_roles);
        @memset(input_roles, null);
        for (input_roles_src) |role| {
            if (role.input >= input_signatures.len) return error.InvalidArgument;
            input_roles[role.input] = role;
        }

        const role_input_tids = try allocator.alloc(TensorId, input_signatures.len);
        errdefer allocator.free(role_input_tids);
        @memset(role_input_tids, types_mod.invalid_tensor_id);

        const role_auto_bound = try allocator.alloc(bool, input_signatures.len);
        errdefer allocator.free(role_auto_bound);
        @memset(role_auto_bound, false);

        const symbol_defaults = try allocator.alloc(?u64, dim_symbol_count);
        errdefer allocator.free(symbol_defaults);
        @memset(symbol_defaults, null);

        // Locate the singleton control roles and apply the load-time cache options
        // to role-declared sequence caches (policy + capacity-symbol default).
        var role_tokens_index: ?usize = null;
        var role_positions_index: ?usize = null;
        var role_write_index_index: ?usize = null;
        var role_visible_end_index: ?usize = null;
        for (input_roles, 0..) |role_opt, i| {
            const role = role_opt orelse continue;
            switch (role.kind) {
                .tokens => role_tokens_index = i,
                .positions => role_positions_index = i,
                .cache_write_index => role_write_index_index = i,
                .cache_visible_end => role_visible_end_index = i,
                .sequence_cache => {
                    if (role.capacity_symbol == package_file.invalid_index) continue;
                    if (role.capacity_symbol >= dim_symbol_count) return error.InvalidArgument;
                    if (opts.cache.capacity_tokens == 0) continue;
                    const cap: usize = opts.cache.capacity_tokens;
                    // Lazy growth requires a supported rank-4 layout, package opt-in,
                    // and a host-known write position; otherwise preallocate fully.
                    const growable: ?types_mod.CacheGrowth = opts.cache.growable;
                    if (growable != null and
                        (role.flags & package_file.InputRoleFlags.allow_growable) != 0 and
                        input_signatures[i].rank == 4 and role.axis == 1 and
                        appendPositionsAreExternal(template))
                    {
                        const g = growable.?;
                        const initial: usize = @max(1, @min(g.initial_capacity_tokens, cap));
                        if (!store.hasCache()) {
                            store.configureCache(.{ .ram_budget_bytes = std.math.maxInt(usize) }) catch return error.OutOfMemory;
                        }
                        state_input_policies[i] = .{ .growable = .{
                            .initial_capacity_tokens = initial,
                            .growth_numerator = g.growth_numerator,
                            .growth_denominator = g.growth_denominator,
                            .max_capacity_tokens = cap,
                        } };
                        symbol_defaults[role.capacity_symbol] = initial;
                    } else {
                        symbol_defaults[role.capacity_symbol] = cap;
                    }
                },
                .state => {},
            }
        }
        const auto_positions_enabled = opts.auto_positions and role_tokens_index != null and
            (role_write_index_index != null or role_visible_end_index != null or role_positions_index != null);

        var total_rank: usize = 0;
        for (input_signatures) |sig| total_rank += sig.rank;

        const run_symbol_bindings = try allocator.alloc(?u64, dim_symbol_count);
        errdefer allocator.free(run_symbol_bindings);
        const run_symbol_values = try allocator.alloc(u64, dim_symbol_count);
        errdefer allocator.free(run_symbol_values);
        const run_input_shapes = try allocator.alloc(usize, total_rank);
        errdefer allocator.free(run_input_shapes);
        const run_direct_input_ids = try allocator.alloc(TensorId, input_signatures.len);
        errdefer allocator.free(run_direct_input_ids);

        // Bind an execution session to the store now. Per-model lifetime: device
        // residency persists across `run` calls and is torn down in `deinit`.
        const session = try backend.createSession(store.tensorStore());
        errdefer session.deinit();

        return .{
            .allocator = allocator,
            .backend = backend,
            .store = store,
            .session = session,
            .target = target,
            .template = template,
            .template_owner = owner,
            .params = params,
            .input_signatures = input_signatures,
            .output_signatures = output_signatures,
            .output_aliases = output_aliases,
            .input_alias_output_indices = input_alias_output_indices,
            .output_alias_input_indices = output_alias_input_indices,
            .output_host_mirror_tids = output_host_mirror_tids,
            .bound_inputs = bound_inputs,
            .aliased_input_bind_versions = aliased_input_bind_versions,
            .auto_input_tids = auto_input_tids,
            .auto_init_inputs = opts.auto_init_inputs,
            .aliased_state_tids = aliased_state_tids,
            .aliased_state_synced_versions = aliased_state_synced_versions,
            .state_input_policies = state_input_policies,
            .input_roles = input_roles,
            .role_tokens_index = role_tokens_index,
            .role_positions_index = role_positions_index,
            .role_write_index_index = role_write_index_index,
            .role_visible_end_index = role_visible_end_index,
            .auto_positions_enabled = auto_positions_enabled,
            .role_input_tids = role_input_tids,
            .role_auto_bound = role_auto_bound,
            .symbol_defaults = symbol_defaults,
            .run_symbol_bindings = run_symbol_bindings,
            .run_symbol_values = run_symbol_values,
            .run_input_shapes = run_input_shapes,
            .run_direct_input_ids = run_direct_input_ids,
            .cache_workspace_budget_bytes = opts.plan_cache_budget_bytes,
            .package_hash = package_hash,
            .trace_runs = traceEnabled(),
        };
    }

    fn resolveBindings(self: *Self) error{InvalidArgument}!ResolvedInputs {
        @memset(self.run_symbol_bindings, null);
        @memset(self.run_direct_input_ids, types_mod.invalid_tensor_id);

        var shape_cursor: usize = 0;
        var i: usize = 0;
        while (i < self.input_signatures.len) : (i += 1) {
            const tensor = self.bound_inputs[i] orelse return error.InvalidArgument;
            const sig = self.input_signatures[i];
            const terms = self.template.inputShapeTerms(i);
            if (tensor.dtype != sig.dtype or tensor.shape.len != sig.rank) return error.InvalidArgument;

            var d: usize = 0;
            while (d < tensor.shape.len) : (d += 1) {
                const actual: u64 = @intCast(tensor.shape[d]);
                try signatures.bindInputDimExprs(self.template.dim_exprs, terms[d], actual, self.run_symbol_bindings);
                self.run_input_shapes[shape_cursor] = tensor.shape[d];
                shape_cursor += 1;
            }
            if (self.inputAliasOutputIndex(i) == null) self.run_direct_input_ids[i] = tensor.id;
        }

        for (self.run_symbol_bindings, 0..) |value, idx| self.run_symbol_values[idx] = value orelse 0;

        return .{
            .symbol_values = self.run_symbol_values,
            .input_shapes = self.run_input_shapes[0..shape_cursor],
            .direct_input_ids = self.run_direct_input_ids,
        };
    }

    /// Resolve dimension symbols from the inputs the caller actually bound (skipping
    /// unbound ones). Returns `run_symbol_values` (scratch), with undetermined
    /// symbols left as 0. Used by `ensureAutoInputs` to size unbound inputs.
    fn resolveSymbolsFromBoundInputs(self: *Self) error{InvalidArgument}![]const u64 {
        @memset(self.run_symbol_bindings, null);
        for (self.input_signatures, 0..) |sig, i| {
            const tensor = self.bound_inputs[i] orelse continue;
            const terms = self.template.inputShapeTerms(i);
            if (tensor.dtype != sig.dtype or tensor.shape.len != sig.rank) return error.InvalidArgument;
            var d: usize = 0;
            while (d < tensor.shape.len) : (d += 1) {
                try signatures.bindInputDimExprs(self.template.dim_exprs, terms[d], @intCast(tensor.shape[d]), self.run_symbol_bindings);
            }
        }
        for (self.run_symbol_bindings, 0..) |v, idx| self.run_symbol_values[idx] = v orelse 0;
        return self.run_symbol_values;
    }

    /// Bind one auto-managed i32 control input: (re)allocate its persistent tensor
    /// on shape change, fill it with `values`, and self-bind it. Skips inputs the
    /// caller bound manually (`role_auto_bound[i] == false` while bound).
    fn bindRoleTensor(self: *Self, index: usize, shape: []const usize, values: []const i32) api_errors.ExecuteError!void {
        if (self.bound_inputs[index] != null and !self.role_auto_bound[index]) return;

        var tid = self.role_input_tids[index];
        const need_new = blk: {
            if (tid == types_mod.invalid_tensor_id) break :blk true;
            const meta = self.store.getConst(tid) catch break :blk true;
            break :blk !signatures.sameUsize(meta.shape, shape);
        };
        if (need_new) {
            tid = initializers.createTensorForShape(self.store, self.target.tiles, .i32, shape) catch return error.OutOfMemory;
            self.role_input_tids[index] = tid;
        }
        const meta = self.store.getConst(tid) catch return error.InvalidArgument;
        const t = Tensor{ .store = self.store, .id = tid, .dtype = .i32, .shape = meta.shape };
        t.write(values) catch return error.InvalidArgument;
        self.bound_inputs[index] = t;
        self.role_auto_bound[index] = true;
    }

    /// Populate role-declared controls from the tracked sequence position and the bound
    /// token input's current sequence length.
    fn ensureAutoPositions(self: *Self, trace: bool) api_errors.ExecuteError!void {
        if (!self.auto_positions_enabled) return;
        const ti = self.role_tokens_index.?;
        const tokens = self.bound_inputs[ti] orelse return; // nothing to derive from
        const tokens_role = self.input_roles[ti].?;
        if (tokens_role.axis == package_file.input_role_no_axis or tokens_role.axis >= tokens.shape.len) return error.InvalidArgument;

        const new_tokens: u64 = @intCast(tokens.shape[tokens_role.axis]);
        const batch: usize = if (tokens.shape.len >= 2) tokens.shape[0] else 1;
        if (self.position_tokens + new_tokens > std.math.maxInt(i32)) return error.InvalidArgument;
        const pos: i32 = @intCast(self.position_tokens);
        const end: i32 = @intCast(self.position_tokens + new_tokens);

        if (trace) std.debug.print(
            "[aion][run] auto positions: pos={d} new_tokens={d} batch={d}\n",
            .{ pos, new_tokens, batch },
        );

        var small: [8]i32 = undefined;
        if (self.role_write_index_index) |wi| {
            if (self.input_signatures[wi].rank != 1) return error.InvalidArgument;
            const vals = if (batch <= small.len) small[0..batch] else try self.allocator.alloc(i32, batch);
            defer if (batch > small.len) self.allocator.free(vals);
            @memset(vals, pos);
            try self.bindRoleTensor(wi, &[_]usize{batch}, vals);
        }
        if (self.role_visible_end_index) |vi| {
            if (self.input_signatures[vi].rank != 1) return error.InvalidArgument;
            const vals = if (batch <= small.len) small[0..batch] else try self.allocator.alloc(i32, batch);
            defer if (batch > small.len) self.allocator.free(vals);
            @memset(vals, end);
            try self.bindRoleTensor(vi, &[_]usize{batch}, vals);
        }
        if (self.role_positions_index) |pi| {
            const sig = self.input_signatures[pi];
            const n: usize = @intCast(new_tokens);
            const rows: usize = if (sig.rank == 2) batch else if (sig.rank == 1) 1 else return error.InvalidArgument;
            const vals = try self.allocator.alloc(i32, rows * n);
            defer self.allocator.free(vals);
            for (0..rows) |r| {
                for (0..n) |s| vals[r * n + s] = pos + @as(i32, @intCast(s));
            }
            if (sig.rank == 2) {
                try self.bindRoleTensor(pi, &[_]usize{ rows, n }, vals);
            } else {
                try self.bindRoleTensor(pi, &[_]usize{n}, vals);
            }
        }

        self.pending_position_advance = new_tokens;
    }

    /// Bind unbound inputs to persistent zeroed slots shaped from resolved symbols.
    /// Recurrent slots are seeded once and carried across runs; disabling auto-init
    /// makes any unbound input an error.
    fn ensureAutoInputs(self: *Self, trace: bool) api_errors.ExecuteError!void {
        var any_unbound = false;
        for (self.bound_inputs) |b| {
            if (b == null) {
                any_unbound = true;
                break;
            }
        }
        if (!any_unbound) return;

        if (!self.auto_init_inputs) {
            for (self.input_signatures, 0..) |sig, i| {
                if (self.bound_inputs[i] == null) {
                    if (trace) std.debug.print("[aion][run] strict mode: required input '{s}' is unbound\n", .{sig.name});
                    return error.InvalidArgument;
                }
            }
            return;
        }

        const symbol_values = try self.resolveSymbolsFromBoundInputs();
        const optional_symbols = self.allocator.alloc(?u64, symbol_values.len) catch return error.OutOfMemory;
        // A symbol left at 0 is one no bound input determined; shape terms treat that as
        // unresolved rather than as a zero-sized axis.
        for (symbol_values, 0..) |v, i| optional_symbols[i] = if (v == 0) null else v;
        defer self.allocator.free(optional_symbols);

        // Symbols no bound input determines fall back to their load-time default
        // (role-declared cache capacity from `LoadModelOptions.cache`), letting the
        // runtime auto-allocate caches with a free capacity axis.
        for (optional_symbols, 0..) |*sym, k| {
            if (sym.* == null) sym.* = self.symbol_defaults[k];
        }

        for (self.input_signatures, 0..) |sig, i| {
            if (self.bound_inputs[i] != null) continue;

            const shape = package_file.resolveShapeTermsExprs(self.allocator, self.template.dim_exprs, self.template.inputShapeTerms(i), optional_symbols) catch {
                if (trace) std.debug.print(
                    "[aion][run] cannot auto-size unbound input '{s}': a symbolic dimension is not determined by any bound input — bind it explicitly\n",
                    .{sig.name},
                );
                return error.InvalidArgument;
            };
            defer self.allocator.free(shape);

            const is_aliased = self.inputAliasOutputIndex(i) != null;

            // Allocate a fresh zero slot if none exists yet or the resolved shape changed.
            const need_new = blk: {
                const cur = self.auto_input_tids[i];
                if (cur == types_mod.invalid_tensor_id) break :blk true;
                const meta = self.store.getConst(cur) catch break :blk true;
                break :blk !signatures.sameUsize(meta.shape, shape);
            };

            if (need_new) {
                const tid = if (is_aliased)
                    try initializers.createTensorSingleTile(self.store, self.target.tiles, sig.dtype, shape)
                else
                    try initializers.createTensorForShape(self.store, self.target.tiles, sig.dtype, shape);
                // Freshly created store tensors are zero-initialized.
                self.auto_input_tids[i] = tid;
                if (trace) std.debug.print(
                    "[aion][run] auto-init input '{s}' dtype={s} shape={any} aliased={any}\n",
                    .{ sig.name, @tagName(sig.dtype), shape, is_aliased },
                );
            }

            const tid = self.auto_input_tids[i];
            const meta = try self.store.getConst(tid);
            self.bound_inputs[i] = .{ .store = self.store, .id = tid, .dtype = meta.dtype, .shape = meta.shape };

            // Seed the zero slot into the cache entry exactly once; from then on the
            // alias sync carries state. Bumping only on a fresh allocation avoids
            // clobbering carried state on later runs.
            if (is_aliased and need_new) {
                self.aliased_input_bind_versions[i] +%= 1;
                if (self.aliased_input_bind_versions[i] == 0) self.aliased_input_bind_versions[i] = 1;
            }
        }
    }

    fn ensureCacheEntry(
        self: *Self,
        symbol_values: []const u64,
        input_shapes: []const usize,
        direct_input_ids: []const TensorId,
    ) api_errors.ExecuteError!usize {
        for (self.cache_entries.items, 0..) |*entry, idx| {
            if (signatures.sameU64(entry.symbol_values, symbol_values) and signatures.sameUsize(entry.input_shapes, input_shapes)) {
                if (try self.prepareCacheEntryInputs(entry, direct_input_ids)) {
                    self.cache_clock +%= 1;
                    entry.last_used = self.cache_clock;
                    return idx;
                }
            }
        }

        var entry = try self.buildCacheEntry(symbol_values, input_shapes, direct_input_ids);
        errdefer self.destroyCacheEntry(&entry);
        self.cache_build_count +%= 1;
        self.cache_clock +%= 1;
        entry.last_used = self.cache_clock;

        // Protect the actual resource, not an arbitrary entry count. Keep one
        // oversized specialization rather than creating a compile-on-every-run
        // cliff for a shape whose workspace alone exceeds the configured budget.
        if (self.cache_entries.items.len != 0 and
            self.cache_workspace_bytes + entry.workspace_bytes > self.cache_workspace_budget_bytes)
        {
            self.session.retireResources();
        }
        while (self.cache_entries.items.len != 0 and
            self.cache_workspace_bytes + entry.workspace_bytes > self.cache_workspace_budget_bytes)
        {
            var oldest_index: usize = 0;
            var oldest_tick = self.cache_entries.items[0].last_used;
            for (self.cache_entries.items[1..], 1..) |candidate, idx| {
                if (candidate.last_used < oldest_tick) {
                    oldest_tick = candidate.last_used;
                    oldest_index = idx;
                }
            }
            var evicted = self.cache_entries.orderedRemove(oldest_index);
            if (self.last_run_cache_index) |last| {
                self.last_run_cache_index = if (last == oldest_index)
                    null
                else if (last > oldest_index)
                    last - 1
                else
                    last;
            }
            self.cache_workspace_bytes -= evicted.workspace_bytes;
            self.destroyCacheEntry(&evicted);
            self.cache_eviction_count +%= 1;
        }

        try self.placeProgram(&entry.program);

        try self.cache_entries.append(self.allocator, entry);
        self.cache_workspace_bytes += entry.workspace_bytes;
        self.traceDeviceBreakdown(&self.cache_entries.items[self.cache_entries.items.len - 1]);
        return self.cache_entries.items.len - 1;
    }

    /// Materialize every device placement declared by the compiled program.
    /// The placement table covers parameters, derived weights, workspace, and input
    /// slots; failures are errors rather than host-staging downgrades.
    fn placeProgram(self: *Self, program: *const program_mod.Program) api_errors.ExecuteError!void {
        if (self.target.device.kind == .cpu) return;

        const mem = self.store.deviceMemoryFor(self.target.device) orelse return error.InvalidArgument;
        program_mod.materializePlacements(self.store, program, self.target.device, mem) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.InvalidArgument,
        };

        traceStorageTotals(self, "placed");
    }

    /// `AION_TRACE_MEMORY` accounting: host bytes still held vs device bytes.
    fn traceStorageTotals(self: *const Self, phase: []const u8) void {
        if (!env_util.flagEnabled("AION_TRACE_MEMORY")) return;
        var cpu_bytes: usize = 0;
        var gpu_bytes: usize = 0;
        for (self.store.tensors.items) |tensor| {
            if (tensor.backing_owner != null) continue;
            if (tensor.device.kind == .cpu) {
                cpu_bytes +|= tensor.backing_bytes;
            } else {
                gpu_bytes +|= tensor.backing_bytes;
            }
        }
        std.debug.print("[aion][storage] {s} cpu_backing={d} gpu_backing={d}\n", .{ phase, cpu_bytes, gpu_bytes });
    }

    fn idIn(ids: []const TensorId, id: TensorId) bool {
        for (ids) |x| if (x == id) return true;
        return false;
    }

    /// Device bytes split by role, so a growth in one is not mistaken for another.
    fn traceDeviceBreakdown(self: *const Self, entry: *const CacheEntry) void {
        if (!env_util.flagEnabled("AION_TRACE_MEMORY")) return;
        var init_b: usize = 0;
        var ws_b: usize = 0;
        var slot_b: usize = 0;
        var other_b: usize = 0;
        var ws_n: usize = 0;
        for (self.store.tensors.items, 0..) |tensor, idx| {
            if (tensor.backing_owner != null) continue;
            if (tensor.device.kind == .cpu) continue;
            const b = tensor.backing_bytes;
            const id: TensorId = @intCast(idx);
            if (self.params.contains(id)) {
                init_b +|= b;
            } else if (idIn(entry.program.owned_tensors, id)) {
                ws_b +|= b;
                ws_n += 1;
            } else if (idIn(entry.owned_input_slots, id)) {
                slot_b +|= b;
            } else {
                other_b +|= b;
            }
        }
        std.debug.print("[aion][storage] breakdown init={d} workspace={d} (n={d}) slots={d} other={d}\n", .{ init_b, ws_b, ws_n, slot_b, other_b });
    }

    fn destroyCacheEntry(self: *Self, entry: *CacheEntry) void {
        lease.release(self.store, &entry.program);
        for (entry.owned_input_slots) |tid| self.store.releaseTensorData(tid) catch {};
        if (entry.owned_input_slots.len != 0) self.allocator.free(entry.owned_input_slots);
        for (entry.program.owned_tensors) |tid| {
            self.store.releaseTensorData(tid) catch {};
        }
        entry.program.deinit();
        self.allocator.free(entry.symbol_values);
        self.allocator.free(entry.input_shapes);
        self.allocator.free(entry.input_slots);
        self.allocator.free(entry.direct_input_ids);
    }

    /// Specialize either template source for the requested shapes, bind input slots,
    /// and compile through `finishCacheEntry`.
    fn buildCacheEntry(
        self: *Self,
        symbol_values: []const u64,
        input_shapes: []const usize,
        direct_input_ids: []const TensorId,
    ) api_errors.ExecuteError!CacheEntry {
        const in_ids = try self.allocator.alloc(graph_mod.ValueId, self.input_signatures.len);
        defer self.allocator.free(in_ids);
        const out_ids = try self.allocator.alloc(graph_mod.ValueId, self.output_signatures.len);
        defer self.allocator.free(out_ids);

        var graph = self.template.specialize(
            self.allocator,
            self.params.by_value,
            symbol_values,
            input_shapes,
            in_ids,
            out_ids,
        ) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.InvalidArgument,
        };
        defer graph.deinit();

        return self.finishCacheEntry(&graph, in_ids, out_ids, symbol_values, input_shapes, direct_input_ids);
    }

    /// Shared tail of cache-entry construction: bind each public input's value to its
    /// slot (aliased → model-level shared state slot; non-aliased → the caller's
    /// tensor), set outputs, and compile. Operates on `graph` in place.
    fn finishCacheEntry(
        self: *Self,
        graph: *graph_mod.Graph,
        in_ids: []const graph_mod.ValueId,
        out_ids: []const graph_mod.ValueId,
        symbol_values: []const u64,
        input_shapes: []const usize,
        direct_input_ids: []const TensorId,
    ) api_errors.ExecuteError!CacheEntry {
        const input_slots = try self.allocator.alloc(TensorId, self.input_signatures.len);
        errdefer self.allocator.free(input_slots);
        const entry_direct_input_ids = try self.allocator.dupe(TensorId, direct_input_ids);
        errdefer self.allocator.free(entry_direct_input_ids);
        var owned_input_slots: std.ArrayList(TensorId) = .empty;
        errdefer {
            for (owned_input_slots.items) |tid| self.store.releaseTensorData(tid) catch {};
            owned_input_slots.deinit(self.allocator);
        }

        var input_shape_cursor: usize = 0;
        for (self.input_signatures, 0..) |_, sig_idx| {
            const concrete_shape = buildConcreteShapeFromFlat(input_shapes, self.input_signatures, sig_idx, &input_shape_cursor);
            const slot_tid = if (self.inputAliasOutputIndex(sig_idx) != null)
                // Share recurrent state across cache entries and keep it single-tile
                // because SequenceAppend requires a contiguous last axis.
                try self.ensureAliasedStateSlot(sig_idx, concrete_shape)
            else if (self.target.device.kind != .cpu) blk: {
                // A GPU program binds a stable device slot, never the caller's
                // host tensor. `run()` performs the explicit H2D copy into this
                // slot, so execution has one backing and no staged-input cache.
                const tid = try initializers.createTensorSingleTile(self.store, self.target.tiles, self.input_signatures[sig_idx].dtype, concrete_shape);
                owned_input_slots.append(self.allocator, tid) catch {
                    self.store.releaseTensorData(tid) catch {};
                    return error.OutOfMemory;
                };
                self.migrateStateSlotToDevice(tid, self.input_signatures[sig_idx].dtype, concrete_shape);
                if ((self.store.tensorDevice(tid) catch DeviceRef{}).kind == .cpu) return error.InvalidArgument;
                break :blk tid;
            } else blk: {
                const tid = direct_input_ids[sig_idx];
                if (tid == types_mod.invalid_tensor_id) return error.InvalidArgument;
                break :blk tid;
            };
            input_slots[sig_idx] = slot_tid;
            try graph.bindExternal(in_ids[sig_idx], slot_tid);
        }

        try graph.setOutputs(out_ids);

        var program = try program_mod.compileGraph(self.allocator, graph, self.store, self.target);
        errdefer {
            for (program.owned_tensors) |tid| self.store.releaseTensorData(tid) catch {};
            program.deinit();
        }
        // The program now exists, which is the only thing this layer knows about weight
        // lifetime: the store decides what that implies (see `graph/program/lease.zig`).
        lease.acquire(self.store, &program) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            else => return error.InvalidArgument,
        };
        errdefer lease.release(self.store, &program);
        traceStorageTotals(self, "leased");

        var workspace_bytes: usize = program.workspace_bytes;
        for (owned_input_slots.items) |tid| {
            const bytes = self.store.tensorBackingBytes(tid) catch return error.InvalidArgument;
            workspace_bytes = std.math.add(usize, workspace_bytes, bytes) catch return error.InvalidArgument;
        }
        return .{
            .symbol_values = try self.allocator.dupe(u64, symbol_values),
            .input_shapes = try self.allocator.dupe(usize, input_shapes),
            .input_slots = input_slots,
            .direct_input_ids = entry_direct_input_ids,
            .owned_input_slots = try owned_input_slots.toOwnedSlice(self.allocator),
            .program = program,
            .workspace_bytes = workspace_bytes,
        };
    }

    /// Return or lazily allocate the shared single-tile slot for an io-aliased input.
    /// Replacing it after a capacity change resets the seeded bind version.
    fn ensureAliasedStateSlot(self: *Self, input_index: usize, shape: []const usize) api_errors.ExecuteError!TensorId {
        const sig = self.input_signatures[input_index];
        const growable = self.stateInputGrowable(input_index);
        const cur = self.aliased_state_tids[input_index];
        if (cur != types_mod.invalid_tensor_id) {
            const meta = self.store.getConst(cur) catch return error.InvalidArgument;
            // Reuse grown slots for smaller declared capacities so later cache entries
            // do not discard existing state.
            if (stateSlotCompatible(meta.shape, shape, growable)) return cur;
            // Capacity truly changed: re-seed into the new slot on the next run.
            self.aliased_state_synced_versions[input_index] = 0;
        }
        const tid = try initializers.createTensorSingleTile(self.store, self.target.tiles, sig.dtype, shape);
        if (self.state_input_policies[input_index]) |pol| {
            self.store.registerSequenceCachePolicy(tid, pol) catch return error.InvalidArgument;
        }
        self.migrateStateSlotToDevice(tid, sig.dtype, shape);
        self.aliased_state_tids[input_index] = tid;
        return tid;
    }

    fn stateInputGrowable(self: *const Self, input_index: usize) bool {
        if (self.state_input_policies[input_index]) |pol| return std.meta.activeTag(pol) == .growable;
        return false;
    }

    /// Whether `have` can back `want`: exact shape, or a growable rank-4 cache whose
    /// sequence axis has expanded while all other axes match.
    fn stateSlotCompatible(have: []const usize, want: []const usize, growable: bool) bool {
        if (have.len != want.len) return false;
        if (growable and have.len == 4) {
            for (have, want, 0..) |h, w, d| {
                if (d == 2) {
                    if (h < w) return false;
                } else if (h != w) return false;
            }
            return true;
        }
        return signatures.sameUsize(have, want);
    }

    /// Move a new recurrent-state slot to device-exclusive storage on a discrete GPU
    /// when it fits one buffer. CPU, unified memory, oversized slots, and migration
    /// failures retain the correct host/staged path.
    fn migrateStateSlotToDevice(self: *Self, tid: TensorId, dtype: DType, shape: []const usize) void {
        if (self.target.device.kind == .cpu) return;
        const mem = self.store.deviceMemoryFor(self.target.device) orelse return;

        const info = dtype.info();
        if (info.is_quantized) return; // createTensorSingleTile already rejects these
        var elems: u64 = 1;
        for (shape) |d| elems *= @as(u64, @intCast(d));
        const bytes: u64 = elems * @as(u64, @intCast(info.block_bytes));
        if (bytes > mem.maxBindingBytes()) return; // over the device's per-buffer ceiling

        const policy = self.store.policyFor(self.target.device);
        // Single-tile target (tile_shape == shape), matching createTensorSingleTile.
        self.store.moveTensor(tid, self.target.device, mem, shape, policy.tile_alignment) catch return;
    }

    /// Seed a recurrent slot from a caller-bound tensor using the destination's current
    /// placement: packed host copy or device-aware staged copy.
    fn copyTensorInto(self: *Self, dst_tid: TensorId, src: Tensor) api_errors.ExecuteError!void {
        const on_device = (self.store.tensorDevice(dst_tid) catch DeviceRef{}).kind != .cpu;
        if (on_device) return self.store.copyTensorData(dst_tid, src.id);
        const meta = try self.store.getConst(dst_tid);
        const dst: Tensor = .{ .store = self.store, .id = dst_tid, .dtype = meta.dtype, .shape = meta.shape };
        return dst.copyFrom(self.allocator, src);
    }

    fn findInputIndex(self: *const Self, name: []const u8) ?usize {
        for (self.input_signatures, 0..) |sig, idx| {
            if (std.mem.eql(u8, sig.name, name)) return idx;
        }
        return null;
    }

    fn findOutputIndex(self: *const Self, name: []const u8) ?usize {
        for (self.output_signatures, 0..) |sig, idx| {
            if (std.mem.eql(u8, sig.name, name)) return idx;
        }
        return null;
    }

    fn findStaticCacheEntry(self: *Self, desired_direct_input_ids: []const TensorId) api_errors.ExecuteError!?usize {
        if (self.template.dim_symbol_count != 0) return null;
        for (self.cache_entries.items, 0..) |*entry, idx| {
            if (try self.prepareCacheEntryInputs(entry, desired_direct_input_ids)) {
                self.cache_clock +%= 1;
                entry.last_used = self.cache_clock;
                return idx;
            }
        }
        return null;
    }

    fn prepareCacheEntryInputs(
        self: *Self,
        entry: *CacheEntry,
        desired_direct_input_ids: []const TensorId,
    ) api_errors.ExecuteError!bool {
        var shape_cursor: usize = 0;
        for (self.input_signatures, 0..) |sig, idx| {
            const tensor = self.bound_inputs[idx] orelse return false;
            if (tensor.dtype != sig.dtype) return false;
            if (tensor.shape.len != sig.rank) return false;
            for (tensor.shape) |dim| {
                if (shape_cursor >= entry.input_shapes.len) return false;
                if (entry.input_shapes[shape_cursor] != dim) return false;
                shape_cursor += 1;
            }

            if (self.inputAliasOutputIndex(idx) != null) continue;

            const desired_tid = desired_direct_input_ids[idx];
            if (desired_tid == types_mod.invalid_tensor_id) return false;
            const current_tid = entry.direct_input_ids[idx];
            if (current_tid == types_mod.invalid_tensor_id) return false;
            if (current_tid == desired_tid) continue;
            if (self.target.device.kind != .cpu) {
                if (!try signatures.tensorsHaveCompatibleLayout(self.store, entry.input_slots[idx], desired_tid)) return false;
                entry.direct_input_ids[idx] = desired_tid;
                continue;
            }
            if (!try signatures.tensorsHaveCompatibleLayout(self.store, current_tid, desired_tid)) return false;
            retarget.retargetProgramTensorIds(&entry.program, current_tid, desired_tid);
            // The program names the caller's tensor now, and teardown will release what
            // it names, so the count has to follow the id.
            self.store.releaseTensor(current_tid);
            self.store.retainTensor(desired_tid);
            entry.direct_input_ids[idx] = desired_tid;
            entry.input_slots[idx] = desired_tid;
        }
        return shape_cursor == entry.input_shapes.len;
    }

    fn inputAliasOutputIndex(self: *const Self, input_index: usize) ?usize {
        if (input_index >= self.input_alias_output_indices.len) return null;
        const raw = self.input_alias_output_indices[input_index];
        if (raw == types_mod.invalid_alias_index) return null;
        return @intCast(raw);
    }
};

/// Backwards-compatible alias. `Model` is the single unified executable type,
/// produced by both `ctx.compile` (in-process graph) and `ctx.loadModel*` (package).
pub const LoadedModel = Model;

const ResolvedInputs = struct {
    symbol_values: []const u64,
    input_shapes: []const usize,
    direct_input_ids: []const TensorId,
};

const CacheEntry = struct {
    symbol_values: []u64,
    input_shapes: []usize,
    input_slots: []TensorId,
    direct_input_ids: []TensorId,
    owned_input_slots: []TensorId,
    program: program_mod.Program,
    workspace_bytes: usize,
    last_used: u64 = 0,
};

fn buildConcreteShapeFromFlat(
    flat_shapes: []const usize,
    sigs: []const SignatureInfo,
    input_index: usize,
    cursor: *usize,
) []const usize {
    const start = cursor.*;
    const len = sigs[input_index].rank;
    cursor.* += len;
    return flat_shapes[start .. start + len];
}
