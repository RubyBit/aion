// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const env_util = @import("../env.zig");

const backend_mod = @import("../backend/backend.zig");
const graph_mod = @import("../graph/graph.zig");
const program_mod = @import("../graph/program.zig");
const package_file = @import("../storage/aion_file.zig");

const api_errors = @import("errors.zig");

const types_mod = @import("loaded_model/types.zig");
const signatures = @import("loaded_model/signatures.zig");
const retarget = @import("loaded_model/retarget.zig");
const instantiate = @import("loaded_model/instantiate.zig");
const initializers = @import("loaded_model/initializers.zig");
const fuse = @import("../graph/opt/fuse_horizontal_matmul.zig");

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
pub const Package = types_mod.Package;
pub const SignatureInfo = types_mod.SignatureInfo;
pub const IoAliasInfo = types_mod.IoAliasInfo;
pub const LoadModelOptions = types_mod.LoadModelOptions;
pub const CacheOptions = types_mod.CacheOptions;
pub const CacheGrowth = types_mod.CacheGrowth;

/// In-process Builder graph source (via `ctx.compile`). The graph is compiled
/// directly — no serialization round trip. Owns the graph + an arena backing the
/// per-input shape terms / value-id maps.
pub const BuilderSource = struct {
    graph: graph_mod.Graph,
    arena: std.heap.ArenaAllocator,
    input_value_ids: []const graph_mod.ValueId,
    output_value_ids: []const graph_mod.ValueId,
};

/// Where a `Model`'s concrete graphs come from. The runtime is otherwise
/// source-agnostic: it reads native metadata fields (signatures, shape terms, dim
/// exprs) and only consults the source to *instantiate* a concrete graph per shape.
pub const GraphSource = union(enum) {
    /// Loaded from an `.aion`: instantiate by walking package node records.
    package: Package,
    /// Compiled in-process: instantiate by (re)using the retained Builder graph.
    builder: BuilderSource,
};

/// Declares that input `input_index`'s axis `axis` is the free (variable) dim
/// bound to dim symbol `symbol_index`. Used to give an in-process compiled model
/// symbolic shapes so one model serves multiple input sizes (e.g. prefill then
/// decode) without recompiling from scratch.
pub const SymbolicAxis = struct {
    input_index: u32,
    axis: u32,
    symbol_index: u32,
};

pub const Model = struct {
    allocator: std.mem.Allocator,
    backend: backend_mod.Backend,
    store: *StorageManager,
    /// Execution session bound to `store`. Owns the backend's per-store state
    /// (device residency on GPU) for this model's lifetime, so weights stay
    /// device-resident across `run` calls. Released in `deinit`.
    session: backend_mod.Session,
    policy: TilePolicy,
    /// The device this model executes on (`.cpu` or a registered `.gpu`). Drives
    /// whether recurrent-state slots are migrated to device-exclusive residency
    /// (discrete GPU only — see `ensureAliasedStateSlot`); `.cpu`/unified leave
    /// state host-backed.
    device: DeviceRef,
    /// Where concrete graphs are instantiated from (loaded package vs compiled graph).
    source: GraphSource,
    initializer_tids: []TensorId,
    input_signatures: []SignatureInfo,
    output_signatures: []SignatureInfo,
    /// Native per-public-input shape terms (source-agnostic): loaded models borrow
    /// these from the package's value records; compiled models own constant terms in
    /// the builder arena. Read by `resolveBindings`/`ensureAutoInputs` so the hot path
    /// never depends on `Package`.
    input_shape_terms: []const []const package_file.ShapeTerm,
    /// Dimension-expression table for symbol resolution (loaded: package's; compiled: empty).
    dim_exprs: []const package_file.DimExpr,
    /// Number of dimension symbols (loaded: package's; compiled: 0).
    dim_symbol_count: usize,
    output_aliases: []IoAliasInfo,
    input_alias_output_indices: []u32,
    output_alias_input_indices: []u32,
    /// Lazily-allocated host mirror tensor id per output (`invalid_tensor_id`
    /// until first use). A device-exclusive recurrent-state output has no host
    /// bytes; `resolveOutputTensor` gathers it (D2H) into its mirror and returns
    /// that, so host reads stay correct while the state lives only on the device.
    /// Only a device-resident output that is actually read ever allocates one.
    output_host_mirror_tids: []TensorId,
    bound_inputs: []?Tensor,
    aliased_input_bind_versions: []u64,
    /// Persistent store tensors auto-allocated for inputs the caller never binds
    /// (zero-initialized; aliased ones carry state across runs). One slot per input;
    /// `invalid_tensor_id` until first auto-init. See `ensureAutoInputs`.
    auto_input_tids: []TensorId,
    auto_init_inputs: bool,
    /// Model-level (NOT per cache-entry) backing storage for io-aliased "state"
    /// inputs — KV caches, LSTM h/c, decode carries. Every compiled cache entry
    /// binds the SAME slot for a given aliased input, so recurrent state carries
    /// across entries with different input shapes (e.g. prefill seq=S -> decode
    /// seq=1). One slot per input; `invalid_tensor_id` for non-aliased / not-yet-built.
    aliased_state_tids: []TensorId,
    /// Optional per-input sequence-cache policy (grow-on-demand / ring), set via
    /// `setStateInputPolicy` and applied to the recurrent-state slot when it is
    /// created. `null` = fixed capacity (the slot never grows). One slot per input.
    state_input_policies: []?SequenceCachePolicy,
    /// Per-input bind-version that has already been copied into `aliased_state_tids`.
    /// Tracked at model level (not per entry) so the caller's bound tensor is seeded
    /// into the shared slot exactly once per bind, not re-copied when a new-shape
    /// entry first runs (which would clobber carried state).
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
        // Release device residency first — device buffers free through the still-alive
        // backend/device (Context must outlive its models; see Context.deinit ordering).
        self.session.deinit();
        for (self.cache_entries.items) |*entry| {
            entry.program.deinit();
            self.allocator.free(entry.symbol_values);
            self.allocator.free(entry.input_shapes);
            self.allocator.free(entry.input_slots);
            self.allocator.free(entry.direct_input_ids);
        }
        self.cache_entries.deinit(self.allocator);
        self.allocator.free(self.initializer_tids);
        self.allocator.free(self.input_signatures);
        self.allocator.free(self.output_signatures);
        self.allocator.free(self.input_shape_terms);
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
        switch (self.source) {
            .package => |*p| p.deinit(),
            .builder => |*b| {
                // Owns the Builder graph + an arena backing shape terms / value-id maps.
                b.graph.deinit();
                b.arena.deinit();
            },
        }
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

    /// The backing package, if this model was loaded (not compiled in-process).
    /// Debug-name lookups and initializer/weight-swap are package-only.
    fn packageOrNull(self: *const Self) ?*const Package {
        return switch (self.source) {
            .package => |*p| p,
            .builder => null,
        };
    }

    /// Return the debug-name table persisted in the loaded package (empty for
    /// compiled models, which have no debug-name table).
    pub fn debugNames(self: *const Self) []const package_file.DebugName {
        const p = self.packageOrNull() orelse return &[_]package_file.DebugName{};
        return p.debug_names;
    }

    /// Return the value index for a debug name, or null if not present.
    pub fn findValueByDebugName(self: *const Self, name: []const u8) ?u32 {
        const p = self.packageOrNull() orelse return null;
        for (p.debug_names) |entry| {
            if (std.mem.eql(u8, entry.name, name)) return entry.value;
        }
        return null;
    }

    /// Resolve an initializer value index to its store tensor id.
    fn initializerTidByValue(self: *Self, value_index: u32) api_errors.ApiError!TensorId {
        const p = self.packageOrNull() orelse return api_errors.ApiError.InvalidArgument;
        if (value_index >= p.values.len) return api_errors.ApiError.InvalidArgument;
        const value = p.values[value_index];
        if (value.source != .initializer) return api_errors.ApiError.InvalidArgument;
        const init_idx: u32 = value.initializer_index orelse return api_errors.ApiError.InvalidArgument;
        if (init_idx >= self.initializer_tids.len) return api_errors.ApiError.InvalidArgument;
        return self.initializer_tids[init_idx];
    }

    /// Return an owned tensor handle for an initializer value.
    ///
    /// The returned tensor can be mutated in-place to swap weights without
    /// recompiling. NOT available for a weight that an optimization pass fused into
    /// a combined tensor — that weight has no standalone in-place storage; use
    /// `overwriteInitializerByValue`, which writes through to the fused weight.
    pub fn initializerTensorByValue(self: *Self, value_index: u32) api_errors.ApiError!Tensor {
        const tid: TensorId = try self.initializerTidByValue(value_index);
        if (self.store.findDerivedWeightBySource(tid) != null) return api_errors.ApiError.InvalidArgument;
        const meta = try self.store.getConst(tid);
        return .{ .store = self.store, .id = tid, .dtype = meta.dtype, .shape = meta.shape };
    }

    /// Like `initializerTensorByValue`, but looks up the initializer via a
    /// persisted debug name.
    pub fn initializerTensorByDebugName(self: *Self, debug_name: []const u8) api_errors.ApiError!Tensor {
        const value_index: u32 = self.findValueByDebugName(debug_name) orelse return api_errors.ApiError.InvalidArgument;
        return self.initializerTensorByValue(value_index);
    }

    /// Overwrite an initializer's bytes (copy) without changing tensor IDs.
    ///
    /// This is the safest weight-swap primitive: it does not require retargeting
    /// compiled programs, and it can copy between different (but compatible)
    /// tilings via pack/unpack when needed.
    pub fn overwriteInitializerByValue(self: *Self, value_index: u32, src: Tensor) api_errors.ApiError!void {
        if (src.store != self.store) return api_errors.ApiError.InvalidArgument;
        const tid: TensorId = try self.initializerTidByValue(value_index);

        // If this weight was fused into a combined tensor, the fused tensor is the
        // canonical store — write through to its sub-region (the original buffer may
        // have been reclaimed). Takes effect next run with no recompile.
        if (self.store.findDerivedWeightBySource(tid)) |ref| {
            return fuse.overwriteFusedColumns(self.allocator, self.store, ref, src.id) catch api_errors.ApiError.InvalidArgument;
        }

        if (tid == src.id) return;
        const meta = try self.store.getConst(tid);
        const dst: Tensor = .{ .store = self.store, .id = tid, .dtype = meta.dtype, .shape = meta.shape };
        try dst.copyFrom(self.allocator, src);
    }

    pub fn overwriteInitializerByDebugName(self: *Self, debug_name: []const u8, src: Tensor) api_errors.ApiError!void {
        const value_index: u32 = self.findValueByDebugName(debug_name) orelse return api_errors.ApiError.InvalidArgument;
        return self.overwriteInitializerByValue(value_index, src);
    }

    /// Read an initializer's current bytes into `dst` (the read counterpart of
    /// `overwriteInitializerByValue`). Works whether or not the weight was fused: a
    /// fused weight has no standalone storage, so its value is materialized out of
    /// the combined tensor — mirrors `_packed_params` unpacking on read. `dst` must
    /// match the logical weight's dtype/shape and is owned by the caller.
    pub fn readInitializerByValue(self: *Self, value_index: u32, dst: Tensor) api_errors.ApiError!void {
        if (dst.store != self.store) return api_errors.ApiError.InvalidArgument;
        const tid: TensorId = try self.initializerTidByValue(value_index);

        if (self.store.findDerivedWeightBySource(tid)) |ref| {
            return fuse.readFusedColumns(self.allocator, self.store, ref, dst.id) catch api_errors.ApiError.InvalidArgument;
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

    /// Retarget an initializer to a different underlying tensor id.
    ///
    /// This updates:
    /// - the initializer tensor-id mapping used for future cache entries
    /// - all existing cached compiled programs (in-place patch of tensor ids)
    ///
    /// Requirements:
    /// - `new_tensor` must belong to the same `Context` storage manager
    /// - layout must be compatible (dtype, shape, tile layout)
    pub fn retargetInitializerByValue(self: *Self, value_index: u32, new_tensor: Tensor) api_errors.ApiError!void {
        if (new_tensor.store != self.store) return api_errors.ApiError.InvalidArgument;
        const p = self.packageOrNull() orelse return api_errors.ApiError.InvalidArgument;
        if (value_index >= p.values.len) return api_errors.ApiError.InvalidArgument;
        const value = p.values[value_index];
        if (value.source != .initializer) return api_errors.ApiError.InvalidArgument;
        const init_idx: u32 = value.initializer_index orelse return api_errors.ApiError.InvalidArgument;
        if (init_idx >= self.initializer_tids.len) return api_errors.ApiError.InvalidArgument;

        const old_tid: TensorId = self.initializer_tids[init_idx];
        const new_tid: TensorId = new_tensor.id;
        if (old_tid == new_tid) return;

        // A fused weight isn't referenced by the program (the combined tensor is), so
        // retargeting its id would patch nothing. Write the new contents through to
        // the fused weight's sub-region instead; the alias mapping stays put.
        if (self.store.findDerivedWeightBySource(old_tid)) |ref| {
            return fuse.overwriteFusedColumns(self.allocator, self.store, ref, new_tid) catch api_errors.ApiError.InvalidArgument;
        }

        const ok = try signatures.tensorsHaveCompatibleLayout(self.store, old_tid, new_tid);
        if (!ok) return api_errors.ApiError.InvalidArgument;

        self.initializer_tids[init_idx] = new_tid;
        for (self.cache_entries.items) |*entry| {
            retarget.retargetProgramTensorIds(&entry.program, old_tid, new_tid);
        }
    }

    pub fn retargetInitializerByDebugName(self: *Self, debug_name: []const u8, new_tensor: Tensor) api_errors.ApiError!void {
        const value_index: u32 = self.findValueByDebugName(debug_name) orelse return api_errors.ApiError.InvalidArgument;
        return self.retargetInitializerByValue(value_index, new_tensor);
    }

    pub fn cacheEntryCount(self: *const Self) usize {
        return self.cache_entries.items.len;
    }

    /// Introspection (debug/tests): whether the named io-aliased input's recurrent
    /// state slot currently lives in device-exclusive memory (migrated off the
    /// host — no staging duplicate) rather than host-backed/staged. False for a
    /// non-aliased input, before the slot is first allocated (first `run`), and on
    /// CPU. See `migrateStateSlotToDevice`.
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

    /// Reset recurrent state: zero every io-aliased "state" slot (KV caches, LSTM
    /// h/c, decode state). The slots are shared across all compiled cache entries,
    /// so one pass clears the state seen by every entry. Use between independent
    /// sequences (e.g. utterances) without reloading the model. A no-op before the
    /// first `run()` (no slots allocated yet).
    /// Declare a sequence-cache policy (grow-on-demand or ring) for an io-aliased
    /// recurrent-state input, by name. Applied to the runtime-owned state slot at
    /// creation (or immediately if it already exists). Growable state may start
    /// small and grow on demand up to `GrowablePolicy.max_capacity_tokens` — the
    /// runtime owns and resizes the slot, device-resident growth included, so the
    /// caller need not pre-allocate the maximum. Call before the first `run()` for
    /// the policy to shape the initial slot. Generic across any recurrent state
    /// (LLM KV, streaming buffers, RNN history), not LLM-specific.
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
                    if (src.id != dst.id) self.copyIntoStateSlot(dst, src) catch |e| {
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
            if (src.id != dst.id) dst.copyFrom(self.allocator, src) catch |e| {
                if (trace) {
                    std.debug.print(
                        "[aion][run] copyFrom failed for input {s} (src_id={d} dst_id={d}): {s}\n",
                        .{ self.input_signatures[i].name, src.id, dst.id, @errorName(e) },
                    );
                }
                return e;
            };
        }

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
            // Non-in-place recurrent outputs (LSTM h/c, convolution state, etc.)
            // must be copied back into their state slot.  The slot may be
            // device-exclusive, so use the residency-aware path; in-place aliases
            // were skipped above and incur no copy at all.
            self.copyIntoStateSlot(dst, src) catch |e| {
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

    /// Resolve output `output_index` of cache entry `cache_index` to a Tensor
    /// handle whose host bytes are current. Recurrent-state outputs `execute`
    /// leaves device-resident (see `ExecutableProgram.output_device_resident`)
    /// take one of two on-demand paths so a host read still sees correct data:
    ///   - staged (host-backed, discrete-GPU cache): flush the device copy to the
    ///     host store (`syncToHost`) and return the tensor itself;
    ///   - device-exclusive (migrated via `moveTensor`, no host bytes): gather it
    ///     (D2H) into a host mirror and return the mirror.
    /// Decode never reads its own carried state, so both stay off the hot path;
    /// everything else (genuine outputs, CPU) returns the tensor directly.
    fn resolveOutputTensor(self: *Self, cache_index: usize, output_index: usize) api_errors.ApiError!Tensor {
        const entry: *const CacheEntry = &self.cache_entries.items[cache_index];
        const tid = entry.program.outputs[output_index];
        if (entry.program.outputStaysResident(output_index)) {
            const on_device = (self.store.tensorDevice(tid) catch DeviceRef{}).kind != .cpu;
            if (on_device) return self.mirrorOutputToHost(output_index, tid);
            try self.session.syncToHost(tid);
        }
        const t = try self.store.getConst(tid);
        return .{ .store = self.store, .id = tid, .dtype = t.dtype, .shape = t.shape };
    }

    /// Materialize a device-exclusive output into its host mirror (D2H gather) and
    /// return a handle to the mirror. The mirror is created lazily on first read
    /// (matching the source dtype/shape, single-tile host) and reused across
    /// reads, re-created if the source's shape changed (a grown cache). This keeps
    /// device-exclusive state host-readable without a permanent host duplicate in
    /// the hot path — decode never calls this.
    fn mirrorOutputToHost(self: *Self, output_index: usize, src_tid: TensorId) api_errors.ApiError!Tensor {
        const src_meta = try self.store.getConst(src_tid);
        const need_new = blk: {
            const cur = self.output_host_mirror_tids[output_index];
            if (cur == types_mod.invalid_tensor_id) break :blk true;
            const m = self.store.getConst(cur) catch break :blk true;
            break :blk m.dtype != src_meta.dtype or !signatures.sameUsize(m.shape, src_meta.shape);
        };
        if (need_new) {
            self.output_host_mirror_tids[output_index] =
                try initializers.createTensorSingleTile(self.store, self.policy, src_meta.dtype, src_meta.shape);
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

    /// Shared constructor: allocates the runtime bookkeeping arrays + io-alias index
    /// maps from already-extracted, source-agnostic metadata. Both `initLoaded` and
    /// `initCompiled` populate the metadata from their source, then delegate here.
    /// On success the returned `Model` owns `source`, `initializer_tids`,
    /// `input_signatures`, `output_signatures`, and the outer `input_shape_terms`
    /// array; on error the caller's `errdefer`s free them.
    fn initCommon(
        allocator: std.mem.Allocator,
        backend: backend_mod.Backend,
        store: *StorageManager,
        policy: TilePolicy,
        device: DeviceRef,
        source: GraphSource,
        initializer_tids: []TensorId,
        input_signatures: []SignatureInfo,
        output_signatures: []SignatureInfo,
        input_shape_terms: []const []const package_file.ShapeTerm,
        dim_exprs: []const package_file.DimExpr,
        dim_symbol_count: usize,
        io_aliases: []const package_file.IoAlias,
        input_roles_src: []const package_file.InputRole,
        package_hash: u64,
        opts: LoadModelOptions,
    ) api_errors.LoadError!Self {
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
                    // Grow-on-demand needs the runtime-supported layout (rank-4,
                    // sequence axis 2 — see `stateSlotCompatible`/`mapSequenceStep`)
                    // and the package's blessing; otherwise fall back to a full
                    // pre-allocation rather than failing the load.
                    const growable: ?types_mod.CacheGrowth = opts.cache.growable;
                    if (growable != null and
                        (role.flags & package_file.InputRoleFlags.allow_growable) != 0 and
                        input_signatures[i].rank == 4 and role.axis == 2)
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
            .policy = policy,
            .device = device,
            .source = source,
            .initializer_tids = initializer_tids,
            .input_signatures = input_signatures,
            .output_signatures = output_signatures,
            .input_shape_terms = input_shape_terms,
            .dim_exprs = dim_exprs,
            .dim_symbol_count = dim_symbol_count,
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
            .package_hash = package_hash,
            .trace_runs = traceEnabled(),
        };
    }

    /// Build a `Model` from a parsed `.aion` package. Metadata (signatures, shape
    /// terms, dim exprs) is taken from the package; the package is retained as the
    /// graph source for per-shape instantiation + weight-swap-by-debug-name.
    pub fn initLoaded(
        allocator: std.mem.Allocator,
        backend: backend_mod.Backend,
        store: *StorageManager,
        policy: TilePolicy,
        device: DeviceRef,
        package: Package,
        initializer_tids: []TensorId,
        package_hash: u64,
        opts: LoadModelOptions,
    ) api_errors.LoadError!Self {
        const input_signatures = try signatures.buildSignatures(allocator, &package, package.inputs);
        errdefer allocator.free(input_signatures);
        const output_signatures = try signatures.buildSignatures(allocator, &package, package.outputs);
        errdefer allocator.free(output_signatures);

        // Borrow each public input's shape terms from its value record (zero-copy;
        // the package — retained in `source` — keeps them alive).
        const input_shape_terms = try allocator.alloc([]const package_file.ShapeTerm, input_signatures.len);
        errdefer allocator.free(input_shape_terms);
        for (input_signatures, 0..) |sig, i| input_shape_terms[i] = package.values[sig.value].shape_terms;

        return initCommon(
            allocator,
            backend,
            store,
            policy,
            device,
            .{ .package = package },
            initializer_tids,
            input_signatures,
            output_signatures,
            input_shape_terms,
            package.dim_exprs,
            package.dim_symbols.len,
            package.io_aliases,
            package.input_roles,
            package_hash,
            opts,
        );
    }

    /// Build a `Model` from an in-process Builder graph (via `ctx.compile`).
    ///
    /// Takes ownership of `graph_in`. Metadata (signatures, constant shape terms) is
    /// derived natively from the graph + names — no synthetic package, no
    /// serialization round trip. The program is compiled directly from the retained
    /// graph by the `builder` source's `instantiate`. `graph_in` must already have had
    /// shape inference run (concrete shapes). Inputs are the graph's *public* inputs
    /// (params are baked initializers already bound in the graph).
    pub fn initCompiled(
        allocator: std.mem.Allocator,
        backend: backend_mod.Backend,
        store: *StorageManager,
        policy: TilePolicy,
        device: DeviceRef,
        graph_in: graph_mod.Graph,
        input_value_ids: []const graph_mod.ValueId,
        input_names: []const []const u8,
        output_value_ids: []const graph_mod.ValueId,
        output_names: []const []const u8,
        io_aliases: []const package_file.IoAlias,
        input_roles: []const package_file.InputRole,
        dim_symbol_count: usize,
        symbolic_axes: []const SymbolicAxis,
        opts: LoadModelOptions,
    ) api_errors.LoadError!Self {
        var graph = graph_in;
        errdefer graph.deinit();

        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();
        const aa = arena.allocator();

        const n_in = input_value_ids.len;
        const n_out = output_value_ids.len;

        const input_signatures = try allocator.alloc(SignatureInfo, n_in);
        errdefer allocator.free(input_signatures);
        const output_signatures = try allocator.alloc(SignatureInfo, n_out);
        errdefer allocator.free(output_signatures);
        const input_shape_terms = try allocator.alloc([]const package_file.ShapeTerm, n_in);
        errdefer allocator.free(input_shape_terms);

        for (input_value_ids, 0..) |vid, k| {
            const v = graph.values.items[@intCast(vid)];
            const dtype = v.dtype orelse return error.InvalidArgument;
            const terms = try aa.alloc(package_file.ShapeTerm, v.shape.len);
            for (v.shape, 0..) |d, i| terms[i] = .{ .constant = @intCast(d) };
            // Mark declared-symbolic axes as expressions so a differently-sized
            // bound input resolves the symbol instead of being rejected.
            for (symbolic_axes) |sa| {
                if (sa.input_index == k) {
                    if (sa.axis >= terms.len or sa.symbol_index >= dim_symbol_count) return error.InvalidArgument;
                    terms[sa.axis] = .{ .expr = sa.symbol_index };
                }
            }
            input_signatures[k] = .{ .name = try aa.dupe(u8, input_names[k]), .value = vid, .dtype = dtype, .rank = @intCast(v.shape.len) };
            input_shape_terms[k] = terms;
        }

        // One dim-expr per symbol (`expr_index == symbol_index`), mirroring the
        // package-export encoding so `bindInputDimExprs` resolves them uniformly.
        const dim_exprs = try aa.alloc(package_file.DimExpr, dim_symbol_count);
        for (0..dim_symbol_count) |i| dim_exprs[i] = .{ .symbol = @intCast(i) };
        for (output_value_ids, 0..) |vid, j| {
            const v = graph.values.items[@intCast(vid)];
            const dtype = v.dtype orelse return error.InvalidArgument;
            output_signatures[j] = .{ .name = try aa.dupe(u8, output_names[j]), .value = vid, .dtype = dtype, .rank = @intCast(v.shape.len) };
        }

        // IMPORTANT: complete ALL arena allocations *before* copying `arena` into
        // `source`. `ArenaAllocator` is a value type; copying it snapshots its buffer
        // state, so allocations made via `aa` after the copy would not be tracked by
        // the stored arena and would leak.
        const in_ids_dup = try aa.dupe(graph_mod.ValueId, input_value_ids);
        const out_ids_dup = try aa.dupe(graph_mod.ValueId, output_value_ids);
        const aliases_dup = try aa.dupe(package_file.IoAlias, io_aliases);
        const roles_dup = try aa.dupe(package_file.InputRole, input_roles);

        const empty_tids = try allocator.alloc(TensorId, 0);
        errdefer allocator.free(empty_tids);

        const source = GraphSource{ .builder = .{
            .graph = graph,
            .arena = arena,
            .input_value_ids = in_ids_dup,
            .output_value_ids = out_ids_dup,
        } };

        return initCommon(
            allocator,
            backend,
            store,
            policy,
            device,
            source,
            empty_tids,
            input_signatures,
            output_signatures,
            input_shape_terms,
            dim_exprs,
            dim_symbol_count,
            aliases_dup,
            roles_dup,
            0,
            opts,
        );
    }

    fn resolveBindings(self: *Self) error{InvalidArgument}!ResolvedInputs {
        @memset(self.run_symbol_bindings, null);
        @memset(self.run_direct_input_ids, types_mod.invalid_tensor_id);

        var shape_cursor: usize = 0;
        var i: usize = 0;
        while (i < self.input_signatures.len) : (i += 1) {
            const tensor = self.bound_inputs[i] orelse return error.InvalidArgument;
            const sig = self.input_signatures[i];
            const terms = self.input_shape_terms[i];
            if (tensor.dtype != sig.dtype or tensor.shape.len != sig.rank) return error.InvalidArgument;

            var d: usize = 0;
            while (d < tensor.shape.len) : (d += 1) {
                const actual: u64 = @intCast(tensor.shape[d]);
                try signatures.bindInputDimExprs(self.dim_exprs, terms[d], actual, self.run_symbol_bindings);
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
            const terms = self.input_shape_terms[i];
            if (tensor.dtype != sig.dtype or tensor.shape.len != sig.rank) return error.InvalidArgument;
            var d: usize = 0;
            while (d < tensor.shape.len) : (d += 1) {
                try signatures.bindInputDimExprs(self.dim_exprs, terms[d], @intCast(tensor.shape[d]), self.run_symbol_bindings);
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
            tid = initializers.createTensorForShape(self.store, self.policy, .i32, shape) catch return error.OutOfMemory;
            self.role_input_tids[index] = tid;
        }
        const meta = self.store.getConst(tid) catch return error.InvalidArgument;
        const t = Tensor{ .store = self.store, .id = tid, .dtype = .i32, .shape = meta.shape };
        t.write(values) catch return error.InvalidArgument;
        self.bound_inputs[index] = t;
        self.role_auto_bound[index] = true;
    }

    /// Feed role-declared control inputs from the auto-tracked sequence position:
    /// cache_write_index = position, cache_visible_end = position + new_tokens,
    /// positions = [position .. position + new_tokens) per batch row. `new_tokens`
    /// is read from the bound tokens-role input's sequence axis each run, so the
    /// prefill (seq=S) -> decode (seq=1) switch needs no caller bookkeeping.
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

    /// Bind any input the caller left unbound to a persistent, zero-initialized slot
    /// whose shape is inferred from the symbols contributed by the bound inputs.
    ///
    /// Aliased (recurrent) inputs are seeded once and then carry their contents
    /// across runs via the output-alias sync at the end of `run()`; non-aliased
    /// unbound inputs simply stay zero. When `auto_init_inputs` is disabled this is
    /// strict: the first unbound input is an error.
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
        const optional_symbols = instantiate.optionalizeSymbols(self.allocator, symbol_values) catch return error.OutOfMemory;
        defer self.allocator.free(optional_symbols);

        // Symbols no bound input determines fall back to their load-time default
        // (role-declared cache capacity from `LoadModelOptions.cache`), letting the
        // runtime auto-allocate caches with a free capacity axis.
        for (optional_symbols, 0..) |*sym, k| {
            if (sym.* == null) sym.* = self.symbol_defaults[k];
        }

        for (self.input_signatures, 0..) |sig, i| {
            if (self.bound_inputs[i] != null) continue;

            const shape = package_file.resolveShapeTermsExprs(self.allocator, self.dim_exprs, self.input_shape_terms[i], optional_symbols) catch {
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
                    try initializers.createTensorSingleTile(self.store, self.policy, sig.dtype, shape)
                else
                    try initializers.createTensorForShape(self.store, self.policy, sig.dtype, shape);
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
                if (try self.prepareCacheEntryInputs(entry, direct_input_ids)) return idx;
            }
        }

        const entry = try self.buildCacheEntry(symbol_values, input_shapes, direct_input_ids);
        try self.cache_entries.append(self.allocator, entry);
        self.reclaimFusedInitializers();
        return self.cache_entries.items.len - 1;
    }

    /// Reclaim weights that an optimization pass fused into a combined tensor: their
    /// data is now redundant with the fused weight (the canonical store), so free
    /// the backing buffer while keeping metadata. Lifecycle lives here, not in the
    /// pass. A weight is freed only if no compiled program still references it
    /// directly (guards a weight shared with an unfused op). Idempotent; weight-swap
    /// stays correct via write-through to the fused weight.
    fn reclaimFusedInitializers(self: *Self) void {
        for (self.initializer_tids) |tid| {
            if (self.store.findDerivedWeightBySource(tid) == null) continue;
            var referenced = false;
            for (self.cache_entries.items) |*entry| {
                if (retarget.programReferencesTensorId(&entry.program, tid)) {
                    referenced = true;
                    break;
                }
            }
            if (!referenced) self.store.releaseTensorData(tid) catch {};
        }
    }

    /// Produce a concrete graph for the requested shapes (source-specific), then
    /// bind public-input slots + compile via the shared `finishCacheEntry`.
    fn buildCacheEntry(
        self: *Self,
        symbol_values: []const u64,
        input_shapes: []const usize,
        direct_input_ids: []const TensorId,
    ) api_errors.ExecuteError!CacheEntry {
        switch (self.source) {
            // Compiled model: clone the pristine authored template into a fresh
            // concrete graph for this run's shapes, then compile it (infer +
            // optimize + lower) — mirroring the loaded path below. The template is
            // never mutated; the clone shares weight tensors (external ids copied
            // by value) and is discarded once its program is built.
            .builder => |*b| {
                var g = b.graph.clone(self.allocator) catch return error.OutOfMemory;
                defer g.deinit();
                // A freshly-materialized graph: produced shapes are (re)inferred, so
                // clear them and set each public input to this run's concrete shape.
                for (g.values.items) |*v| {
                    if (v.producer != null) v.shape = &[_]usize{};
                }
                var cursor: usize = 0;
                for (self.input_signatures, 0..) |_, si| {
                    const cshape = buildConcreteShapeFromFlat(input_shapes, self.input_signatures, si, &cursor);
                    g.values.items[@intCast(b.input_value_ids[si])].shape = g.dupeShape(cshape) catch return error.OutOfMemory;
                }
                return self.finishCacheEntry(&g, b.input_value_ids, b.output_value_ids, symbol_values, input_shapes, direct_input_ids);
            },
            // Loaded model: walk the package node records to build a fresh concrete
            // graph specialized to these shapes/symbols.
            .package => |*pkg| {
                var graph = graph_mod.Graph.init(self.allocator);
                defer graph.deinit();

                var value_map = try self.allocator.alloc(graph_mod.ValueId, pkg.values.len);
                defer self.allocator.free(value_map);
                @memset(value_map, std.math.maxInt(graph_mod.ValueId));

                const in_ids = try self.allocator.alloc(graph_mod.ValueId, self.input_signatures.len);
                defer self.allocator.free(in_ids);
                const out_ids = try self.allocator.alloc(graph_mod.ValueId, self.output_signatures.len);
                defer self.allocator.free(out_ids);

                // Public inputs: add (unbound — slot binding happens in finishCacheEntry).
                var input_shape_cursor: usize = 0;
                for (self.input_signatures, 0..) |sig, sig_idx| {
                    const concrete_shape = buildConcreteShapeFromFlat(input_shapes, self.input_signatures, sig_idx, &input_shape_cursor);
                    const vid = try graph.addInput(sig.dtype, concrete_shape);
                    value_map[sig.value] = vid;
                    in_ids[sig_idx] = vid;
                }

                const optional_symbols = try instantiate.optionalizeSymbols(self.allocator, symbol_values);
                defer self.allocator.free(optional_symbols);

                // Initializers (weights): bound to their store tensors.
                for (pkg.values, 0..) |value, idx| {
                    if (value.source != .initializer) continue;
                    const shape = try package_file.resolveShapeTerms(self.allocator, pkg, value.shape_terms, optional_symbols);
                    defer self.allocator.free(shape);
                    const vid = try graph.addInput(value.dtype, shape);
                    try graph.bindExternal(vid, self.initializer_tids[value.initializer_index.?]);
                    value_map[idx] = vid;
                }

                // Regions are built lazily, just before the main `If`/`Loop` node that
                // references them, so a region body can reference values produced by
                // earlier main nodes (e.g. an in-graph decode Loop reading the encoder out).
                const region_built = try self.allocator.alloc(bool, pkg.regions.len);
                defer self.allocator.free(region_built);
                @memset(region_built, false);
                const region_map = try self.allocator.alloc(graph_mod.RegionId, pkg.regions.len);
                defer self.allocator.free(region_map);

                for (pkg.nodes) |node| {
                    switch (node.op) {
                        .If => |iff| {
                            try ensureRegionBuilt(pkg, symbol_values, &graph, value_map, region_built, region_map, @intCast(iff.then_region), self.allocator);
                            try ensureRegionBuilt(pkg, symbol_values, &graph, value_map, region_built, region_map, @intCast(iff.else_region), self.allocator);
                        },
                        .Loop => |lp| try ensureRegionBuilt(pkg, symbol_values, &graph, value_map, region_built, region_map, @intCast(lp.body_region), self.allocator),
                        else => {},
                    }

                    const mapped_inputs = try self.allocator.alloc(graph_mod.ValueId, node.inputs.len);
                    defer self.allocator.free(mapped_inputs);
                    for (node.inputs, 0..) |input, i| {
                        if (input >= value_map.len) return error.InvalidArgument;
                        mapped_inputs[i] = value_map[input];
                    }
                    const extra_ids = package_file.nodeExtraOutputs(node);
                    const extra_buf = try self.allocator.alloc(graph_mod.ValueId, extra_ids.len);
                    defer self.allocator.free(extra_buf);
                    const out_vid = try instantiate.instantiateNode(self.allocator, pkg, symbol_values, &graph, node, mapped_inputs, region_map, extra_buf);
                    value_map[node.output] = out_vid;
                    for (extra_ids, 0..) |eid, i| {
                        if (eid >= value_map.len) return error.InvalidArgument;
                        value_map[eid] = extra_buf[i];
                    }
                }

                for (pkg.outputs, 0..) |sig, idx| out_ids[idx] = value_map[sig.value];

                return self.finishCacheEntry(&graph, in_ids, out_ids, symbol_values, input_shapes, direct_input_ids);
            },
        }
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

        var input_shape_cursor: usize = 0;
        for (self.input_signatures, 0..) |_, sig_idx| {
            const concrete_shape = buildConcreteShapeFromFlat(input_shapes, self.input_signatures, sig_idx, &input_shape_cursor);
            const slot_tid = if (self.inputAliasOutputIndex(sig_idx) != null)
                // Aliased inputs (KV caches, recurrent state) use a model-level slot
                // shared by every cache entry, so state carries across entries with
                // different input shapes (prefill vs decode). Single-tile (SequenceAppend
                // needs a contiguous last axis; the default tiler would split it).
                try self.ensureAliasedStateSlot(sig_idx, concrete_shape)
            else blk: {
                const tid = direct_input_ids[sig_idx];
                if (tid == types_mod.invalid_tensor_id) return error.InvalidArgument;
                break :blk tid;
            };
            input_slots[sig_idx] = slot_tid;
            try graph.bindExternal(in_ids[sig_idx], slot_tid);
        }

        try graph.setOutputs(out_ids);

        var program = try program_mod.compileGraph(self.allocator, graph, self.store, self.policy);
        errdefer program.deinit();

        try self.markDeviceResidentOutputs(&program, input_slots);

        return .{
            .symbol_values = try self.allocator.dupe(u64, symbol_values),
            .input_shapes = try self.allocator.dupe(usize, input_shapes),
            .input_slots = input_slots,
            .direct_input_ids = entry_direct_input_ids,
            .program = program,
        };
    }

    /// Flag the program outputs that are recurrent state aliased *in place* to an
    /// input, so a device backend keeps them resident instead of flushing them to
    /// the host each run (see `ExecutableProgram.output_device_resident`). "In
    /// place" is exactly the condition under which `run()`'s io-alias write-back is
    /// a no-op: the output's compiled tensor id equals the input slot it feeds back
    /// into (e.g. SequenceAppend, whose output aliases the cache storage). Non-in-
    /// place aliases still need the flush — their host write-back reads the output
    /// bytes — so they are deliberately left flushable. No allocation when nothing
    /// qualifies, keeping non-recurrent models allocation-free.
    fn markDeviceResidentOutputs(self: *Self, program: *program_mod.Program, input_slots: []const TensorId) api_errors.ExecuteError!void {
        var any = false;
        for (self.output_aliases) |alias| {
            if (program.outputs[alias.output_index] == input_slots[alias.input_index]) {
                any = true;
                break;
            }
        }
        if (!any) return;

        const mask = self.allocator.alloc(bool, program.outputs.len) catch return error.OutOfMemory;
        @memset(mask, false);
        for (self.output_aliases) |alias| {
            if (program.outputs[alias.output_index] == input_slots[alias.input_index]) mask[alias.output_index] = true;
        }
        program.output_device_resident = mask;
    }

    /// Return the model-level backing slot for io-aliased input `input_index`,
    /// allocating a zeroed single-tile tensor on first use. If a slot already
    /// exists with a different shape (a capacity change), it is replaced and the
    /// seed must be re-copied, so the synced version is reset.
    fn ensureAliasedStateSlot(self: *Self, input_index: usize, shape: []const usize) api_errors.ExecuteError!TensorId {
        const sig = self.input_signatures[input_index];
        const growable = self.stateInputGrowable(input_index);
        const cur = self.aliased_state_tids[input_index];
        if (cur != types_mod.invalid_tensor_id) {
            const meta = self.store.getConst(cur) catch return error.InvalidArgument;
            // Reuse the existing slot when it can still back this shape — including a
            // growable slot that has grown past the declared capacity (so state built
            // up in an earlier entry, e.g. prefill, isn't discarded when a later
            // decode entry re-requests the small initial capacity).
            if (stateSlotCompatible(meta.shape, shape, growable)) return cur;
            // Capacity truly changed: re-seed into the new slot on the next run.
            self.aliased_state_synced_versions[input_index] = 0;
        }
        const tid = try initializers.createTensorSingleTile(self.store, self.policy, sig.dtype, shape);
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

    /// Whether a recurrent-state slot of shape `have` can back a request for
    /// `want`. Exact match always qualifies; a growable rank-4 cache also qualifies
    /// when only the sequence axis (axis 2) has grown larger (`have[2] >= want[2]`),
    /// which is how a slot grown in one cache entry carries into another whose
    /// declared capacity is still the small initial value.
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

    /// Migrate a freshly created (host, zeroed) recurrent-state slot to
    /// device-exclusive residency when the model runs on a discrete GPU and the
    /// slot fits the device's single-buffer ceiling. This drops the host copy —
    /// the KV cache then lives ONLY in device memory (`moveTensor`/`deviceTile`),
    /// so decode kernels hit the residency fast path with no staging duplicate.
    ///
    /// A deliberate no-op when: on CPU; on a (future) unified backend that imports
    /// host memory zero-copy (migration would gain nothing); or the slot exceeds
    /// the device's per-buffer ceiling. Any of these keeps the staged/host path,
    /// which stays correct. Growable slots ARE migrated — device-resident growth is
    /// handled by `StorageManager.ensureTensorAxisCapacity` (round-trip re-migrate).
    /// Migration failure falls back to staged: `moveTensor` leaves the source
    /// tensor untouched on error.
    fn migrateStateSlotToDevice(self: *Self, tid: TensorId, dtype: DType, shape: []const usize) void {
        if (self.device.kind == .cpu) return;
        const mem = self.store.deviceMemoryFor(self.device) orelse return;
        if (mem.model() == .unified) return;

        const info = dtype.info();
        if (info.is_quantized) return; // createTensorSingleTile already rejects these
        var elems: u64 = 1;
        for (shape) |d| elems *= @as(u64, @intCast(d));
        const bytes: u64 = elems * @as(u64, @intCast(info.block_bytes));
        if (bytes > mem.maxBindingBytes()) return; // over the device's per-buffer ceiling

        const policy = self.store.policyFor(self.device);
        // Single-tile target (tile_shape == shape), matching createTensorSingleTile.
        self.store.moveTensor(tid, self.device, mem, shape, policy.tile_alignment) catch return;
    }

    /// Seed recurrent-state slot `dst` from a caller-bound source, honoring the
    /// slot's residency: a device-exclusive slot is written through the store's
    /// device-aware copy (H2D scatter), a host slot via the fast in-place copy.
    fn copyIntoStateSlot(self: *Self, dst: Tensor, src: Tensor) api_errors.ExecuteError!void {
        const on_device = (self.store.tensorDevice(dst.id) catch DeviceRef{}).kind != .cpu;
        if (on_device) return self.store.copyTensorData(dst.id, src.id);
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
        if (self.dim_symbol_count != 0) return null;
        for (self.cache_entries.items, 0..) |*entry, idx| {
            if (try self.prepareCacheEntryInputs(entry, desired_direct_input_ids)) return idx;
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
            if (!try signatures.tensorsHaveCompatibleLayout(self.store, current_tid, desired_tid)) return false;
            retarget.retargetProgramTensorIds(&entry.program, current_tid, desired_tid);
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
    program: program_mod.Program,
};

/// Build one control-flow region's body into the graph, lazily. Sub-regions
/// referenced by `If`/`Loop` nodes inside the body are built first (recursively),
/// so no nested active region is needed. Region nodes may reference any value
/// already in `value_map` — including outputs of earlier main nodes (e.g. the
/// encoder output an in-graph decode Loop reads via GatherRows).
fn ensureRegionBuilt(
    package: *const Package,
    symbol_values: []const u64,
    graph: *graph_mod.Graph,
    value_map: []graph_mod.ValueId,
    region_built: []bool,
    region_map: []graph_mod.RegionId,
    region_idx: usize,
    allocator: std.mem.Allocator,
) api_errors.ExecuteError!void {
    if (region_idx >= package.regions.len) return error.InvalidArgument;
    if (region_built[region_idx]) return;
    const region = package.regions[region_idx];

    // Pre-build sub-regions (so they are complete before this region's beginRegion).
    for (region.nodes) |n| {
        switch (n.op) {
            .If => |iff| {
                try ensureRegionBuilt(package, symbol_values, graph, value_map, region_built, region_map, @intCast(iff.then_region), allocator);
                try ensureRegionBuilt(package, symbol_values, graph, value_map, region_built, region_map, @intCast(iff.else_region), allocator);
            },
            .Loop => |lp| try ensureRegionBuilt(package, symbol_values, graph, value_map, region_built, region_map, @intCast(lp.body_region), allocator),
            else => {},
        }
    }

    try graph.beginRegion();
    errdefer {
        if (graph.active_region) graph.active_region = false;
    }
    for (region.nodes) |node| {
        const mapped_inputs = try allocator.alloc(graph_mod.ValueId, node.inputs.len);
        defer allocator.free(mapped_inputs);
        for (node.inputs, 0..) |input, i| {
            if (input >= value_map.len) return error.InvalidArgument;
            mapped_inputs[i] = value_map[input];
        }
        const extra_ids = package_file.nodeExtraOutputs(node);
        const extra_buf = try allocator.alloc(graph_mod.ValueId, extra_ids.len);
        defer allocator.free(extra_buf);
        const out_vid = try instantiate.instantiateNode(allocator, package, symbol_values, graph, node, mapped_inputs, region_map, extra_buf);
        value_map[node.output] = out_vid;
        for (extra_ids, 0..) |eid, i| {
            if (eid >= value_map.len) return error.InvalidArgument;
            value_map[eid] = extra_buf[i];
        }
    }
    const mapped_outputs = try allocator.alloc(graph_mod.ValueId, region.outputs.len);
    defer allocator.free(mapped_outputs);
    for (region.outputs, 0..) |output, i| {
        if (output >= value_map.len) return error.InvalidArgument;
        mapped_outputs[i] = value_map[output];
    }
    region_map[region_idx] = try graph.endRegion(mapped_outputs);
    region_built[region_idx] = true;
}

pub fn importInitializersForLoadedModel(
    allocator: std.mem.Allocator,
    store: *StorageManager,
    policy: TilePolicy,
    package: *const Package,
) api_errors.LoadError![]TensorId {
    return initializers.importInitializersForLoadedModel(allocator, store, policy, package);
}

pub fn importInitializersStreaming(
    allocator: std.mem.Allocator,
    store: *StorageManager,
    policy: TilePolicy,
    package: *Package,
    file: std.Io.File,
    source_bytes: []const u8,
) api_errors.LoadError![]TensorId {
    return initializers.importInitializersStreaming(allocator, store, policy, package, file, source_bytes);
}

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
