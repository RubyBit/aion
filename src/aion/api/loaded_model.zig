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
pub const DType = types_mod.DType;
pub const TilePolicy = types_mod.TilePolicy;
pub const Package = types_mod.Package;
pub const SignatureInfo = types_mod.SignatureInfo;
pub const IoAliasInfo = types_mod.IoAliasInfo;
pub const LoadModelOptions = types_mod.LoadModelOptions;

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

pub const Model = struct {
    allocator: std.mem.Allocator,
    backend: backend_mod.Backend,
    store: *StorageManager,
    /// Execution session bound to `store`. Owns the backend's per-store state
    /// (device residency on GPU) for this model's lifetime, so weights stay
    /// device-resident across `run` calls. Released in `deinit`.
    session: backend_mod.Session,
    policy: TilePolicy,
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
    /// Per-input bind-version that has already been copied into `aliased_state_tids`.
    /// Tracked at model level (not per entry) so the caller's bound tensor is seeded
    /// into the shared slot exactly once per bind, not re-copied when a new-shape
    /// entry first runs (which would clobber carried state).
    aliased_state_synced_versions: []u64,
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
        self.allocator.free(self.bound_inputs);
        self.allocator.free(self.aliased_input_bind_versions);
        self.allocator.free(self.auto_input_tids);
        self.allocator.free(self.aliased_state_tids);
        self.allocator.free(self.aliased_state_synced_versions);
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

    pub fn bindInput(self: *Self, name: []const u8, tensor: Tensor) api_errors.ApiError!void {
        const index = self.findInputIndex(name) orelse return api_errors.ApiError.InvalidArgument;
        const sig = self.input_signatures[index];
        if (tensor.store != self.store) return api_errors.ApiError.InvalidArgument;
        if (tensor.dtype != sig.dtype) return api_errors.ApiError.InvalidArgument;
        if (tensor.shape.len != sig.rank) return api_errors.ApiError.InvalidArgument;
        self.bound_inputs[index] = tensor;
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
    pub fn resetState(self: *Self) api_errors.ExecuteError!void {
        for (self.aliased_state_tids) |tid| {
            if (tid == types_mod.invalid_tensor_id) continue;
            initializers.zeroStoreTensor(self.store, tid) catch return error.InvalidArgument;
        }
    }

    pub fn run(self: *Self) api_errors.ExecuteError!void {
        const trace: bool = self.trace_runs;
        if (trace) std.debug.print("[aion][run] begin\n", .{});

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
                    if (src.id != dst.id) dst.copyFrom(self.allocator, src) catch |e| {
                        if (trace) {
                            std.debug.print(
                                "[aion][run] copyFrom failed for aliased input {s} (src_id={d} dst_id={d}): {s}\n",
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
            dst.copyFrom(self.allocator, src) catch |e| {
                if (trace) {
                    std.debug.print(
                        "[aion][run] output alias copyFrom failed (input={s} output={s}): {s}\n",
                        .{ alias.input_name, alias.output_name, @errorName(e) },
                    );
                }
                return e;
            };
        }

        if (trace) {
            std.debug.print("[aion][run] done\n", .{});
        }

        self.last_run_cache_index = cache_index;
    }

    pub fn outputTensor(self: *Self, name: []const u8) api_errors.ApiError!Tensor {
        const cache_index = self.last_run_cache_index orelse return api_errors.ApiError.InvalidArgument;
        const output_index = self.findOutputIndex(name) orelse return api_errors.ApiError.InvalidArgument;
        const tid = self.cache_entries.items[cache_index].program.outputs[output_index];
        const t = try self.store.getConst(tid);
        return .{ .store = self.store, .id = tid, .dtype = t.dtype, .shape = t.shape };
    }

    pub fn outputCount(self: *const Self) usize {
        return self.output_signatures.len;
    }

    /// Fetch the most recent run's output by position (in declared output order).
    pub fn outputTensorAt(self: *Self, index: usize) api_errors.ApiError!Tensor {
        const cache_index = self.last_run_cache_index orelse return api_errors.ApiError.InvalidArgument;
        if (index >= self.output_signatures.len) return api_errors.ApiError.InvalidArgument;
        const tid = self.cache_entries.items[cache_index].program.outputs[index];
        const t = try self.store.getConst(tid);
        return .{ .store = self.store, .id = tid, .dtype = t.dtype, .shape = t.shape };
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
        source: GraphSource,
        initializer_tids: []TensorId,
        input_signatures: []SignatureInfo,
        output_signatures: []SignatureInfo,
        input_shape_terms: []const []const package_file.ShapeTerm,
        dim_exprs: []const package_file.DimExpr,
        dim_symbol_count: usize,
        io_aliases: []const package_file.IoAlias,
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
            .bound_inputs = bound_inputs,
            .aliased_input_bind_versions = aliased_input_bind_versions,
            .auto_input_tids = auto_input_tids,
            .auto_init_inputs = opts.auto_init_inputs,
            .aliased_state_tids = aliased_state_tids,
            .aliased_state_synced_versions = aliased_state_synced_versions,
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
            .{ .package = package },
            initializer_tids,
            input_signatures,
            output_signatures,
            input_shape_terms,
            package.dim_exprs,
            package.dim_symbols.len,
            package.io_aliases,
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
        graph_in: graph_mod.Graph,
        input_value_ids: []const graph_mod.ValueId,
        input_names: []const []const u8,
        output_value_ids: []const graph_mod.ValueId,
        output_names: []const []const u8,
        io_aliases: []const package_file.IoAlias,
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
            input_signatures[k] = .{ .name = try aa.dupe(u8, input_names[k]), .value = vid, .dtype = dtype, .rank = @intCast(v.shape.len) };
            input_shape_terms[k] = terms;
        }
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
            source,
            empty_tids,
            input_signatures,
            output_signatures,
            input_shape_terms,
            &[_]package_file.DimExpr{},
            0,
            aliases_dup,
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
            // Compiled model: the graph already exists (nodes + bound params). Bind
            // public inputs + compile in place — no graph reconstruction.
            .builder => |*b| return self.finishCacheEntry(&b.graph, b.input_value_ids, b.output_value_ids, symbol_values, input_shapes, direct_input_ids),
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
                // different input shapes (prefill vs decode). Single-tile (KVCacheAppend
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

        return .{
            .symbol_values = try self.allocator.dupe(u64, symbol_values),
            .input_shapes = try self.allocator.dupe(usize, input_shapes),
            .input_slots = input_slots,
            .direct_input_ids = entry_direct_input_ids,
            .program = program,
        };
    }

    /// Return the model-level backing slot for io-aliased input `input_index`,
    /// allocating a zeroed single-tile tensor on first use. If a slot already
    /// exists with a different shape (a capacity change), it is replaced and the
    /// seed must be re-copied, so the synced version is reset.
    fn ensureAliasedStateSlot(self: *Self, input_index: usize, shape: []const usize) api_errors.ExecuteError!TensorId {
        const sig = self.input_signatures[input_index];
        const cur = self.aliased_state_tids[input_index];
        if (cur != types_mod.invalid_tensor_id) {
            const meta = self.store.getConst(cur) catch return error.InvalidArgument;
            if (signatures.sameUsize(meta.shape, shape)) return cur;
            // Shape (capacity) changed: re-seed into the new slot on the next run.
            self.aliased_state_synced_versions[input_index] = 0;
        }
        const tid = try initializers.createTensorSingleTile(self.store, self.policy, sig.dtype, shape);
        self.aliased_state_tids[input_index] = tid;
        return tid;
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
