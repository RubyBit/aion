// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const graph_mod = @import("../../graph/graph.zig");
const backend_types = @import("../../backend/types.zig");

pub const DType = backend_types.DType;
pub const ElemwiseBinaryOp = backend_types.ElemwiseBinaryOp;
pub const UnaryOp = backend_types.UnaryOp;
pub const ReduceOp = backend_types.ReduceOp;
pub const PadMode = backend_types.PadMode;
pub const ValueId = graph_mod.ValueId;
pub const AttentionWindow = graph_mod.AttentionWindow;

pub const magic_bytes: [4]u8 = .{ 'A', 'I', 'O', 'N' };
/// v12: Attention and RelPosMHA share one `AttentionWindow` (left/right/chunk) in
/// place of `causal`/`sliding_window` and `window_kind`/`chunk_size`. Parsing
/// requires an exact match, so every `.aion` must be re-converted.
pub const current_version: u32 = 12;
pub const header_size: usize = 72;
pub const section_desc_size: usize = 24;
pub const invalid_index: u32 = std.math.maxInt(u32);
const max_rank: usize = 8;

pub const PackageError = error{
    InvalidArgument,
    InvalidFormat,
    UnsupportedVersion,
    UnsupportedFeature,
    OutOfMemory,
    IoFailure,
};

pub const SectionType = enum(u32) {
    strings = 1,
    tensors = 2,
    values = 3,
    nodes = 4,
    signatures = 5,
    graph_meta = 6,
    dim_symbols = 7,
    dim_exprs = 8,
    metadata = 9,
    debug_names = 10,
    regions = 11,
    io_aliases = 12,
    input_roles = 13,
};

pub const section_slot_count = @typeInfo(SectionType).@"enum".field_names.len;

pub const SectionFlags = struct {
    pub const required: u32 = 1 << 0;
};

pub const ShapeTerm = union(enum) {
    constant: u64,
    expr: u32,
};

pub const BinaryDimExpr = struct {
    lhs: ShapeTerm,
    rhs: ShapeTerm,
};

pub const DimExpr = union(enum) {
    symbol: u32,
    add: BinaryDimExpr,
    sub: BinaryDimExpr,
    mul: BinaryDimExpr,
    floor_div: BinaryDimExpr,
    max: BinaryDimExpr,
};

pub const DimSymbol = struct {
    name: []const u8,
};

pub const QuantizedEncoding = struct {
    scheme: []const u8,
    logical_dtype: DType,
    block_elems: u32,
    block_bytes: u32,
    quant_axis: i32,
    params: []const u8,
};

pub const TensorEncoding = union(enum) {
    plain: DType,
    quantized: QuantizedEncoding,
};

pub const Initializer = struct {
    encoding: TensorEncoding,
    data: []const u8,
};

pub const ValueSource = enum(u8) {
    public_input = 0,
    initializer = 1,
    produced = 2,
};

pub const ValueRecord = struct {
    dtype: DType,
    rank: u8,
    source: ValueSource,
    shape_terms: []const ShapeTerm,
    initializer_index: ?u32 = null,
};

pub const NamedValue = struct {
    name: []const u8,
    value: u32,
};

pub const MetadataEntry = struct {
    key: []const u8,
    value: []const u8,
};

pub const DebugName = struct {
    value: u32,
    name: []const u8,
};

pub const IoAlias = struct {
    input: u32,
    output: u32,
};

/// Semantic role of a public input, declared by the package author. Roles let the
/// runtime auto-allocate caches (including ones with a free capacity symbol) and
/// auto-drive position/index inputs, instead of every caller re-deriving them.
pub const InputRoleKind = enum(u8) {
    /// io-aliased KV-style sequence cache; `axis` is the time/capacity axis.
    sequence_cache = 1,
    /// i32 [B]: physical write position fed to SequenceAppend.
    cache_write_index = 2,
    /// i32 [B]: visible end (write position + new tokens) fed to attention.
    cache_visible_end = 3,
    /// i32 [B,S]: absolute positions of the new tokens; `axis` is the sequence axis.
    positions = 4,
    /// The input whose `axis` extent defines "new tokens this run" (token ids).
    tokens = 5,
    /// Generic io-aliased recurrent state (LSTM h/c etc.); zero-initialized.
    state = 6,
};

pub const InputRoleFlags = struct {
    pub const zero_init: u8 = 1 << 0;
    pub const allow_growable: u8 = 1 << 1;
    pub const allow_ring: u8 = 1 << 2;
    pub const all: u8 = zero_init | allow_growable | allow_ring;
};

pub const input_role_no_axis: u8 = 0xFF;

pub const InputRole = struct {
    /// Index into the signatures section's input list.
    input: u32,
    kind: InputRoleKind,
    axis: u8 = input_role_no_axis,
    flags: u8 = InputRoleFlags.zero_init,
    /// dim_symbols index of the free capacity symbol, or `invalid_index` (fixed shape).
    capacity_symbol: u32 = invalid_index,
};

/// The graph's own op, which is what a package stores.
///
/// There is deliberately no second op type. The two used to differ in integer widths, in
/// whether a free axis was a `ShapeTerm` or a size plus a symbol, and in whether a Loop's
/// extra outputs sat on the op or the node — twelve of thirty-two arms — and each
/// difference cost a 32-arm conversion in both directions. `graph.Op` absorbed all three,
/// so `parse` and `write` are the only places that know an op field by name, which is
/// what a serializer is for.
pub const NodeOp = graph_mod.Op;

pub const NodeOpKind = graph_mod.OpTag;

pub fn nodeOpKind(op: NodeOp) NodeOpKind {
    return std.meta.activeTag(op);
}

pub fn parseNodeOpKind(raw: u8) ?NodeOpKind {
    return std.enums.fromInt(NodeOpKind, raw);
}

/// A stored node, which is the graph's own node.
pub const NodeRecord = graph_mod.Node;

pub const RegionRecord = graph_mod.Region;

pub const GraphMeta = struct {
    value_count: u32,
    node_count: u32,
    region_count: u32,
    initializer_count: u32,
    input_count: u32,
    output_count: u32,
    dim_symbol_count: u32,
    dim_expr_count: u32,
    metadata_count: u32,
    debug_name_count: u32,
    io_alias_count: u32,
};

pub const Package = struct {
    allocator: std.mem.Allocator,
    initializers: []Initializer,
    values: []ValueRecord,
    nodes: []NodeRecord,
    regions: []RegionRecord = &[_]RegionRecord{},
    inputs: []NamedValue,
    outputs: []NamedValue,
    dim_symbols: []DimSymbol,
    dim_exprs: []DimExpr,
    metadata: []MetadataEntry,
    debug_names: []DebugName,
    io_aliases: []IoAlias,
    input_roles: []InputRole = &[_]InputRole{},

    /// When non-null, this Package owns the raw file-byte buffer and some of its
    /// internal slices (notably `Initializer.data`, the largest by far) are borrows
    /// into this buffer instead of separately-allocated copies. `deinit` frees the
    /// buffer once. Set by `parseTakeOwned`; `parse` leaves it null and keeps the
    /// original copy-on-parse semantics.
    source_bytes: ?[]u8 = null,

    const Self = @This();

    pub fn graphMeta(self: *const Self) GraphMeta {
        return .{
            .value_count = @intCast(self.values.len),
            .node_count = @intCast(self.nodes.len),
            .region_count = @intCast(self.regions.len),
            .initializer_count = @intCast(self.initializers.len),
            .input_count = @intCast(self.inputs.len),
            .output_count = @intCast(self.outputs.len),
            .dim_symbol_count = @intCast(self.dim_symbols.len),
            .dim_expr_count = @intCast(self.dim_exprs.len),
            .metadata_count = @intCast(self.metadata.len),
            .debug_name_count = @intCast(self.debug_names.len),
            .io_alias_count = @intCast(self.io_aliases.len),
        };
    }

    /// Free the raw file-byte buffer once tensor data has been copied elsewhere.
    ///
    /// After this runs, every `Initializer.data` (and borrowed `QuantizedEncoding.params`)
    /// slice in this Package is emptied — callers must not read them afterwards. This
    /// exists to reclaim the 4–5 GB file buffer the moment weights have been imported
    /// into the storage manager, halving peak RSS for loaded models.
    pub fn releaseSourceBytes(self: *Self) void {
        const buf = self.source_bytes orelse return;
        for (self.initializers) |*init| {
            init.data = &[_]u8{};
            switch (init.encoding) {
                .quantized => |*q| {
                    q.params = &[_]u8{};
                },
                else => {},
            }
        }
        self.allocator.free(buf);
        self.source_bytes = null;
    }

    pub fn deinit(self: *Self) void {
        // If we own `source_bytes`, most `Initializer.data` and `QuantizedEncoding.params`
        // slices borrow into that buffer; skip freeing them individually.
        const borrowed_init_data: bool = self.source_bytes != null;
        freeInitializersImpl(self.allocator, self.initializers, borrowed_init_data);
        freeValues(self.allocator, self.values);
        freeNodes(self.allocator, self.nodes);
        freeRegions(self.allocator, self.regions);
        freeNamedValues(self.allocator, self.inputs);
        freeNamedValues(self.allocator, self.outputs);
        freeDimSymbols(self.allocator, self.dim_symbols);
        self.allocator.free(self.dim_exprs);
        freeMetadata(self.allocator, self.metadata);
        freeDebugNames(self.allocator, self.debug_names);
        self.allocator.free(self.io_aliases);
        if (self.input_roles.len != 0) self.allocator.free(self.input_roles);
        if (self.source_bytes) |buf| self.allocator.free(buf);
        self.* = undefined;
    }

    pub fn validate(self: *const Self) PackageError!void {
        for (self.values, 0..) |value, idx| {
            if (value.rank == 0 or value.rank > max_rank) return PackageError.InvalidFormat;
            if (value.shape_terms.len != 0 and value.shape_terms.len != value.rank) return PackageError.InvalidFormat;
            switch (value.source) {
                .public_input => {
                    if (value.shape_terms.len != value.rank) return PackageError.InvalidFormat;
                    if (value.initializer_index != null) return PackageError.InvalidFormat;
                },
                .initializer => {
                    if (value.shape_terms.len != value.rank) return PackageError.InvalidFormat;
                    if (value.initializer_index == null or value.initializer_index.? >= self.initializers.len) return PackageError.InvalidFormat;
                },
                .produced => {
                    if (value.initializer_index != null) return PackageError.InvalidFormat;
                },
            }
            for (value.shape_terms) |term| try validateShapeTerm(self, term);
            _ = idx;
        }

        try validateUniqueSignatureNames(self.inputs, self.outputs);
        try validateIoAliases(self);
        try validateInputRoles(self);

        var producer_counts = self.allocator.alloc(u8, self.values.len) catch return PackageError.OutOfMemory;
        defer self.allocator.free(producer_counts);
        @memset(producer_counts, 0);

        var available = self.allocator.alloc(bool, self.values.len) catch return PackageError.OutOfMemory;
        defer self.allocator.free(available);
        @memset(available, false);
        for (self.values, 0..) |value, idx| {
            if (value.source == .public_input or value.source == .initializer) available[idx] = true;
        }

        for (self.inputs) |sig| {
            if (sig.value >= self.values.len) return PackageError.InvalidFormat;
            if (self.values[sig.value].source != .public_input) return PackageError.InvalidFormat;
        }
        for (self.outputs) |sig| {
            if (sig.value >= self.values.len) return PackageError.InvalidFormat;
        }

        // Regions are validated lazily, just before the main `If`/`Loop` node that
        // references them — mirroring the loader's lazy instantiation. This lets a
        // region body reference values produced by earlier main nodes (e.g. an
        // in-graph decode Loop reading the encoder output).
        const region_validated = self.allocator.alloc(bool, self.regions.len) catch return PackageError.OutOfMemory;
        defer self.allocator.free(region_validated);
        @memset(region_validated, false);

        for (self.nodes) |node| {
            switch (node.op) {
                .If => |iff| {
                    try validateRegionLazy(self, @intCast(iff.then_region), &producer_counts, available, region_validated);
                    try validateRegionLazy(self, @intCast(iff.else_region), &producer_counts, available, region_validated);
                },
                .Loop => |lp| try validateRegionLazy(self, @intCast(lp.body_region), &producer_counts, available, region_validated),
                else => {},
            }
            try validateNodeRefs(self, node, &producer_counts, available);
            available[node.output] = true;
            for (nodeExtraOutputs(node)) |extra| available[extra] = true;
        }

        for (self.values, 0..) |value, idx| {
            if (value.source == .produced and producer_counts[idx] != 1) return PackageError.InvalidFormat;
        }

        for (self.initializers) |init| try validateInitializer(init);
    }
};

pub fn validate(pkg: *const Package) PackageError!void {
    try pkg.validate();
}

/// An op attribute's free axes: one slot per axis (or none at all), each naming a dim
/// expression the package declares.
fn validateAttrFreeDims(pkg: *const Package, rank: usize, free_dims: []const ?u32) PackageError!void {
    if (free_dims.len == 0) return;
    if (free_dims.len != rank) return PackageError.InvalidFormat;
    for (free_dims) |free_dim| {
        if (free_dim) |expr| if (expr >= pkg.dim_exprs.len) return PackageError.InvalidFormat;
    }
}

fn validateShapeTerm(pkg: *const Package, term: ShapeTerm) PackageError!void {
    switch (term) {
        .constant => |value| if (value == 0) return PackageError.InvalidFormat,
        .expr => |expr_idx| if (expr_idx >= pkg.dim_exprs.len) return PackageError.InvalidFormat,
    }
}

fn validateInitializer(init: Initializer) PackageError!void {
    switch (init.encoding) {
        .plain => |dtype| if (dtype.info().is_quantized) return PackageError.InvalidFormat,
        .quantized => |q| {
            if (q.scheme.len == 0) return PackageError.InvalidFormat;
            if (q.block_elems == 0 or q.block_bytes == 0) return PackageError.InvalidFormat;
            if (q.logical_dtype.info().is_quantized) return PackageError.InvalidFormat;
        },
    }
    if (init.data.len == 0) return PackageError.InvalidFormat;
}

fn validateUniqueSignatureNames(inputs: []const NamedValue, outputs: []const NamedValue) PackageError!void {
    var i: usize = 0;
    while (i < inputs.len) : (i += 1) {
        var j: usize = i + 1;
        while (j < inputs.len) : (j += 1) {
            if (std.mem.eql(u8, inputs[i].name, inputs[j].name)) return PackageError.InvalidFormat;
        }
        j = 0;
        while (j < outputs.len) : (j += 1) {
            if (std.mem.eql(u8, inputs[i].name, outputs[j].name)) return PackageError.InvalidFormat;
        }
    }
    i = 0;
    while (i < outputs.len) : (i += 1) {
        var j: usize = i + 1;
        while (j < outputs.len) : (j += 1) {
            if (std.mem.eql(u8, outputs[i].name, outputs[j].name)) return PackageError.InvalidFormat;
        }
    }
}

fn validateIoAliases(pkg: *const Package) PackageError!void {
    var used_inputs = pkg.allocator.alloc(bool, pkg.inputs.len) catch return PackageError.OutOfMemory;
    defer pkg.allocator.free(used_inputs);
    @memset(used_inputs, false);

    var used_outputs = pkg.allocator.alloc(bool, pkg.outputs.len) catch return PackageError.OutOfMemory;
    defer pkg.allocator.free(used_outputs);
    @memset(used_outputs, false);

    for (pkg.io_aliases) |alias| {
        const input_idx: usize = alias.input;
        const output_idx: usize = alias.output;
        if (input_idx >= pkg.inputs.len or output_idx >= pkg.outputs.len) return PackageError.InvalidFormat;

        // The signature entries must reference valid value indices.
        const input_value_idx: usize = pkg.inputs[input_idx].value;
        const output_value_idx: usize = pkg.outputs[output_idx].value;
        if (input_value_idx >= pkg.values.len or output_value_idx >= pkg.values.len) return PackageError.InvalidFormat;

        const input_value = pkg.values[input_value_idx];
        const output_value = pkg.values[output_value_idx];

        // `io_aliases` are a performance hint (copy output back into an input slot).
        // They must always be compatible (same dtype + rank).
        if (input_value.dtype != output_value.dtype) return PackageError.InvalidFormat;
        if (input_value.rank != output_value.rank) return PackageError.InvalidFormat;

        if (used_inputs[input_idx] or used_outputs[output_idx]) return PackageError.InvalidFormat;
        used_inputs[input_idx] = true;
        used_outputs[output_idx] = true;
    }
}

fn validateInputRoles(pkg: *const Package) PackageError!void {
    var role_seen = pkg.allocator.alloc(bool, pkg.inputs.len) catch return PackageError.OutOfMemory;
    defer pkg.allocator.free(role_seen);
    @memset(role_seen, false);

    var aliased = pkg.allocator.alloc(bool, pkg.inputs.len) catch return PackageError.OutOfMemory;
    defer pkg.allocator.free(aliased);
    @memset(aliased, false);
    for (pkg.io_aliases) |alias| {
        if (alias.input < pkg.inputs.len) aliased[alias.input] = true;
    }

    var singleton_seen: [4]bool = @splat(false); // write_index, visible_end, positions, tokens

    for (pkg.input_roles) |role| {
        const input_idx: usize = role.input;
        if (input_idx >= pkg.inputs.len) return PackageError.InvalidFormat;
        if (role_seen[input_idx]) return PackageError.InvalidFormat;
        role_seen[input_idx] = true;
        if ((role.flags & ~InputRoleFlags.all) != 0) return PackageError.InvalidFormat;

        const value_idx: usize = pkg.inputs[input_idx].value;
        if (value_idx >= pkg.values.len) return PackageError.InvalidFormat;
        const value = pkg.values[value_idx];

        switch (role.kind) {
            .sequence_cache, .state => {
                if (!aliased[input_idx]) return PackageError.InvalidFormat;
                if (role.kind == .sequence_cache) {
                    if (role.axis == input_role_no_axis or role.axis >= value.rank) return PackageError.InvalidFormat;
                    if (role.capacity_symbol != invalid_index) {
                        if (role.capacity_symbol >= pkg.dim_symbols.len) return PackageError.InvalidFormat;
                        if (value.shape_terms[role.axis] != .expr) return PackageError.InvalidFormat;
                    }
                }
            },
            .cache_write_index, .cache_visible_end, .positions, .tokens => {
                const slot: usize = switch (role.kind) {
                    .cache_write_index => 0,
                    .cache_visible_end => 1,
                    .positions => 2,
                    .tokens => 3,
                    else => unreachable,
                };
                if (singleton_seen[slot]) return PackageError.InvalidFormat;
                singleton_seen[slot] = true;
                if (value.dtype != .i32) return PackageError.InvalidFormat;
                if (role.kind == .positions or role.kind == .tokens) {
                    if (role.axis == input_role_no_axis or role.axis >= value.rank) return PackageError.InvalidFormat;
                }
                if (role.capacity_symbol != invalid_index) return PackageError.InvalidFormat;
            },
        }
    }
}

fn validateRegionLazy(
    pkg: *const Package,
    region_idx: usize,
    producer_counts: *[]u8,
    available: []bool,
    region_validated: []bool,
) PackageError!void {
    if (region_idx >= pkg.regions.len) return PackageError.InvalidFormat;
    if (region_validated[region_idx]) return;
    const region = pkg.regions[region_idx];
    if (region.outputs.len == 0) return PackageError.InvalidFormat;
    // Pre-validate sub-regions referenced by nested If/Loop nodes.
    for (region.nodes) |n| {
        switch (n.op) {
            .If => |iff| {
                try validateRegionLazy(pkg, @intCast(iff.then_region), producer_counts, available, region_validated);
                try validateRegionLazy(pkg, @intCast(iff.else_region), producer_counts, available, region_validated);
            },
            .Loop => |lp| try validateRegionLazy(pkg, @intCast(lp.body_region), producer_counts, available, region_validated),
            else => {},
        }
    }
    for (region.nodes) |node| {
        try validateNodeRefs(pkg, node, producer_counts, available);
        available[node.output] = true;
        for (nodeExtraOutputs(node)) |extra| available[extra] = true;
    }
    for (region.outputs) |output| {
        if (output >= pkg.values.len or !available[output]) return PackageError.InvalidFormat;
    }
    region_validated[region_idx] = true;
}

/// Outputs of a node beyond the primary `output` (multi-carry `Loop` only).
pub fn nodeExtraOutputs(node: NodeRecord) []const u32 {
    return node.extra_outputs;
}

fn validateNodeRefs(pkg: *const Package, node: NodeRecord, producer_counts: *[]u8, available: []const bool) PackageError!void {
    if (node.output >= pkg.values.len) return PackageError.InvalidFormat;
    if (pkg.values[node.output].source != .produced) return PackageError.InvalidFormat;
    if (producer_counts.*[node.output] == std.math.maxInt(u8)) return PackageError.InvalidFormat;
    producer_counts.*[node.output] += 1;
    for (nodeExtraOutputs(node)) |extra| {
        if (extra >= pkg.values.len) return PackageError.InvalidFormat;
        if (pkg.values[extra].source != .produced) return PackageError.InvalidFormat;
        if (producer_counts.*[extra] == std.math.maxInt(u8)) return PackageError.InvalidFormat;
        producer_counts.*[extra] += 1;
    }
    for (node.inputs) |input| {
        if (input >= pkg.values.len or !available[input]) return PackageError.InvalidFormat;
    }
    try validateNode(pkg, node);
}

fn validateNode(pkg: *const Package, node: NodeRecord) PackageError!void {
    switch (node.op) {
        .LayerNorm => |ln| {
            if (ln.normalized_shape.len == 0 or ln.normalized_shape.len > max_rank) return PackageError.InvalidFormat;
            try validateAttrFreeDims(pkg, ln.normalized_shape.len, ln.free_dims);
        },
        .RMSNorm => |ln| {
            if (ln.normalized_shape.len == 0 or ln.normalized_shape.len > max_rank) return PackageError.InvalidFormat;
            try validateAttrFreeDims(pkg, ln.normalized_shape.len, ln.free_dims);
        },
        .RelPosMHA => |attn| {
            if (!(attn.scale > 0.0) or !std.math.isFinite(attn.scale)) return PackageError.InvalidFormat;
            if (!std.math.isFinite(attn.attn_logits_soft_cap) or attn.attn_logits_soft_cap < 0) return PackageError.InvalidFormat;
        },
        .Attention => |attn| {
            const expected_inputs: usize = 3 +
                @as(usize, @intFromBool(attn.has_query_positions)) +
                @as(usize, @intFromBool(attn.has_kv_lengths));
            if (node.inputs.len != expected_inputs) return PackageError.InvalidFormat;
            if (!(attn.scale > 0.0) or !std.math.isFinite(attn.scale)) return PackageError.InvalidFormat;
            if (!std.math.isFinite(attn.attn_logits_soft_cap) or attn.attn_logits_soft_cap < 0.0) return PackageError.InvalidFormat;
        },
        .STFT => |st| if (st.n_fft < 4 or (st.n_fft & (st.n_fft - 1)) != 0 or st.hop_length == 0) return PackageError.InvalidFormat,
        .If => |iff| {
            if (iff.then_region >= pkg.regions.len or iff.else_region >= pkg.regions.len) return PackageError.InvalidFormat;
            if (pkg.regions[iff.then_region].outputs.len != 1 or pkg.regions[iff.else_region].outputs.len != 1) return PackageError.InvalidFormat;
        },
        .Loop => |lp| {
            if (lp.body_region >= pkg.regions.len or lp.static_max_trip_count == 0) return PackageError.InvalidFormat;
            const ncar = pkg.regions[lp.body_region].outputs.len;
            if (ncar == 0) return PackageError.InvalidFormat;
            // N carried inits, N body outputs, N node outputs (primary + extras).
            if (node.inputs.len != ncar or node.extra_outputs.len + 1 != ncar) return PackageError.InvalidFormat;
            if (lp.cond_carry) |c| if (c >= ncar) return PackageError.InvalidFormat;
            for (node.extra_outputs) |e| if (e >= pkg.values.len) return PackageError.InvalidFormat;
        },
        .ViewReshape => |vr| {
            if (vr.new_shape.len == 0 or vr.new_shape.len > max_rank) return PackageError.InvalidFormat;
            try validateAttrFreeDims(pkg, vr.new_shape.len, vr.free_dims);
        },
        .ViewSliceND => |sl| {
            if (sl.starts.len == 0 or sl.starts.len != sl.lens.len or sl.starts.len > max_rank) return PackageError.InvalidFormat;
            try validateAttrFreeDims(pkg, sl.lens.len, sl.free_dims);
        },
        else => {},
    }
}

fn freeInitializers(allocator: std.mem.Allocator, initializers: []Initializer) void {
    freeInitializersImpl(allocator, initializers, false);
}

fn freeInitializersImpl(allocator: std.mem.Allocator, initializers: []Initializer, borrowed_data: bool) void {
    for (initializers) |*init| {
        switch (init.encoding) {
            .plain => {},
            .quantized => |q| {
                allocator.free(q.scheme);
                if (!borrowed_data) allocator.free(q.params);
            },
        }
        if (!borrowed_data) allocator.free(init.data);
    }
    allocator.free(initializers);
}

fn freeValues(allocator: std.mem.Allocator, values: []ValueRecord) void {
    for (values) |value| allocator.free(value.shape_terms);
    allocator.free(values);
}

fn freeRegions(allocator: std.mem.Allocator, regions: []RegionRecord) void {
    for (regions) |region| {
        freeNodes(allocator, region.nodes);
        allocator.free(region.outputs);
    }
    if (regions.len != 0) allocator.free(regions);
}

fn freeNodes(allocator: std.mem.Allocator, nodes: []NodeRecord) void {
    for (nodes) |node| {
        allocator.free(node.inputs);
        if (node.extra_outputs.len != 0) allocator.free(@constCast(node.extra_outputs));
        deinitNodeOp(allocator, node.op);
    }
    allocator.free(nodes);
}

fn freeNamedValues(allocator: std.mem.Allocator, values: []NamedValue) void {
    for (values) |sig| allocator.free(sig.name);
    allocator.free(values);
}

fn freeMetadata(allocator: std.mem.Allocator, metadata: []MetadataEntry) void {
    for (metadata) |entry| {
        allocator.free(entry.key);
        allocator.free(entry.value);
    }
    allocator.free(metadata);
}

fn freeDebugNames(allocator: std.mem.Allocator, debug_names: []DebugName) void {
    for (debug_names) |entry| allocator.free(entry.name);
    allocator.free(debug_names);
}

fn freeDimSymbols(allocator: std.mem.Allocator, symbols: []DimSymbol) void {
    for (symbols) |sym| allocator.free(sym.name);
    allocator.free(symbols);
}

pub fn deinitNodeOp(allocator: std.mem.Allocator, op: NodeOp) void {
    var mutable = op;
    if (graph_mod.symbolicAttr(&mutable)) |attr| {
        allocator.free(attr.sizes);
        if (attr.free_dims.len != 0) allocator.free(@constCast(attr.free_dims));
    }
    // `starts` is the one slice attribute that is never symbolic.
    switch (op) {
        .ViewSliceND => |sl| allocator.free(sl.starts),
        else => {},
    }
}
