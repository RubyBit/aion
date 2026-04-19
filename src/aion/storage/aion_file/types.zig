const std = @import("std");

const graph_mod = @import("../../graph/graph.zig");
const backend_types = @import("../../backend/types.zig");

pub const DType = backend_types.DType;
pub const ElemwiseBinaryOp = backend_types.ElemwiseBinaryOp;
pub const UnaryOp = backend_types.UnaryOp;
pub const ReduceOp = backend_types.ReduceOp;
pub const PadMode = backend_types.PadMode;
pub const ValueId = graph_mod.ValueId;

pub const magic_bytes: [4]u8 = .{ 'A', 'I', 'O', 'N' };
pub const current_version: u32 = 4;
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
    // NOTE: v4 removed the `subgraphs` section (formerly 11) entirely.
    io_aliases = 12,
};

pub const section_slot_count = @typeInfo(SectionType).@"enum".fields.len;

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

pub const NodeOp = union(graph_mod.OpTag) {
    MatMul: struct { alpha: f32, beta: f32 },
    ElemwiseBinary: struct { op: ElemwiseBinaryOp },
    BroadcastLastDimBinary: struct { op: ElemwiseBinaryOp },
    Unary: struct { op: UnaryOp },
    Softmax: struct { axis: i32 },
    Conv1D: struct {
        stride: u64,
        dilation: u64,
        pad_left: u64,
        pad_right: u64,
        pad_mode: PadMode,
        groups: u64,
    },
    Conv2D: struct {
        stride_h: u64,
        stride_w: u64,
        dilation_h: u64,
        dilation_w: u64,
        pad_top: u64,
        pad_bottom: u64,
        pad_left: u64,
        pad_right: u64,
        pad_mode: PadMode,
        groups: u64,
    },
    LayerNorm: struct { eps: f32, normalized_shape: []const ShapeTerm },
    RMSNorm: struct { eps: f32, normalized_shape: []const ShapeTerm },
    Attention: struct { scale: f32, causal: bool },
    MultiHeadAttention: struct { scale: f32, causal: bool, heads: u64 },
    Reduce: struct { op: ReduceOp, axis: ?i32 },
    Concat: struct { axis: i32 },
    LSTMCell: struct { has_bias: bool },
    ComplexAbsMean: struct { out_channels: u64 },
    Copy: void,
    ViewReshape: struct { new_shape: []const ShapeTerm },
    ViewSqueeze: struct { axis: ?i32 },
    ViewUnsqueeze: struct { axis: i32 },
    ViewTranspose2D: void,
    ViewSliceND: struct { starts: []const u64, lens: []const ShapeTerm },

    /// Gather rows from a 2D table using i32 indices.
    ///
    /// Shapes:
    /// - table:   [V, D]
    /// - indices: [B, L]
    /// - out:     [B, L, D]
    GatherRows: void,

    RoPE1D: struct {
        base_frequency: f32,
        scale_factor: f32,
        rope_proportion: f32,
    },

    KVCacheAppend: void,

    MultiHeadAttentionCached: struct {
        scale: f32,
        causal: bool,
        sliding_window: u64,
        attn_logits_soft_cap: f32,
    },

    Cast: struct { to_dtype: DType },
    MatMulNT: struct { alpha: f32, beta: f32 },
};

/// Stable on-disk op ids are sourced from `graph.OpTag` so graph/runtime and
/// package serialization share one canonical opcode definition.
pub const NodeOpKind = graph_mod.OpTag;

pub fn nodeOpKind(op: NodeOp) NodeOpKind {
    return std.meta.activeTag(op);
}

pub fn parseNodeOpKind(raw: u8) ?NodeOpKind {
    return std.enums.fromInt(NodeOpKind, raw);
}

pub const NodeRecord = struct {
    inputs: []const u32,
    output: u32,
    op: NodeOp,
};

pub const GraphMeta = struct {
    value_count: u32,
    node_count: u32,
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
    inputs: []NamedValue,
    outputs: []NamedValue,
    dim_symbols: []DimSymbol,
    dim_exprs: []DimExpr,
    metadata: []MetadataEntry,
    debug_names: []DebugName,
    io_aliases: []IoAlias,

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
        freeNamedValues(self.allocator, self.inputs);
        freeNamedValues(self.allocator, self.outputs);
        freeDimSymbols(self.allocator, self.dim_symbols);
        self.allocator.free(self.dim_exprs);
        freeMetadata(self.allocator, self.metadata);
        freeDebugNames(self.allocator, self.debug_names);
        self.allocator.free(self.io_aliases);
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

        for (self.nodes) |node| {
            if (node.output >= self.values.len) return PackageError.InvalidFormat;
            if (self.values[node.output].source != .produced) return PackageError.InvalidFormat;
            if (producer_counts[node.output] == std.math.maxInt(u8)) return PackageError.InvalidFormat;
            producer_counts[node.output] += 1;
            for (node.inputs) |input| {
                if (input >= self.values.len or !available[input]) return PackageError.InvalidFormat;
            }
            try validateNode(self, node);
            available[node.output] = true;
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

fn validateNode(pkg: *const Package, node: NodeRecord) PackageError!void {
    switch (node.op) {
        .LayerNorm => |ln| {
            if (ln.normalized_shape.len == 0 or ln.normalized_shape.len > max_rank) return PackageError.InvalidFormat;
            for (ln.normalized_shape) |term| try validateShapeTerm(pkg, term);
        },
        .RMSNorm => |ln| {
            if (ln.normalized_shape.len == 0 or ln.normalized_shape.len > max_rank) return PackageError.InvalidFormat;
            for (ln.normalized_shape) |term| try validateShapeTerm(pkg, term);
        },
        .MultiHeadAttention => |attn| if (attn.heads == 0) return PackageError.InvalidFormat,
        .MultiHeadAttentionCached => |attn| {
            _ = attn.causal;
            if (!(attn.scale > 0.0) or !std.math.isFinite(attn.scale)) return PackageError.InvalidFormat;
            if (!std.math.isFinite(attn.attn_logits_soft_cap) or attn.attn_logits_soft_cap < 0.0) return PackageError.InvalidFormat;
        },
        .ComplexAbsMean => |cm| if (cm.out_channels == 0) return PackageError.InvalidFormat,
        .ViewReshape => |vr| {
            if (vr.new_shape.len == 0 or vr.new_shape.len > max_rank) return PackageError.InvalidFormat;
            for (vr.new_shape) |term| try validateShapeTerm(pkg, term);
        },
        .ViewSliceND => |sl| {
            if (sl.starts.len == 0 or sl.starts.len != sl.lens.len or sl.starts.len > max_rank) return PackageError.InvalidFormat;
            for (sl.lens) |term| try validateShapeTerm(pkg, term);
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

fn freeNodes(allocator: std.mem.Allocator, nodes: []NodeRecord) void {
    for (nodes) |node| {
        allocator.free(node.inputs);
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
    switch (op) {
        .LayerNorm => |ln| allocator.free(ln.normalized_shape),
        .RMSNorm => |ln| allocator.free(ln.normalized_shape),
        .ViewReshape => |vr| allocator.free(vr.new_shape),
        .ViewSliceND => |sl| {
            allocator.free(sl.starts);
            allocator.free(sl.lens);
        },
        else => {},
    }
}
