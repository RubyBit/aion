// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const types = @import("types.zig");

const DType = types.DType;
const ElemwiseBinaryOp = types.ElemwiseBinaryOp;
const UnaryOp = types.UnaryOp;
const ReduceOp = types.ReduceOp;
const PadMode = types.PadMode;
const current_version = types.current_version;
const magic_bytes = types.magic_bytes;
const header_size = types.header_size;
const section_desc_size = types.section_desc_size;
const invalid_index = types.invalid_index;
const PackageError = types.PackageError;
const SectionType = types.SectionType;
const SectionFlags = types.SectionFlags;
const ShapeTerm = types.ShapeTerm;
const BinaryDimExpr = types.BinaryDimExpr;
const DimExpr = types.DimExpr;
const DimSymbol = types.DimSymbol;
const Initializer = types.Initializer;
const NamedValue = types.NamedValue;
const MetadataEntry = types.MetadataEntry;
const DebugName = types.DebugName;
const IoAlias = types.IoAlias;
const ValueRecord = types.ValueRecord;
const NodeRecord = types.NodeRecord;
const RegionRecord = types.RegionRecord;
const NodeOpKind = types.NodeOpKind;
const NodeOp = types.NodeOp;
const GraphMeta = types.GraphMeta;
const Package = types.Package;

pub fn writeFile(file: std.Io.File, pkg: *const Package) PackageError!void {
    try pkg.validate();

    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    var interner = StringInterner.init(scratch);
    try collectPackageStrings(&interner, pkg);

    var sections: std.ArrayList(EncodedSection) = .empty;
    defer sections.deinit(scratch);

    try sections.append(scratch, .{ .section_type = .strings, .flags = SectionFlags.required, .bytes = try encodeStringsSection(scratch, &interner) });
    try sections.append(scratch, .{ .section_type = .tensors, .flags = SectionFlags.required, .bytes = try encodeInitializersSection(scratch, &interner, pkg.initializers) });
    try sections.append(scratch, .{ .section_type = .values, .flags = SectionFlags.required, .bytes = try encodeValuesSection(scratch, pkg.values) });
    try sections.append(scratch, .{ .section_type = .nodes, .flags = SectionFlags.required, .bytes = try encodeNodesSection(scratch, pkg.nodes) });
    if (pkg.regions.len != 0) try sections.append(scratch, .{ .section_type = .regions, .flags = 0, .bytes = try encodeRegionsSection(scratch, pkg.regions) });
    try sections.append(scratch, .{ .section_type = .signatures, .flags = SectionFlags.required, .bytes = try encodeSignaturesSection(scratch, &interner, pkg.inputs, pkg.outputs) });
    try sections.append(scratch, .{ .section_type = .graph_meta, .flags = SectionFlags.required, .bytes = try encodeGraphMetaSection(scratch, pkg.graphMeta()) });
    if (pkg.dim_symbols.len != 0) try sections.append(scratch, .{ .section_type = .dim_symbols, .flags = 0, .bytes = try encodeDimSymbolsSection(scratch, &interner, pkg.dim_symbols) });
    if (pkg.dim_exprs.len != 0) try sections.append(scratch, .{ .section_type = .dim_exprs, .flags = 0, .bytes = try encodeDimExprsSection(scratch, pkg.dim_exprs) });
    if (pkg.metadata.len != 0) try sections.append(scratch, .{ .section_type = .metadata, .flags = 0, .bytes = try encodeMetadataSection(scratch, &interner, pkg.metadata) });
    if (pkg.debug_names.len != 0) try sections.append(scratch, .{ .section_type = .debug_names, .flags = 0, .bytes = try encodeDebugNamesSection(scratch, &interner, pkg.debug_names) });
    if (pkg.io_aliases.len != 0) try sections.append(scratch, .{ .section_type = .io_aliases, .flags = 0, .bytes = try encodeIoAliasesSection(scratch, pkg.io_aliases) });
    if (pkg.input_roles.len != 0) try sections.append(scratch, .{ .section_type = .input_roles, .flags = 0, .bytes = try encodeInputRolesSection(scratch, pkg.input_roles) });

    const dir_offset: usize = header_size;
    const dir_size: usize = sections.items.len * section_desc_size;
    var total_size: usize = dir_offset + dir_size;
    for (sections.items) |section| {
        total_size = std.math.add(usize, total_size, section.bytes.len) catch return PackageError.InvalidArgument;
    }

    var writer: WriteCursor = .{ .file = file };
    try writeHeader(&writer, @intCast(sections.items.len), @intCast(dir_offset), @intCast(total_size));

    var next_offset: usize = dir_offset + dir_size;
    for (sections.items) |section| {
        try writeSectionDesc(&writer, section.section_type, section.flags, @intCast(next_offset), @intCast(section.bytes.len));
        next_offset += section.bytes.len;
    }
    for (sections.items) |section| try writer.writeAll(section.bytes);

    var io_backend: std.Io.Threaded = .init_single_threaded;
    const io = io_backend.io();
    file.setLength(io, @intCast(total_size)) catch return PackageError.IoFailure;
}

const EncodedSection = struct {
    section_type: SectionType,
    flags: u32,
    bytes: []u8,
};

fn encodeStringsSection(allocator: std.mem.Allocator, interner: *const StringInterner) PackageError![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try appendInt(&out, allocator, u32, @intCast(interner.items.items.len));
    for (interner.items.items) |s| {
        try appendInt(&out, allocator, u32, @intCast(s.len));
        try out.appendSlice(allocator, s);
    }
    return out.toOwnedSlice(allocator);
}

fn encodeDimSymbolsSection(allocator: std.mem.Allocator, interner: *const StringInterner, symbols: []const DimSymbol) PackageError![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try appendInt(&out, allocator, u32, @intCast(symbols.len));
    for (symbols) |sym| try appendInt(&out, allocator, u32, interner.get(sym.name).?);
    return out.toOwnedSlice(allocator);
}

fn encodeDimExprsSection(allocator: std.mem.Allocator, exprs: []const DimExpr) PackageError![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try appendInt(&out, allocator, u32, @intCast(exprs.len));
    for (exprs) |expr| {
        switch (expr) {
            .symbol => |sym_idx| {
                try appendInt(&out, allocator, u8, 0);
                try out.appendNTimes(allocator, 0, 3);
                try appendInt(&out, allocator, u32, sym_idx);
                try appendShapeTerm(&out, allocator, .{ .constant = 1 });
                try appendShapeTerm(&out, allocator, .{ .constant = 1 });
            },
            .add => |bin| try appendBinaryDimExpr(&out, allocator, 1, bin),
            .sub => |bin| try appendBinaryDimExpr(&out, allocator, 2, bin),
            .mul => |bin| try appendBinaryDimExpr(&out, allocator, 3, bin),
            .floor_div => |bin| try appendBinaryDimExpr(&out, allocator, 4, bin),
            .max => |bin| try appendBinaryDimExpr(&out, allocator, 5, bin),
        }
    }
    return out.toOwnedSlice(allocator);
}

fn encodeInitializersSection(allocator: std.mem.Allocator, interner: *const StringInterner, initializers: []const Initializer) PackageError![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try appendInt(&out, allocator, u32, @intCast(initializers.len));
    for (initializers) |init| {
        switch (init.encoding) {
            .plain => |dtype| {
                try appendInt(&out, allocator, u8, 0);
                try appendInt(&out, allocator, u8, @intFromEnum(dtype));
                try appendInt(&out, allocator, u8, @intFromEnum(dtype));
                try appendInt(&out, allocator, u8, 0);
                try appendInt(&out, allocator, u32, invalid_index);
                try appendInt(&out, allocator, u32, @intCast(dtype.info().block_elems));
                try appendInt(&out, allocator, u32, @intCast(dtype.info().block_bytes));
                try appendInt(&out, allocator, i32, 0);
                try appendInt(&out, allocator, u32, 0);
                try appendInt(&out, allocator, u64, @intCast(init.data.len));
                try out.appendSlice(allocator, init.data);
            },
            .quantized => |q| {
                try appendInt(&out, allocator, u8, 1);
                try appendInt(&out, allocator, u8, 0);
                try appendInt(&out, allocator, u8, @intFromEnum(q.logical_dtype));
                try appendInt(&out, allocator, u8, 0);
                try appendInt(&out, allocator, u32, interner.get(q.scheme).?);
                try appendInt(&out, allocator, u32, q.block_elems);
                try appendInt(&out, allocator, u32, q.block_bytes);
                try appendInt(&out, allocator, i32, q.quant_axis);
                try appendInt(&out, allocator, u32, @intCast(q.params.len));
                try appendInt(&out, allocator, u64, @intCast(init.data.len));
                try out.appendSlice(allocator, q.params);
                try out.appendSlice(allocator, init.data);
            },
        }
    }
    return out.toOwnedSlice(allocator);
}

fn encodeValuesSection(allocator: std.mem.Allocator, values: []const ValueRecord) PackageError![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try appendInt(&out, allocator, u32, @intCast(values.len));
    for (values) |value| {
        try appendInt(&out, allocator, u8, @intFromEnum(value.dtype));
        try appendInt(&out, allocator, u8, value.rank);
        try appendInt(&out, allocator, u8, @intFromEnum(value.source));
        try appendInt(&out, allocator, u8, 0);
        try appendInt(&out, allocator, u32, value.initializer_index orelse invalid_index);
        try appendInt(&out, allocator, u32, @intCast(value.shape_terms.len));
        for (value.shape_terms) |term| try appendShapeTerm(&out, allocator, term);
    }
    return out.toOwnedSlice(allocator);
}

fn appendNodeRecords(out: *std.ArrayList(u8), allocator: std.mem.Allocator, nodes: []const NodeRecord) PackageError!void {
    try appendInt(out, allocator, u32, @intCast(nodes.len));
    for (nodes) |node| {
        var attr: std.ArrayList(u8) = .empty;
        defer attr.deinit(allocator);
        const kind = try encodeNodeOp(&attr, allocator, node);
        try appendInt(out, allocator, u8, @intFromEnum(kind));
        try out.appendNTimes(allocator, 0, 3);
        try appendInt(out, allocator, u32, node.output);
        try appendInt(out, allocator, u32, @intCast(node.inputs.len));
        try appendInt(out, allocator, u32, @intCast(attr.items.len));
        for (node.inputs) |input| try appendInt(out, allocator, u32, input);
        try out.appendSlice(allocator, attr.items);
    }
}

fn encodeNodesSection(allocator: std.mem.Allocator, nodes: []const NodeRecord) PackageError![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try appendNodeRecords(&out, allocator, nodes);
    return out.toOwnedSlice(allocator);
}

fn encodeRegionsSection(allocator: std.mem.Allocator, regions: []const RegionRecord) PackageError![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try appendInt(&out, allocator, u32, @intCast(regions.len));
    for (regions) |region| {
        try appendNodeRecords(&out, allocator, region.nodes);
        try appendInt(&out, allocator, u32, @intCast(region.outputs.len));
        for (region.outputs) |output| try appendInt(&out, allocator, u32, output);
    }
    return out.toOwnedSlice(allocator);
}

/// One node's attribute blob. Takes the NODE, not just the op: a Loop's extra outputs
/// live in its payload but belong to the node (see `parse.ParsedOp`).
fn encodeNodeOp(out: *std.ArrayList(u8), allocator: std.mem.Allocator, node: NodeRecord) PackageError!NodeOpKind {
    const op = node.op;
    const kind: NodeOpKind = types.nodeOpKind(op);
    switch (op) {
        .MatMul => |mm| {
            try appendInt(out, allocator, f32, mm.alpha);
            try appendInt(out, allocator, f32, mm.beta);
        },
        .ElemwiseBinary => |eb| {
            try appendInt(out, allocator, u8, @intFromEnum(eb.op));
        },
        .Unary => |u| {
            try appendInt(out, allocator, u8, @intFromEnum(u.op));
        },
        .Softmax => |s| {
            try appendInt(out, allocator, i32, s.axis);
        },
        .Conv1D => |cv| {
            try appendSize(out, allocator, cv.stride);
            try appendSize(out, allocator, cv.dilation);
            try appendSize(out, allocator, cv.pad_left);
            try appendSize(out, allocator, cv.pad_right);
            try appendInt(out, allocator, u8, @intFromEnum(cv.pad_mode));
            try appendSize(out, allocator, cv.groups);
        },
        .Conv2D => |cv| {
            try appendSize(out, allocator, cv.stride_h);
            try appendSize(out, allocator, cv.stride_w);
            try appendSize(out, allocator, cv.dilation_h);
            try appendSize(out, allocator, cv.dilation_w);
            try appendSize(out, allocator, cv.pad_top);
            try appendSize(out, allocator, cv.pad_bottom);
            try appendSize(out, allocator, cv.pad_left);
            try appendSize(out, allocator, cv.pad_right);
            try appendInt(out, allocator, u8, @intFromEnum(cv.pad_mode));
            try appendSize(out, allocator, cv.groups);
        },
        .LayerNorm => |ln| {
            try appendInt(out, allocator, f32, ln.eps);
            try appendSymbolicAttr(out, allocator, ln.normalized_shape, ln.free_dims);
        },
        .RMSNorm => |ln| {
            try appendInt(out, allocator, f32, ln.eps);
            try appendSymbolicAttr(out, allocator, ln.normalized_shape, ln.free_dims);
        },
        .Attention => |attn| {
            try appendInt(out, allocator, f32, attn.scale);
            try appendInt(out, allocator, u8, if (attn.causal) 1 else 0);
            try appendSize(out, allocator, attn.sliding_window);
            try appendInt(out, allocator, f32, attn.attn_logits_soft_cap);
            // Packed back into the one control byte the file stores.
            const controls: u8 = (if (attn.has_query_positions) @as(u8, 1) else 0) |
                (if (attn.has_kv_lengths) @as(u8, 2) else 0);
            try appendInt(out, allocator, u8, controls);
        },
        .Reduce => |rr| {
            try appendInt(out, allocator, u8, @intFromEnum(rr.op));
            try appendInt(out, allocator, u8, if (rr.axis != null) 1 else 0);
            if (rr.axis) |axis| try appendInt(out, allocator, i32, axis);
        },
        .Concat => |cc| {
            try appendInt(out, allocator, i32, cc.axis);
        },
        .LSTMCell => |lc| {
            try appendInt(out, allocator, u8, if (lc.has_bias) 1 else 0);
        },
        .RFFT => {},
        .STFT => |st| {
            try appendSize(out, allocator, st.n_fft);
            try appendSize(out, allocator, st.hop_length);
            try appendInt(out, allocator, u8, if (st.center) 1 else 0);
        },
        .RelPosMHA => |attn| {
            try appendInt(out, allocator, f32, attn.scale);
            try appendInt(out, allocator, u8, if (attn.has_mask) 1 else 0);
            try appendSize(out, allocator, attn.chunk_size);
            try appendSize(out, allocator, attn.chunk_left);
        },
        .ArgMax => |am| {
            try appendInt(out, allocator, i32, am.axis);
        },
        .ScatterRow => {},
        .Gather => |gg| {
            try appendInt(out, allocator, i32, gg.axis);
            try appendSize(out, allocator, gg.batch_dims);
        },
        .Dim => |dd| try appendInt(out, allocator, i32, dd.axis),
        .Iota => |io| try appendInt(out, allocator, i32, io.axis),
        .Copy => {},
        .RoPE1D => |rp| {
            try appendInt(out, allocator, f32, rp.base_frequency);
            try appendInt(out, allocator, f32, rp.scale_factor);
            try appendInt(out, allocator, f32, rp.rope_proportion);
        },
        .SequenceAppend => {},
        .Cast => |ct| {
            try appendInt(out, allocator, u8, @intFromEnum(ct.to_dtype));
        },
        .MatMulNT => |mm| {
            try appendInt(out, allocator, f32, mm.alpha);
            try appendInt(out, allocator, f32, mm.beta);
        },
        .If => |iff| {
            try appendInt(out, allocator, u32, iff.then_region);
            try appendInt(out, allocator, u32, iff.else_region);
        },
        .Loop => |lp| {
            try appendInt(out, allocator, u32, lp.body_region);
            try appendSize(out, allocator, lp.static_max_trip_count);
            try appendInt(out, allocator, i32, if (lp.cond_carry) |c| @intCast(c) else -1);
            try appendInt(out, allocator, u8, @intFromBool(lp.check_before));
            try appendInt(out, allocator, u32, @intCast(node.extra_outputs.len));
            for (node.extra_outputs) |e| try appendInt(out, allocator, u32, e);
        },
        .ViewReshape => |vr| {
            try appendSymbolicAttr(out, allocator, vr.new_shape, vr.free_dims);
        },
        .ViewSqueeze => |vs| {
            try appendInt(out, allocator, u8, if (vs.axis != null) 1 else 0);
            if (vs.axis) |axis| try appendInt(out, allocator, i32, axis);
        },
        .ViewUnsqueeze => |vu| {
            try appendInt(out, allocator, i32, vu.axis);
        },
        .ViewTranspose2D => {},
        .ViewSliceND => |sl| {
            try appendSizeArray(out, allocator, sl.starts);
            try appendSymbolicAttr(out, allocator, sl.lens, sl.free_dims);
        },
    }

    return kind;
}

fn encodeSignaturesSection(allocator: std.mem.Allocator, interner: *const StringInterner, inputs: []const NamedValue, outputs: []const NamedValue) PackageError![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try appendInt(&out, allocator, u32, @intCast(inputs.len));
    try appendInt(&out, allocator, u32, @intCast(outputs.len));
    try appendNamedValues(&out, allocator, interner, inputs);
    try appendNamedValues(&out, allocator, interner, outputs);
    return out.toOwnedSlice(allocator);
}

fn encodeMetadataSection(allocator: std.mem.Allocator, interner: *const StringInterner, metadata: []const MetadataEntry) PackageError![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try appendInt(&out, allocator, u32, @intCast(metadata.len));
    for (metadata) |entry| {
        try appendInt(&out, allocator, u32, interner.get(entry.key).?);
        try appendInt(&out, allocator, u32, interner.get(entry.value).?);
    }
    return out.toOwnedSlice(allocator);
}

fn encodeDebugNamesSection(allocator: std.mem.Allocator, interner: *const StringInterner, debug_names: []const DebugName) PackageError![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try appendInt(&out, allocator, u32, @intCast(debug_names.len));
    for (debug_names) |entry| {
        try appendInt(&out, allocator, u32, entry.value);
        try appendInt(&out, allocator, u32, interner.get(entry.name).?);
    }
    return out.toOwnedSlice(allocator);
}

fn encodeIoAliasesSection(allocator: std.mem.Allocator, io_aliases: []const IoAlias) PackageError![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try appendInt(&out, allocator, u32, @intCast(io_aliases.len));
    for (io_aliases) |alias| {
        try appendInt(&out, allocator, u32, alias.input);
        try appendInt(&out, allocator, u32, alias.output);
    }
    return out.toOwnedSlice(allocator);
}

fn encodeInputRolesSection(allocator: std.mem.Allocator, input_roles: []const types.InputRole) PackageError![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try appendInt(&out, allocator, u32, @intCast(input_roles.len));
    for (input_roles) |role| {
        try appendInt(&out, allocator, u32, role.input);
        try appendInt(&out, allocator, u8, @intFromEnum(role.kind));
        try appendInt(&out, allocator, u8, role.axis);
        try appendInt(&out, allocator, u8, role.flags);
        try appendInt(&out, allocator, u8, 0); // reserved
        try appendInt(&out, allocator, u32, role.capacity_symbol);
    }
    return out.toOwnedSlice(allocator);
}

fn encodeGraphMetaSection(allocator: std.mem.Allocator, meta: GraphMeta) PackageError![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);
    try appendInt(&out, allocator, u32, meta.value_count);
    try appendInt(&out, allocator, u32, meta.node_count);
    try appendInt(&out, allocator, u32, meta.region_count);
    try appendInt(&out, allocator, u32, meta.initializer_count);
    try appendInt(&out, allocator, u32, meta.input_count);
    try appendInt(&out, allocator, u32, meta.output_count);
    try appendInt(&out, allocator, u32, meta.dim_symbol_count);
    try appendInt(&out, allocator, u32, meta.dim_expr_count);
    try appendInt(&out, allocator, u32, meta.metadata_count);
    try appendInt(&out, allocator, u32, meta.debug_name_count);
    try appendInt(&out, allocator, u32, meta.io_alias_count);
    return out.toOwnedSlice(allocator);
}

fn appendNamedValues(out: *std.ArrayList(u8), allocator: std.mem.Allocator, interner: *const StringInterner, values: []const NamedValue) PackageError!void {
    for (values) |sig| {
        try appendInt(out, allocator, u32, interner.get(sig.name).?);
        try appendInt(out, allocator, u32, sig.value);
    }
}

fn appendBinaryDimExpr(out: *std.ArrayList(u8), allocator: std.mem.Allocator, kind: u8, bin: BinaryDimExpr) PackageError!void {
    try appendInt(out, allocator, u8, kind);
    try out.appendNTimes(allocator, 0, 3);
    try appendInt(out, allocator, u32, 0);
    try appendShapeTerm(out, allocator, bin.lhs);
    try appendShapeTerm(out, allocator, bin.rhs);
}

fn appendShapeTerm(out: *std.ArrayList(u8), allocator: std.mem.Allocator, term: ShapeTerm) PackageError!void {
    switch (term) {
        .constant => |value| {
            try appendInt(out, allocator, u8, 0);
            try out.appendNTimes(allocator, 0, 7);
            try appendInt(out, allocator, u64, value);
        },
        .expr => |expr_idx| {
            try appendInt(out, allocator, u8, 1);
            try out.appendNTimes(allocator, 0, 7);
            try appendInt(out, allocator, u64, expr_idx);
        },
    }
}

/// An op's shape attribute, written back as the `ShapeTerm` list the file stores: a free
/// axis becomes its expression index, a fixed one its size. The inverse of
/// `parse.readSymbolicAttr`.
fn appendSymbolicAttr(
    out: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    sizes: []const usize,
    free_dims: []const ?u32,
) !void {
    try appendInt(out, allocator, u32, @intCast(sizes.len));
    for (sizes, 0..) |size, i| {
        const free_dim: ?u32 = if (i < free_dims.len) free_dims[i] else null;
        if (free_dim) |expr| {
            try appendShapeTerm(out, allocator, .{ .expr = expr });
        } else {
            try appendShapeTerm(out, allocator, .{ .constant = @intCast(size) });
        }
    }
}

/// A `usize` field, stored as u64.
fn appendSize(out: *std.ArrayList(u8), allocator: std.mem.Allocator, value: usize) PackageError!void {
    return appendInt(out, allocator, u64, @intCast(value));
}

fn appendSizeArray(out: *std.ArrayList(u8), allocator: std.mem.Allocator, values: []const usize) PackageError!void {
    try appendInt(out, allocator, u32, @intCast(values.len));
    for (values) |value| try appendSize(out, allocator, value);
}



fn collectPackageStrings(interner: *StringInterner, pkg: *const Package) PackageError!void {
    for (pkg.initializers) |init| {
        switch (init.encoding) {
            .plain => {},
            .quantized => |q| {
                _ = try interner.put(q.scheme);
            },
        }
    }
    for (pkg.inputs) |sig| _ = try interner.put(sig.name);
    for (pkg.outputs) |sig| _ = try interner.put(sig.name);
    for (pkg.dim_symbols) |sym| _ = try interner.put(sym.name);
    for (pkg.metadata) |entry| {
        _ = try interner.put(entry.key);
        _ = try interner.put(entry.value);
    }
    for (pkg.debug_names) |entry| _ = try interner.put(entry.name);
}

fn writeHeader(writer: *WriteCursor, section_count: u32, dir_offset: u64, file_size: u64) PackageError!void {
    try writer.writeAll(&magic_bytes);
    try writeInt(writer, u32, current_version);
    try writeInt(writer, u32, section_count);
    try writeInt(writer, u32, 0);
    try writeInt(writer, u64, dir_offset);
    try writeInt(writer, u64, file_size);
    try writeInt(writer, u64, 0);
    var reserved: [32]u8 = @splat(0);
    try writer.writeAll(&reserved);
}

fn writeSectionDesc(writer: *WriteCursor, section_type: SectionType, flags: u32, offset: u64, size: u64) PackageError!void {
    try writeInt(writer, u32, @intFromEnum(section_type));
    try writeInt(writer, u32, flags);
    try writeInt(writer, u64, offset);
    try writeInt(writer, u64, size);
}

fn appendInt(out: *std.ArrayList(u8), allocator: std.mem.Allocator, comptime T: type, value: T) PackageError!void {
    var buf: [@sizeOf(T)]u8 = undefined;
    switch (@typeInfo(T)) {
        .int => std.mem.writeInt(T, &buf, value, .little),
        .float => |float_info| {
            const IntT = @Int(.unsigned, float_info.bits);
            const bits: IntT = @bitCast(value);
            std.mem.writeInt(IntT, @ptrCast(&buf), bits, .little);
        },
        else => @compileError("appendInt only supports integer and float types"),
    }
    out.appendSlice(allocator, &buf) catch return PackageError.OutOfMemory;
}

fn writeInt(writer: *WriteCursor, comptime T: type, value: T) PackageError!void {
    var buf: [@sizeOf(T)]u8 = undefined;
    std.mem.writeInt(T, &buf, value, .little);
    try writer.writeAll(&buf);
}

const WriteCursor = struct {
    file: std.Io.File,
    offset: u64 = 0,

    fn writeAll(self: *WriteCursor, bytes: []const u8) PackageError!void {
        var io_backend: std.Io.Threaded = .init_single_threaded;
        const io = io_backend.io();
        self.file.writePositionalAll(io, bytes, self.offset) catch return PackageError.IoFailure;
        self.offset += @intCast(bytes.len);
    }
};

const StringInterner = struct {
    allocator: std.mem.Allocator,
    map: std.StringHashMapUnmanaged(u32) = .{},
    items: std.ArrayList([]const u8) = .empty,

    fn init(allocator: std.mem.Allocator) StringInterner {
        return .{ .allocator = allocator };
    }

    fn put(self: *StringInterner, value: []const u8) PackageError!u32 {
        if (self.map.get(value)) |idx| return idx;
        const duped = self.allocator.dupe(u8, value) catch return PackageError.OutOfMemory;
        const idx: u32 = @intCast(self.items.items.len);
        self.items.append(self.allocator, duped) catch return PackageError.OutOfMemory;
        self.map.put(self.allocator, duped, idx) catch return PackageError.OutOfMemory;
        return idx;
    }

    fn get(self: *const StringInterner, value: []const u8) ?u32 {
        return self.map.get(value);
    }
};
