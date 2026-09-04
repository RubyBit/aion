// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const types = @import("types.zig");
const graph_mod = @import("../../graph/graph.zig");

const DType = types.DType;
const ElemwiseBinaryOp = types.ElemwiseBinaryOp;
const UnaryOp = types.UnaryOp;
const ReduceOp = types.ReduceOp;
const PadMode = types.PadMode;
const magic_bytes = types.magic_bytes;
const current_version = types.current_version;
const header_size = types.header_size;
const invalid_index = types.invalid_index;
const section_slot_count = types.section_slot_count;
const PackageError = types.PackageError;
const SectionType = types.SectionType;
const SectionFlags = types.SectionFlags;
const ShapeTerm = types.ShapeTerm;
const DimExpr = types.DimExpr;
const DimSymbol = types.DimSymbol;
const Initializer = types.Initializer;
const ValueSource = types.ValueSource;
const ValueRecord = types.ValueRecord;
const NamedValue = types.NamedValue;
const MetadataEntry = types.MetadataEntry;
const DebugName = types.DebugName;
const IoAlias = types.IoAlias;
const NodeOpKind = types.NodeOpKind;
const NodeOp = types.NodeOp;
const NodeRecord = types.NodeRecord;
const RegionRecord = types.RegionRecord;
const GraphMeta = types.GraphMeta;
const Package = types.Package;

/// Parse a `.aion` file and return a `Package`. The returned Package owns its data
/// (slices are copied out of `bytes`), so the caller remains responsible for the buffer.
pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) PackageError!Package {
    return parseImpl(allocator, bytes, null);
}

/// Parse and transfer ownership of `bytes` into the returned Package.
///
/// Large-initializer slices (notably `Initializer.data` and `QuantizedEncoding.params`)
/// borrow directly into `bytes` rather than being copied out; `Package.deinit` frees
/// `bytes` after tearing down its other owned allocations. Use this when loading
/// multi-GB packages to avoid a peak-memory doubling during parse.
///
/// Ownership is transferred unconditionally: on error `bytes` is freed inside the
/// implementation before the error returns, so the caller must not `free(bytes)`
/// themselves after calling this.
pub fn parseTakeOwned(allocator: std.mem.Allocator, bytes: []u8) PackageError!Package {
    return parseImpl(allocator, bytes, bytes);
}

fn parseImpl(allocator: std.mem.Allocator, bytes: []const u8, source_bytes: ?[]u8) PackageError!Package {
    // If the caller transferred bytes to us, we must free them on any error path that
    // returns before the Package takes ownership. `pending_bytes` tracks "do we still
    // owe the caller that free?"; we null it once the Package holds the bytes and its
    // own deinit takes over.
    var pending_bytes: ?[]u8 = source_bytes;
    errdefer if (pending_bytes) |b| allocator.free(b);

    var pkg: Package = undefined;
    var graph_meta: GraphMeta = undefined;

    // Keep all section-level cleanup (errdefer) inside a scope that ends *before*
    // `pkg.validate()` runs. After the Package is constructed, only `pkg.deinit()`
    // should own teardown on validation errors.
    {
        try validateHeader(bytes);

        var cursor: usize = 4;
        _ = try readIntCursor(bytes, &cursor, u32); // version
        const section_count: usize = std.math.cast(usize, try readIntCursor(bytes, &cursor, u32)) orelse return PackageError.InvalidFormat;
        _ = try readIntCursor(bytes, &cursor, u32); // reserved
        const dir_offset: usize = std.math.cast(usize, try readIntCursor(bytes, &cursor, u64)) orelse return PackageError.InvalidFormat;
        const file_size: usize = std.math.cast(usize, try readIntCursor(bytes, &cursor, u64)) orelse return PackageError.InvalidFormat;
        _ = try readIntCursor(bytes, &cursor, u64); // flags
        _ = try readBytes(bytes, &cursor, 32);
        if (file_size != bytes.len) return PackageError.InvalidFormat;

        var refs: [section_slot_count]?SectionRef = @splat(null);
        try readSectionDirectory(bytes, dir_offset, section_count, &refs);
        try ensureRequiredSections(refs);

        const strings = try parseStringsSection(allocator, sectionBytes(bytes, refs[sectionSlot(.strings)].?));
        defer freeStringTable(allocator, strings);

        const dim_symbols = if (refs[sectionSlot(.dim_symbols)]) |ref|
            try parseDimSymbolsSection(allocator, strings, sectionBytes(bytes, ref))
        else
            try allocator.alloc(DimSymbol, 0);
        errdefer freeDimSymbols(allocator, dim_symbols);

        const dim_exprs = if (refs[sectionSlot(.dim_exprs)]) |ref|
            try parseDimExprsSection(allocator, sectionBytes(bytes, ref))
        else
            try allocator.alloc(DimExpr, 0);
        errdefer allocator.free(dim_exprs);

        const initializers = try parseInitializersSection(allocator, strings, sectionBytes(bytes, refs[sectionSlot(.tensors)].?), source_bytes != null);
        errdefer freeInitializers(allocator, initializers);

        const values = try parseValuesSection(allocator, sectionBytes(bytes, refs[sectionSlot(.values)].?));
        errdefer freeValues(allocator, values);

        const nodes = try parseNodesSection(allocator, sectionBytes(bytes, refs[sectionSlot(.nodes)].?));
        errdefer freeNodes(allocator, nodes);

        const regions = if (refs[sectionSlot(.regions)]) |ref|
            try parseRegionsSection(allocator, sectionBytes(bytes, ref))
        else
            try allocator.alloc(RegionRecord, 0);
        errdefer freeRegions(allocator, regions);

        const signatures = try parseSignaturesSection(allocator, strings, sectionBytes(bytes, refs[sectionSlot(.signatures)].?));
        errdefer {
            freeNamedValues(allocator, signatures.inputs);
            freeNamedValues(allocator, signatures.outputs);
        }

        const metadata = if (refs[sectionSlot(.metadata)]) |ref|
            try parseMetadataSection(allocator, strings, sectionBytes(bytes, ref))
        else
            try allocator.alloc(MetadataEntry, 0);
        errdefer freeMetadata(allocator, metadata);

        const debug_names = if (refs[sectionSlot(.debug_names)]) |ref|
            try parseDebugNamesSection(allocator, strings, sectionBytes(bytes, ref))
        else
            try allocator.alloc(DebugName, 0);
        errdefer freeDebugNames(allocator, debug_names);

        const io_aliases = if (refs[sectionSlot(.io_aliases)]) |ref|
            try parseIoAliasesSection(allocator, sectionBytes(bytes, ref))
        else
            try allocator.alloc(IoAlias, 0);
        errdefer allocator.free(io_aliases);

        const input_roles = if (refs[sectionSlot(.input_roles)]) |ref|
            try parseInputRolesSection(allocator, sectionBytes(bytes, ref))
        else
            try allocator.alloc(types.InputRole, 0);
        errdefer allocator.free(input_roles);

        graph_meta = try parseGraphMetaSection(sectionBytes(bytes, refs[sectionSlot(.graph_meta)].?));

        pkg = Package{
            .allocator = allocator,
            .initializers = initializers,
            .values = values,
            .nodes = nodes,
            .regions = regions,
            .inputs = signatures.inputs,
            .outputs = signatures.outputs,
            .dim_symbols = dim_symbols,
            .dim_exprs = dim_exprs,
            .metadata = metadata,
            .debug_names = debug_names,
            .io_aliases = io_aliases,
            .input_roles = input_roles,
            .source_bytes = source_bytes,
        };

        // From this point, the Package owns all allocations above (and optionally
        // `source_bytes`). Any later error should be handled by `pkg.deinit()`.
        pending_bytes = null;
    }

    errdefer pkg.deinit();
    try pkg.validate();
    try validateGraphMeta(&pkg, graph_meta);
    return pkg;
}

pub fn validate(pkg: *const Package) PackageError!void {
    try pkg.validate();
}

fn validateHeader(bytes: []const u8) PackageError!void {
    if (bytes.len < header_size) return PackageError.InvalidFormat;
    if (!std.mem.eql(u8, bytes[0..4], &magic_bytes)) return PackageError.InvalidFormat;
    const version = std.mem.readInt(u32, bytes[4..8], .little);
    if (version != current_version) return PackageError.UnsupportedVersion;
}

fn readSectionDirectory(bytes: []const u8, dir_offset: usize, section_count: usize, refs: *[section_slot_count]?SectionRef) PackageError!void {
    var cursor = dir_offset;
    var seen_required: u32 = 0;
    var i: usize = 0;
    while (i < section_count) : (i += 1) {
        const type_raw = try readIntCursor(bytes, &cursor, u32);
        const flags = try readIntCursor(bytes, &cursor, u32);
        const offset = std.math.cast(usize, try readIntCursor(bytes, &cursor, u64)) orelse return PackageError.InvalidFormat;
        const size = std.math.cast(usize, try readIntCursor(bytes, &cursor, u64)) orelse return PackageError.InvalidFormat;
        _ = try sliceAt(bytes, offset, size);

        const section_type = std.enums.fromInt(SectionType, type_raw) orelse {
            if ((flags & SectionFlags.required) != 0) return PackageError.UnsupportedFeature;
            continue;
        };
        const slot = sectionSlot(section_type);
        if (refs[slot] != null) return PackageError.InvalidFormat;
        refs[slot] = .{ .section_type = section_type, .flags = flags, .offset = offset, .size = size };
        if ((flags & SectionFlags.required) != 0) seen_required |= requiredBit(section_type);
    }
    if (seen_required != allRequiredBits()) return PackageError.InvalidFormat;
}

fn ensureRequiredSections(refs: [section_slot_count]?SectionRef) PackageError!void {
    if (refs[sectionSlot(.strings)] == null) return PackageError.InvalidFormat;
    if (refs[sectionSlot(.tensors)] == null) return PackageError.InvalidFormat;
    if (refs[sectionSlot(.values)] == null) return PackageError.InvalidFormat;
    if (refs[sectionSlot(.nodes)] == null) return PackageError.InvalidFormat;
    if (refs[sectionSlot(.signatures)] == null) return PackageError.InvalidFormat;
    if (refs[sectionSlot(.graph_meta)] == null) return PackageError.InvalidFormat;
}

fn sectionBytes(bytes: []const u8, ref: SectionRef) []const u8 {
    return bytes[ref.offset .. ref.offset + ref.size];
}

fn validateGraphMeta(pkg: *const Package, meta: GraphMeta) PackageError!void {
    const want = pkg.graphMeta();
    if (want.value_count != meta.value_count) return PackageError.InvalidFormat;
    if (want.node_count != meta.node_count) return PackageError.InvalidFormat;
    if (want.region_count != meta.region_count) return PackageError.InvalidFormat;
    if (want.initializer_count != meta.initializer_count) return PackageError.InvalidFormat;
    if (want.input_count != meta.input_count) return PackageError.InvalidFormat;
    if (want.output_count != meta.output_count) return PackageError.InvalidFormat;
    if (want.dim_symbol_count != meta.dim_symbol_count) return PackageError.InvalidFormat;
    if (want.dim_expr_count != meta.dim_expr_count) return PackageError.InvalidFormat;
    if (want.metadata_count != meta.metadata_count) return PackageError.InvalidFormat;
    if (want.debug_name_count != meta.debug_name_count) return PackageError.InvalidFormat;
    if (want.io_alias_count != meta.io_alias_count) return PackageError.InvalidFormat;
}

fn sectionSlot(section_type: SectionType) usize {
    return switch (section_type) {
        .strings => 0,
        .tensors => 1,
        .values => 2,
        .nodes => 3,
        .signatures => 4,
        .graph_meta => 5,
        .dim_symbols => 6,
        .dim_exprs => 7,
        .metadata => 8,
        .debug_names => 9,
        .regions => 10,
        .io_aliases => 11,
        .input_roles => 12,
    };
}

fn requiredBit(section_type: SectionType) u32 {
    return switch (section_type) {
        .strings => 1 << 0,
        .tensors => 1 << 1,
        .values => 1 << 2,
        .nodes => 1 << 3,
        .signatures => 1 << 4,
        .graph_meta => 1 << 5,
        else => 0,
    };
}

fn allRequiredBits() u32 {
    return requiredBit(.strings) |
        requiredBit(.tensors) |
        requiredBit(.values) |
        requiredBit(.nodes) |
        requiredBit(.signatures) |
        requiredBit(.graph_meta);
}

const SectionRef = struct {
    section_type: SectionType,
    flags: u32,
    offset: usize,
    size: usize,
};

fn parseStringsSection(allocator: std.mem.Allocator, bytes: []const u8) PackageError![][]u8 {
    var cursor: usize = 0;
    const count = std.math.cast(usize, try readIntCursor(bytes, &cursor, u32)) orelse return PackageError.InvalidFormat;
    const out = allocator.alloc([]u8, count) catch return PackageError.OutOfMemory;
    errdefer allocator.free(out);
    for (out) |*slot| {
        const len = std.math.cast(usize, try readIntCursor(bytes, &cursor, u32)) orelse return PackageError.InvalidFormat;
        const raw = try readBytes(bytes, &cursor, len);
        slot.* = allocator.dupe(u8, raw) catch return PackageError.OutOfMemory;
    }
    if (cursor != bytes.len) return PackageError.InvalidFormat;
    return out;
}

fn freeStringTable(allocator: std.mem.Allocator, strings: [][]u8) void {
    for (strings) |s| allocator.free(s);
    allocator.free(strings);
}

fn parseDimSymbolsSection(allocator: std.mem.Allocator, strings: [][]u8, bytes: []const u8) PackageError![]DimSymbol {
    var cursor: usize = 0;
    const count = std.math.cast(usize, try readIntCursor(bytes, &cursor, u32)) orelse return PackageError.InvalidFormat;
    const out = allocator.alloc(DimSymbol, count) catch return PackageError.OutOfMemory;
    errdefer allocator.free(out);
    for (out) |*slot| {
        const idx = try readStringIndex(bytes, &cursor, strings.len);
        slot.* = .{ .name = allocator.dupe(u8, strings[idx]) catch return PackageError.OutOfMemory };
    }
    if (cursor != bytes.len) return PackageError.InvalidFormat;
    return out;
}

fn parseDimExprsSection(allocator: std.mem.Allocator, bytes: []const u8) PackageError![]DimExpr {
    var cursor: usize = 0;
    const count = std.math.cast(usize, try readIntCursor(bytes, &cursor, u32)) orelse return PackageError.InvalidFormat;
    const out = allocator.alloc(DimExpr, count) catch return PackageError.OutOfMemory;
    errdefer allocator.free(out);
    for (out) |*slot| {
        const kind = try readIntCursor(bytes, &cursor, u8);
        _ = try readBytes(bytes, &cursor, 3);
        const aux = try readIntCursor(bytes, &cursor, u32);
        const lhs = try readShapeTerm(bytes, &cursor);
        const rhs = try readShapeTerm(bytes, &cursor);
        slot.* = switch (kind) {
            0 => .{ .symbol = aux },
            1 => .{ .add = .{ .lhs = lhs, .rhs = rhs } },
            2 => .{ .sub = .{ .lhs = lhs, .rhs = rhs } },
            3 => .{ .mul = .{ .lhs = lhs, .rhs = rhs } },
            4 => .{ .floor_div = .{ .lhs = lhs, .rhs = rhs } },
            5 => .{ .max = .{ .lhs = lhs, .rhs = rhs } },
            else => return PackageError.InvalidFormat,
        };
    }
    if (cursor != bytes.len) return PackageError.InvalidFormat;
    return out;
}

fn parseInitializersSection(
    allocator: std.mem.Allocator,
    strings: [][]u8,
    bytes: []const u8,
    borrow_data: bool,
) PackageError![]Initializer {
    var cursor: usize = 0;
    const count = std.math.cast(usize, try readIntCursor(bytes, &cursor, u32)) orelse return PackageError.InvalidFormat;
    const out = allocator.alloc(Initializer, count) catch return PackageError.OutOfMemory;
    errdefer allocator.free(out);
    for (out) |*slot| {
        const kind = try readIntCursor(bytes, &cursor, u8);
        const plain_dtype_raw = try readIntCursor(bytes, &cursor, u8);
        const logical_dtype_raw = try readIntCursor(bytes, &cursor, u8);
        _ = try readIntCursor(bytes, &cursor, u8);
        const scheme_idx = try readIntCursor(bytes, &cursor, u32);
        const block_elems = try readIntCursor(bytes, &cursor, u32);
        const block_bytes = try readIntCursor(bytes, &cursor, u32);
        const quant_axis = try readIntCursor(bytes, &cursor, i32);
        const params_len = std.math.cast(usize, try readIntCursor(bytes, &cursor, u32)) orelse return PackageError.InvalidFormat;
        const data_len = std.math.cast(usize, try readIntCursor(bytes, &cursor, u64)) orelse return PackageError.InvalidFormat;
        const params_raw = try readBytes(bytes, &cursor, params_len);
        const data_raw = try readBytes(bytes, &cursor, data_len);

        // `data` and (for quantized kinds) `params` are the largest per-initializer
        // payloads; when the caller has transferred ownership of the file bytes to the
        // Package, these can be zero-copy borrows instead of duplicated allocations.
        const data: []u8 = if (borrow_data)
            @constCast(data_raw)
        else
            allocator.dupe(u8, data_raw) catch return PackageError.OutOfMemory;
        errdefer if (!borrow_data) allocator.free(data);

        slot.* = switch (kind) {
            0 => .{ .encoding = .{ .plain = std.enums.fromInt(DType, plain_dtype_raw) orelse return PackageError.InvalidFormat }, .data = data },
            1 => blk: {
                if (scheme_idx == invalid_index or scheme_idx >= strings.len) return PackageError.InvalidFormat;
                const scheme = allocator.dupe(u8, strings[scheme_idx]) catch return PackageError.OutOfMemory;
                errdefer allocator.free(scheme);
                const params: []u8 = if (borrow_data)
                    @constCast(params_raw)
                else
                    allocator.dupe(u8, params_raw) catch return PackageError.OutOfMemory;
                errdefer if (!borrow_data) allocator.free(params);
                break :blk .{
                    .encoding = .{ .quantized = .{
                        .scheme = scheme,
                        .logical_dtype = std.enums.fromInt(DType, logical_dtype_raw) orelse return PackageError.InvalidFormat,
                        .block_elems = block_elems,
                        .block_bytes = block_bytes,
                        .quant_axis = quant_axis,
                        .params = params,
                    } },
                    .data = data,
                };
            },
            else => return PackageError.InvalidFormat,
        };
    }
    if (cursor != bytes.len) return PackageError.InvalidFormat;
    return out;
}

fn parseValuesSection(allocator: std.mem.Allocator, bytes: []const u8) PackageError![]ValueRecord {
    var cursor: usize = 0;
    const count = std.math.cast(usize, try readIntCursor(bytes, &cursor, u32)) orelse return PackageError.InvalidFormat;
    const out = allocator.alloc(ValueRecord, count) catch return PackageError.OutOfMemory;
    errdefer allocator.free(out);
    for (out) |*slot| {
        const dtype = std.enums.fromInt(DType, try readIntCursor(bytes, &cursor, u8)) orelse return PackageError.InvalidFormat;
        const rank = try readIntCursor(bytes, &cursor, u8);
        const source = std.enums.fromInt(ValueSource, try readIntCursor(bytes, &cursor, u8)) orelse return PackageError.InvalidFormat;
        _ = try readIntCursor(bytes, &cursor, u8);
        const init_index_raw = try readIntCursor(bytes, &cursor, u32);
        const shape_term_count = std.math.cast(usize, try readIntCursor(bytes, &cursor, u32)) orelse return PackageError.InvalidFormat;
        const shape_terms = allocator.alloc(ShapeTerm, shape_term_count) catch return PackageError.OutOfMemory;
        errdefer allocator.free(shape_terms);
        for (shape_terms) |*term| term.* = try readShapeTerm(bytes, &cursor);
        slot.* = .{
            .dtype = dtype,
            .rank = rank,
            .source = source,
            .shape_terms = shape_terms,
            .initializer_index = if (init_index_raw == invalid_index) null else init_index_raw,
        };
    }
    if (cursor != bytes.len) return PackageError.InvalidFormat;
    return out;
}

fn parseNodeRecords(allocator: std.mem.Allocator, bytes: []const u8, cursor: *usize) PackageError![]NodeRecord {
    const count = std.math.cast(usize, try readIntCursor(bytes, cursor, u32)) orelse return PackageError.InvalidFormat;
    const out = allocator.alloc(NodeRecord, count) catch return PackageError.OutOfMemory;
    errdefer allocator.free(out);
    for (out) |*slot| {
        const op_kind_raw = try readIntCursor(bytes, cursor, u8);
        _ = try readBytes(bytes, cursor, 3);
        const output = try readIntCursor(bytes, cursor, u32);
        const input_count = std.math.cast(usize, try readIntCursor(bytes, cursor, u32)) orelse return PackageError.InvalidFormat;
        const attr_len = std.math.cast(usize, try readIntCursor(bytes, cursor, u32)) orelse return PackageError.InvalidFormat;
        const inputs = allocator.alloc(u32, input_count) catch return PackageError.OutOfMemory;
        errdefer allocator.free(inputs);
        for (inputs) |*input| input.* = try readIntCursor(bytes, cursor, u32);
        const attr_bytes = try readBytes(bytes, cursor, attr_len);
        const op_kind: NodeOpKind = types.parseNodeOpKind(op_kind_raw) orelse return PackageError.InvalidFormat;
        const parsed = try parseNodeOp(allocator, op_kind, attr_bytes);
        errdefer types.deinitNodeOp(allocator, parsed.op);
        errdefer if (parsed.extra_outputs.len != 0) allocator.free(parsed.extra_outputs);
        slot.* = .{
            .inputs = inputs,
            .output = output,
            .op = parsed.op,
            .extra_outputs = parsed.extra_outputs,
        };
    }
    return out;
}

fn parseNodesSection(allocator: std.mem.Allocator, bytes: []const u8) PackageError![]NodeRecord {
    var cursor: usize = 0;
    const out = try parseNodeRecords(allocator, bytes, &cursor);
    if (cursor != bytes.len) {
        freeNodes(allocator, out);
        return PackageError.InvalidFormat;
    }
    return out;
}

fn parseRegionsSection(allocator: std.mem.Allocator, bytes: []const u8) PackageError![]RegionRecord {
    var cursor: usize = 0;
    const count = std.math.cast(usize, try readIntCursor(bytes, &cursor, u32)) orelse return PackageError.InvalidFormat;
    const out = allocator.alloc(RegionRecord, count) catch return PackageError.OutOfMemory;
    errdefer allocator.free(out);
    var initialized: usize = 0;
    errdefer {
        for (out[0..initialized]) |region| {
            freeNodes(allocator, region.nodes);
            allocator.free(region.outputs);
        }
    }
    for (out) |*region| {
        const nodes = try parseNodeRecords(allocator, bytes, &cursor);
        var nodes_owned: bool = true;
        errdefer if (nodes_owned) freeNodes(allocator, nodes);
        const output_count = std.math.cast(usize, try readIntCursor(bytes, &cursor, u32)) orelse return PackageError.InvalidFormat;
        const outputs = allocator.alloc(u32, output_count) catch return PackageError.OutOfMemory;
        var outputs_owned: bool = true;
        errdefer if (outputs_owned) allocator.free(outputs);
        for (outputs) |*output| output.* = try readIntCursor(bytes, &cursor, u32);
        region.* = .{ .nodes = nodes, .outputs = outputs };
        nodes_owned = false;
        outputs_owned = false;
        initialized += 1;
    }
    if (cursor != bytes.len) return PackageError.InvalidFormat;
    return out;
}

/// One node's attribute blob. A Loop's extra outputs are stored in it, but they belong to
/// the NODE (`graph.Node.extra_outputs`), so they come back alongside the op.
const ParsedOp = struct { op: NodeOp, extra_outputs: []const u32 = &.{} };

fn parseNodeOp(allocator: std.mem.Allocator, kind: NodeOpKind, bytes: []const u8) PackageError!ParsedOp {
    var cursor: usize = 0;
    var extra_outputs: []const u32 = &.{};
    const op: NodeOp = switch (kind) {
        .MatMul => .{ .MatMul = .{ .alpha = try readIntCursor(bytes, &cursor, f32), .beta = try readIntCursor(bytes, &cursor, f32) } },
        .ElemwiseBinary => .{ .ElemwiseBinary = .{ .op = try readEnumCursor(bytes, &cursor, ElemwiseBinaryOp) } },
        .Unary => .{ .Unary = .{ .op = try readEnumCursor(bytes, &cursor, UnaryOp) } },
        .Softmax => .{ .Softmax = .{ .axis = try readIntCursor(bytes, &cursor, i32) } },
        .If => .{ .If = .{ .then_region = try readIntCursor(bytes, &cursor, u32), .else_region = try readIntCursor(bytes, &cursor, u32) } },
        .Loop => blk: {
            const body_region = try readIntCursor(bytes, &cursor, u32);
            const static_max = try readSizeCursor(bytes, &cursor);
            const cond_raw = try readIntCursor(bytes, &cursor, i32);
            const check_before = (try readIntCursor(bytes, &cursor, u8)) != 0;
            const n_extra = std.math.cast(usize, try readIntCursor(bytes, &cursor, u32)) orelse return PackageError.InvalidFormat;
            const extra: []const u32 = ex: {
                if (n_extra == 0) break :ex &[_]u32{};
                const buf = allocator.alloc(u32, n_extra) catch return PackageError.OutOfMemory;
                errdefer allocator.free(buf);
                for (buf) |*e| e.* = try readIntCursor(bytes, &cursor, u32);
                break :ex buf;
            };
            extra_outputs = extra;
            break :blk .{ .Loop = .{
                .body_region = body_region,
                .static_max_trip_count = static_max,
                .cond_carry = if (cond_raw < 0) null else @intCast(cond_raw),
                .check_before = check_before,
            } };
        },
        .Conv1D => .{ .Conv1D = .{
            .stride = try readSizeCursor(bytes, &cursor),
            .dilation = try readSizeCursor(bytes, &cursor),
            .pad_left = try readSizeCursor(bytes, &cursor),
            .pad_right = try readSizeCursor(bytes, &cursor),
            .pad_mode = try readEnumCursor(bytes, &cursor, PadMode),
            .groups = try readSizeCursor(bytes, &cursor),
        } },
        .Conv2D => .{ .Conv2D = .{
            .stride_h = try readSizeCursor(bytes, &cursor),
            .stride_w = try readSizeCursor(bytes, &cursor),
            .dilation_h = try readSizeCursor(bytes, &cursor),
            .dilation_w = try readSizeCursor(bytes, &cursor),
            .pad_top = try readSizeCursor(bytes, &cursor),
            .pad_bottom = try readSizeCursor(bytes, &cursor),
            .pad_left = try readSizeCursor(bytes, &cursor),
            .pad_right = try readSizeCursor(bytes, &cursor),
            .pad_mode = try readEnumCursor(bytes, &cursor, PadMode),
            .groups = try readSizeCursor(bytes, &cursor),
        } },
        .LayerNorm => blk: {
            const eps = try readIntCursor(bytes, &cursor, f32);
            const attr = try readSymbolicAttr(allocator, bytes, &cursor);
            break :blk .{ .LayerNorm = .{ .eps = eps, .normalized_shape = attr.sizes, .free_dims = attr.free_dims } };
        },
        .RMSNorm => blk: {
            const eps = try readIntCursor(bytes, &cursor, f32);
            const attr = try readSymbolicAttr(allocator, bytes, &cursor);
            break :blk .{ .RMSNorm = .{ .eps = eps, .normalized_shape = attr.sizes, .free_dims = attr.free_dims } };
        },
        .Attention => blk: {
            const scale = try readIntCursor(bytes, &cursor, f32);
            const window = try readWindow(bytes, &cursor);
            const soft_cap = try readIntCursor(bytes, &cursor, f32);
            // The file packs the two optional control operands into one byte.
            const controls = try readIntCursor(bytes, &cursor, u8);
            if ((controls & ~@as(u8, 0x3)) != 0) return PackageError.InvalidFormat;
            break :blk .{ .Attention = .{
                .scale = scale,
                .window = window,
                .attn_logits_soft_cap = soft_cap,
                .has_query_positions = (controls & 1) != 0,
                .has_kv_lengths = (controls & 2) != 0,
            } };
        },
        .Reduce => .{ .Reduce = .{
            .op = try readEnumCursor(bytes, &cursor, ReduceOp),
            .axis = if ((try readIntCursor(bytes, &cursor, u8)) != 0) try readIntCursor(bytes, &cursor, i32) else null,
        } },
        .Concat => .{ .Concat = .{ .axis = try readIntCursor(bytes, &cursor, i32) } },
        .LSTMCell => .{ .LSTMCell = .{ .has_bias = (try readIntCursor(bytes, &cursor, u8)) != 0 } },
        .RFFT => .RFFT,
        .STFT => .{ .STFT = .{
            .n_fft = try readSizeCursor(bytes, &cursor),
            .hop_length = try readSizeCursor(bytes, &cursor),
            .center = (try readIntCursor(bytes, &cursor, u8)) != 0,
        } },
        .RelPosMHA => .{ .RelPosMHA = .{
            .scale = try readIntCursor(bytes, &cursor, f32),
            .has_mask = (try readIntCursor(bytes, &cursor, u8)) != 0,
            .window = try readWindow(bytes, &cursor),
            .relative_zero_index = try readSizeCursor(bytes, &cursor),
            .attn_logits_soft_cap = try readIntCursor(bytes, &cursor, f32),
        } },
        .ArgMax => .{ .ArgMax = .{ .axis = try readIntCursor(bytes, &cursor, i32) } },
        .ScatterRow => .ScatterRow,
        .Gather => .{ .Gather = .{
            .axis = try readIntCursor(bytes, &cursor, i32),
            .batch_dims = try readSizeCursor(bytes, &cursor),
        } },
        .Dim => .{ .Dim = .{ .axis = try readIntCursor(bytes, &cursor, i32) } },
        .Iota => .{ .Iota = .{ .axis = try readIntCursor(bytes, &cursor, i32) } },
        .Copy => .Copy,
        .RoPE1D => .{ .RoPE1D = .{
            .base_frequency = try readIntCursor(bytes, &cursor, f32),
            .scale_factor = try readIntCursor(bytes, &cursor, f32),
            .rope_proportion = try readIntCursor(bytes, &cursor, f32),
        } },
        .SequenceAppend => .SequenceAppend,
        .Cast => .{ .Cast = .{ .to_dtype = try readEnumCursor(bytes, &cursor, DType) } },
        .MatMulNT => .{ .MatMulNT = .{
            .alpha = try readIntCursor(bytes, &cursor, f32),
            .beta = try readIntCursor(bytes, &cursor, f32),
        } },
        .ViewReshape => blk: {
            const attr = try readSymbolicAttr(allocator, bytes, &cursor);
            break :blk .{ .ViewReshape = .{ .new_shape = attr.sizes, .free_dims = attr.free_dims } };
        },
        .ViewSqueeze => .{ .ViewSqueeze = .{ .axis = if ((try readIntCursor(bytes, &cursor, u8)) != 0) try readIntCursor(bytes, &cursor, i32) else null } },
        .ViewUnsqueeze => .{ .ViewUnsqueeze = .{ .axis = try readIntCursor(bytes, &cursor, i32) } },
        .ViewTranspose2D => .ViewTranspose2D,
        .ViewSliceND => blk: {
            const starts = try readSizeArray(allocator, bytes, &cursor);
            const attr = try readSymbolicAttr(allocator, bytes, &cursor);
            break :blk .{ .ViewSliceND = .{ .starts = starts, .lens = attr.sizes, .free_dims = attr.free_dims } };
        },
    };
    if (cursor != bytes.len) return PackageError.InvalidFormat;
    return .{ .op = op, .extra_outputs = extra_outputs };
}

fn parseSignaturesSection(allocator: std.mem.Allocator, strings: [][]u8, bytes: []const u8) PackageError!struct { inputs: []NamedValue, outputs: []NamedValue } {
    var cursor: usize = 0;
    const input_count = std.math.cast(usize, try readIntCursor(bytes, &cursor, u32)) orelse return PackageError.InvalidFormat;
    const output_count = std.math.cast(usize, try readIntCursor(bytes, &cursor, u32)) orelse return PackageError.InvalidFormat;
    const inputs = try parseNamedValues(allocator, strings, bytes, &cursor, input_count);
    errdefer freeNamedValues(allocator, inputs);
    const outputs = try parseNamedValues(allocator, strings, bytes, &cursor, output_count);
    errdefer freeNamedValues(allocator, outputs);
    if (cursor != bytes.len) return PackageError.InvalidFormat;
    return .{ .inputs = inputs, .outputs = outputs };
}

fn parseNamedValues(
    allocator: std.mem.Allocator,
    strings: [][]u8,
    bytes: []const u8,
    cursor: *usize,
    count: usize,
) PackageError![]NamedValue {
    const out = allocator.alloc(NamedValue, count) catch return PackageError.OutOfMemory;
    errdefer allocator.free(out);
    for (out) |*slot| {
        const name_idx = try readStringIndex(bytes, cursor, strings.len);
        slot.* = .{
            .name = allocator.dupe(u8, strings[name_idx]) catch return PackageError.OutOfMemory,
            .value = try readIntCursor(bytes, cursor, u32),
        };
    }
    return out;
}

fn parseMetadataSection(allocator: std.mem.Allocator, strings: [][]u8, bytes: []const u8) PackageError![]MetadataEntry {
    var cursor: usize = 0;
    const count = std.math.cast(usize, try readIntCursor(bytes, &cursor, u32)) orelse return PackageError.InvalidFormat;
    const out = allocator.alloc(MetadataEntry, count) catch return PackageError.OutOfMemory;
    errdefer allocator.free(out);
    for (out) |*slot| {
        const key_idx = try readStringIndex(bytes, &cursor, strings.len);
        const value_idx = try readStringIndex(bytes, &cursor, strings.len);
        slot.* = .{
            .key = allocator.dupe(u8, strings[key_idx]) catch return PackageError.OutOfMemory,
            .value = allocator.dupe(u8, strings[value_idx]) catch return PackageError.OutOfMemory,
        };
    }
    if (cursor != bytes.len) return PackageError.InvalidFormat;
    return out;
}

fn parseDebugNamesSection(allocator: std.mem.Allocator, strings: [][]u8, bytes: []const u8) PackageError![]DebugName {
    var cursor: usize = 0;
    const count = std.math.cast(usize, try readIntCursor(bytes, &cursor, u32)) orelse return PackageError.InvalidFormat;
    const out = allocator.alloc(DebugName, count) catch return PackageError.OutOfMemory;
    errdefer allocator.free(out);
    for (out) |*slot| {
        const value = try readIntCursor(bytes, &cursor, u32);
        const name_idx = try readStringIndex(bytes, &cursor, strings.len);
        slot.* = .{
            .value = value,
            .name = allocator.dupe(u8, strings[name_idx]) catch return PackageError.OutOfMemory,
        };
    }
    if (cursor != bytes.len) return PackageError.InvalidFormat;
    return out;
}

fn parseIoAliasesSection(allocator: std.mem.Allocator, bytes: []const u8) PackageError![]IoAlias {
    var cursor: usize = 0;
    const count = std.math.cast(usize, try readIntCursor(bytes, &cursor, u32)) orelse return PackageError.InvalidFormat;
    const out = allocator.alloc(IoAlias, count) catch return PackageError.OutOfMemory;
    errdefer allocator.free(out);
    for (out) |*slot| {
        slot.* = .{
            .input = try readIntCursor(bytes, &cursor, u32),
            .output = try readIntCursor(bytes, &cursor, u32),
        };
    }
    if (cursor != bytes.len) return PackageError.InvalidFormat;
    return out;
}

fn parseInputRolesSection(allocator: std.mem.Allocator, bytes: []const u8) PackageError![]types.InputRole {
    var cursor: usize = 0;
    const count = std.math.cast(usize, try readIntCursor(bytes, &cursor, u32)) orelse return PackageError.InvalidFormat;
    const out = allocator.alloc(types.InputRole, count) catch return PackageError.OutOfMemory;
    errdefer allocator.free(out);
    for (out) |*slot| {
        const input = try readIntCursor(bytes, &cursor, u32);
        const kind_raw = try readIntCursor(bytes, &cursor, u8);
        const axis = try readIntCursor(bytes, &cursor, u8);
        const flags = try readIntCursor(bytes, &cursor, u8);
        const reserved = try readIntCursor(bytes, &cursor, u8);
        const capacity_symbol = try readIntCursor(bytes, &cursor, u32);
        if (reserved != 0) return PackageError.InvalidFormat;
        const kind = std.enums.fromInt(types.InputRoleKind, kind_raw) orelse return PackageError.InvalidFormat;
        slot.* = .{
            .input = input,
            .kind = kind,
            .axis = axis,
            .flags = flags,
            .capacity_symbol = capacity_symbol,
        };
    }
    if (cursor != bytes.len) return PackageError.InvalidFormat;
    return out;
}

fn parseGraphMetaSection(bytes: []const u8) PackageError!GraphMeta {
    var cursor: usize = 0;
    const meta: GraphMeta = .{
        .value_count = try readIntCursor(bytes, &cursor, u32),
        .node_count = try readIntCursor(bytes, &cursor, u32),
        .region_count = try readIntCursor(bytes, &cursor, u32),
        .initializer_count = try readIntCursor(bytes, &cursor, u32),
        .input_count = try readIntCursor(bytes, &cursor, u32),
        .output_count = try readIntCursor(bytes, &cursor, u32),
        .dim_symbol_count = try readIntCursor(bytes, &cursor, u32),
        .dim_expr_count = try readIntCursor(bytes, &cursor, u32),
        .metadata_count = try readIntCursor(bytes, &cursor, u32),
        .debug_name_count = try readIntCursor(bytes, &cursor, u32),
        .io_alias_count = try readIntCursor(bytes, &cursor, u32),
    };
    if (cursor != bytes.len) return PackageError.InvalidFormat;
    return meta;
}

fn freeInitializers(allocator: std.mem.Allocator, initializers: []Initializer) void {
    for (initializers) |*init| {
        switch (init.encoding) {
            .plain => {},
            .quantized => |q| {
                allocator.free(q.scheme);
                allocator.free(q.params);
            },
        }
        allocator.free(init.data);
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
    allocator.free(regions);
}

fn freeNodes(allocator: std.mem.Allocator, nodes: []NodeRecord) void {
    for (nodes) |node| {
        allocator.free(node.inputs);
        types.deinitNodeOp(allocator, node.op);
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

/// An op's shape attribute, read as the (sizes, symbols) pair `graph.Op` carries.
///
/// The file stores each axis as a `ShapeTerm`: a constant, or an expression index for a
/// free axis. A free axis therefore has no recorded size, and its placeholder is 0 —
/// `Template.specialize` overwrites it before inference (see `graph.Op`'s `symbols`).
/// `free_dims` comes back empty when every axis is fixed, which is the common case.
fn readSymbolicAttr(allocator: std.mem.Allocator, bytes: []const u8, cursor: *usize) PackageError!struct {
    sizes: []const usize,
    free_dims: []const ?u32,
} {
    const count = std.math.cast(usize, try readIntCursor(bytes, cursor, u32)) orelse return PackageError.InvalidFormat;
    const sizes = allocator.alloc(usize, count) catch return PackageError.OutOfMemory;
    errdefer allocator.free(sizes);
    const free_dims = allocator.alloc(?u32, count) catch return PackageError.OutOfMemory;
    errdefer allocator.free(free_dims);

    var any_free = false;
    for (sizes, free_dims) |*size, *free_dim| {
        switch (try readShapeTerm(bytes, cursor)) {
            .constant => |c| {
                size.* = std.math.cast(usize, c) orelse return PackageError.InvalidFormat;
                free_dim.* = null;
            },
            .expr => |e| {
                size.* = 0;
                free_dim.* = e;
                any_free = true;
            },
        }
    }
    if (!any_free) {
        allocator.free(free_dims);
        return .{ .sizes = sizes, .free_dims = &.{} };
    }
    return .{ .sizes = sizes, .free_dims = free_dims };
}

/// A size stored as u64 and held as `usize`, which is what the graph's ops use.
fn readWindow(bytes: []const u8, cursor: *usize) PackageError!graph_mod.AttentionWindow {
    return .{
        .left = try readIntCursor(bytes, cursor, u32),
        .right = try readIntCursor(bytes, cursor, u32),
        .chunk = try readIntCursor(bytes, cursor, u32),
    };
}

fn readSizeCursor(bytes: []const u8, cursor: *usize) PackageError!usize {
    return std.math.cast(usize, try readIntCursor(bytes, cursor, u64)) orelse PackageError.InvalidFormat;
}

fn readSizeArray(allocator: std.mem.Allocator, bytes: []const u8, cursor: *usize) PackageError![]usize {
    const count = std.math.cast(usize, try readIntCursor(bytes, cursor, u32)) orelse return PackageError.InvalidFormat;
    const out = allocator.alloc(usize, count) catch return PackageError.OutOfMemory;
    errdefer allocator.free(out);
    for (out) |*value| value.* = try readSizeCursor(bytes, cursor);
    return out;
}

fn readShapeTerm(bytes: []const u8, cursor: *usize) PackageError!ShapeTerm {
    const kind = try readIntCursor(bytes, cursor, u8);
    _ = try readBytes(bytes, cursor, 7);
    const payload = try readIntCursor(bytes, cursor, u64);
    return switch (kind) {
        0 => .{ .constant = payload },
        1 => .{ .expr = std.math.cast(u32, payload) orelse return PackageError.InvalidFormat },
        else => return PackageError.InvalidFormat,
    };
}

fn readEnumCursor(bytes: []const u8, cursor: *usize, comptime E: type) PackageError!E {
    const raw = try readIntCursor(bytes, cursor, u8);
    return std.enums.fromInt(E, raw) orelse return PackageError.InvalidFormat;
}

fn readStringIndex(bytes: []const u8, cursor: *usize, string_count: usize) PackageError!usize {
    const idx = std.math.cast(usize, try readIntCursor(bytes, cursor, u32)) orelse return PackageError.InvalidFormat;
    if (idx >= string_count) return PackageError.InvalidFormat;
    return idx;
}

fn readBytes(bytes: []const u8, cursor: *usize, len: usize) PackageError![]const u8 {
    const out = try sliceAt(bytes, cursor.*, len);
    cursor.* += len;
    return out;
}

fn readIntCursor(bytes: []const u8, cursor: *usize, comptime T: type) PackageError!T {
    const slice = try readBytes(bytes, cursor, @sizeOf(T));
    return switch (@typeInfo(T)) {
        .int => std.mem.readInt(T, slice[0..@sizeOf(T)], .little),
        .float => |float_info| blk: {
            const IntT = @Int(.unsigned, float_info.bits);
            const bits = std.mem.readInt(IntT, @ptrCast(slice[0..@sizeOf(T)]), .little);
            break :blk @bitCast(bits);
        },
        else => @compileError("readIntCursor only supports integer and float types"),
    };
}

fn sliceAt(bytes: []const u8, start: usize, len: usize) PackageError![]const u8 {
    if (start > bytes.len or len > bytes.len - start) return PackageError.InvalidFormat;
    return bytes[start .. start + len];
}
