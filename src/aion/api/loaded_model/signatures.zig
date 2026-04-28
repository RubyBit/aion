// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const package_file = @import("../../storage/aion_file.zig");
const manager_mod = @import("../../storage/manager.zig");
const types_mod = @import("types.zig");
const api_errors = @import("../errors.zig");

pub fn buildSignatures(
    allocator: std.mem.Allocator,
    package: *const types_mod.Package,
    named_values: []const package_file.NamedValue,
) api_errors.LoadError![]types_mod.SignatureInfo {
    const out = try allocator.alloc(types_mod.SignatureInfo, named_values.len);
    errdefer allocator.free(out);
    for (named_values, 0..) |sig, i| {
        if (sig.value >= package.values.len) return error.InvalidArgument;
        const value = package.values[sig.value];
        out[i] = .{
            .name = sig.name,
            .value = sig.value,
            .dtype = value.dtype,
            .rank = value.rank,
        };
    }
    return out;
}

pub fn buildIoAliasInfo(
    allocator: std.mem.Allocator,
    package: *const types_mod.Package,
    input_signatures: []const types_mod.SignatureInfo,
    output_signatures: []const types_mod.SignatureInfo,
) api_errors.LoadError![]types_mod.IoAliasInfo {
    const out = try allocator.alloc(types_mod.IoAliasInfo, package.io_aliases.len);
    errdefer allocator.free(out);
    for (package.io_aliases, 0..) |alias, idx| {
        const input_index: usize = alias.input;
        const output_index: usize = alias.output;
        if (input_index >= input_signatures.len or output_index >= output_signatures.len) return error.InvalidArgument;
        out[idx] = .{
            .input_name = input_signatures[input_index].name,
            .output_name = output_signatures[output_index].name,
            .input_index = input_index,
            .output_index = output_index,
        };
    }
    return out;
}

pub fn bindInputDim(
    pkg: *const types_mod.Package,
    term: package_file.ShapeTerm,
    actual: u64,
    symbol_values: []?u64,
) error{InvalidArgument}!void {
    switch (term) {
        .constant => |want| if (want != actual) return error.InvalidArgument,
        .expr => |expr_idx| {
            if (expr_idx >= pkg.dim_exprs.len) return error.InvalidArgument;
            switch (pkg.dim_exprs[expr_idx]) {
                .symbol => |sym_idx| {
                    if (sym_idx >= symbol_values.len) return error.InvalidArgument;
                    if (symbol_values[sym_idx]) |current| {
                        if (current != actual) return error.InvalidArgument;
                    } else {
                        symbol_values[sym_idx] = actual;
                    }
                },
                else => return error.InvalidArgument,
            }
        },
    }
}

pub fn sameU64(a: []const u64, b: []const u64) bool {
    return std.mem.eql(u64, a, b);
}

pub fn sameUsize(a: []const usize, b: []const usize) bool {
    return std.mem.eql(usize, a, b);
}

pub fn tensorsHaveCompatibleLayout(
    store: *types_mod.StorageManager,
    old_tid: types_mod.TensorId,
    new_tid: types_mod.TensorId,
) manager_mod.StorageError!bool {
    const old_meta = try store.getConst(old_tid);
    const new_meta = try store.getConst(new_tid);
    return old_meta.dtype == new_meta.dtype and
        old_meta.rank == new_meta.rank and
        sameUsize(old_meta.shape, new_meta.shape) and
        sameUsize(old_meta.tile_shape, new_meta.tile_shape) and
        sameUsize(old_meta.tile_counts, new_meta.tile_counts);
}
