// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const types = @import("types.zig");

const Package = types.Package;
const ShapeTerm = types.ShapeTerm;
const DimExpr = types.DimExpr;
const PackageError = types.PackageError;

pub fn makeConstantShapeTerms(allocator: std.mem.Allocator, shape: []const usize) PackageError![]ShapeTerm {
    const out = allocator.alloc(ShapeTerm, shape.len) catch return PackageError.OutOfMemory;
    for (shape, 0..) |dim, i| out[i] = .{ .constant = @intCast(dim) };
    return out;
}

pub fn evaluateShapeTerm(pkg: *const Package, term: ShapeTerm, symbol_values: []const ?u64) PackageError!u64 {
    const visiting = pkg.allocator.alloc(bool, pkg.dim_exprs.len) catch return PackageError.OutOfMemory;
    defer pkg.allocator.free(visiting);
    @memset(visiting, false);
    return evaluateShapeTermWithVisiting(pkg.dim_exprs, term, symbol_values, visiting);
}

pub fn resolveShapeTerms(
    allocator: std.mem.Allocator,
    pkg: *const Package,
    terms: []const ShapeTerm,
    symbol_values: []const ?u64,
) PackageError![]usize {
    return resolveShapeTermsExprs(allocator, pkg.dim_exprs, terms, symbol_values);
}

/// Like `resolveShapeTerms`, but takes the `dim_exprs` table directly instead of a
/// `Package`. Lets the execution runtime resolve shapes without depending on the
/// on-disk package representation (compiled models have no package).
pub fn resolveShapeTermsExprs(
    allocator: std.mem.Allocator,
    dim_exprs: []const DimExpr,
    terms: []const ShapeTerm,
    symbol_values: []const ?u64,
) PackageError![]usize {
    const visiting = allocator.alloc(bool, dim_exprs.len) catch return PackageError.OutOfMemory;
    defer allocator.free(visiting);
    @memset(visiting, false);
    const out = allocator.alloc(usize, terms.len) catch return PackageError.OutOfMemory;
    errdefer allocator.free(out);
    for (terms, 0..) |term, i| {
        const value = try evaluateShapeTermWithVisiting(dim_exprs, term, symbol_values, visiting);
        out[i] = std.math.cast(usize, value) orelse return PackageError.InvalidArgument;
    }
    return out;
}

/// One dim expression's value. `resolveShapeTermsExprs` resolves a whole shape; an op
/// attribute resolves axis by axis, because only some of its axes are free.
pub fn evaluateDimExpr(
    allocator: std.mem.Allocator,
    dim_exprs: []const DimExpr,
    expr_idx: u32,
    symbol_values: []const ?u64,
) PackageError!usize {
    const visiting = allocator.alloc(bool, dim_exprs.len) catch return PackageError.OutOfMemory;
    defer allocator.free(visiting);
    @memset(visiting, false);
    const value = try evaluateExpr(dim_exprs, expr_idx, symbol_values, visiting);
    return std.math.cast(usize, value) orelse PackageError.InvalidArgument;
}

fn evaluateShapeTermWithVisiting(
    dim_exprs: []const DimExpr,
    term: ShapeTerm,
    symbol_values: []const ?u64,
    visiting: []bool,
) PackageError!u64 {
    return switch (term) {
        .constant => |value| value,
        .expr => |expr_idx| try evaluateExpr(dim_exprs, expr_idx, symbol_values, visiting),
    };
}

fn evaluateExpr(
    dim_exprs: []const DimExpr,
    expr_idx: u32,
    symbol_values: []const ?u64,
    visiting: []bool,
) PackageError!u64 {
    if (expr_idx >= dim_exprs.len) return PackageError.InvalidArgument;
    const idx: usize = @intCast(expr_idx);
    if (visiting[idx]) return PackageError.InvalidFormat;
    visiting[idx] = true;
    defer visiting[idx] = false;

    return switch (dim_exprs[idx]) {
        .symbol => |sym_idx| blk: {
            if (sym_idx >= symbol_values.len) return PackageError.InvalidArgument;
            const value = symbol_values[sym_idx] orelse return PackageError.InvalidArgument;
            if (value == 0) return PackageError.InvalidArgument;
            break :blk value;
        },
        .add => |bin| blk: {
            const lhs = try evaluateShapeTermWithVisiting(dim_exprs, bin.lhs, symbol_values, visiting);
            const rhs = try evaluateShapeTermWithVisiting(dim_exprs, bin.rhs, symbol_values, visiting);
            break :blk std.math.add(u64, lhs, rhs) catch return PackageError.InvalidArgument;
        },
        .sub => |bin| blk: {
            const lhs = try evaluateShapeTermWithVisiting(dim_exprs, bin.lhs, symbol_values, visiting);
            const rhs = try evaluateShapeTermWithVisiting(dim_exprs, bin.rhs, symbol_values, visiting);
            if (rhs >= lhs) return PackageError.InvalidArgument;
            break :blk lhs - rhs;
        },
        .mul => |bin| blk: {
            const lhs = try evaluateShapeTermWithVisiting(dim_exprs, bin.lhs, symbol_values, visiting);
            const rhs = try evaluateShapeTermWithVisiting(dim_exprs, bin.rhs, symbol_values, visiting);
            break :blk std.math.mul(u64, lhs, rhs) catch return PackageError.InvalidArgument;
        },
        .floor_div => |bin| blk: {
            const lhs = try evaluateShapeTermWithVisiting(dim_exprs, bin.lhs, symbol_values, visiting);
            const rhs = try evaluateShapeTermWithVisiting(dim_exprs, bin.rhs, symbol_values, visiting);
            if (rhs == 0) return PackageError.InvalidArgument;
            break :blk lhs / rhs;
        },
        .max => |bin| blk: {
            const lhs = try evaluateShapeTermWithVisiting(dim_exprs, bin.lhs, symbol_values, visiting);
            const rhs = try evaluateShapeTermWithVisiting(dim_exprs, bin.rhs, symbol_values, visiting);
            break :blk @max(lhs, rhs);
        },
    };
}
