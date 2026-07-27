// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const api_builder = @import("builder.zig");

pub const Builder = api_builder.Builder;
pub const TensorRef = api_builder.TensorRef;

fn baseModuleType(comptime T: type) type {
    return switch (@typeInfo(T)) {
        .pointer => |p| p.child,
        else => T,
    };
}

/// Best-effort user-facing module type name inferred from `T`.
///
/// This strips namespace/path qualifiers from `@typeName(T)` so users get
/// concise names like `Linear`, `Conv2D`, or `MyCustomBlock`.
pub fn moduleTypeName(comptime T: type) []const u8 {
    const U: type = baseModuleType(T);
    const full: []const u8 = @typeName(U);
    const dot = std.mem.lastIndexOfScalar(u8, full, '.') orelse return full;
    if (dot + 1 >= full.len) return full;
    return full[dot + 1 ..];
}

/// Begin a module scope for type `T`, nested inside any enclosing scope.
///
/// `explicit_name` overrides the *segment* name (use it to get `layers.3` instead
/// of `Linear#7`); it does not suppress the scope. Scopes nest, so composing
/// modules yields `state_dict`-style paths.
///
/// Still returns an optional so existing
/// `const scope = try beginModuleScope(...); defer endModuleScope(bld, scope);`
/// call sites compile unchanged — it simply never returns `null` now.
pub fn beginModuleScope(comptime T: type, bld: *Builder, explicit_name: ?[]const u8) Builder.Error!?Builder.Scope {
    if (explicit_name) |n| return try bld.beginScope(n);
    return try bld.beginAutoScope(moduleTypeName(T));
}

/// Close a scope returned by `beginModuleScope`.
pub fn endModuleScope(bld: *Builder, scope: ?Builder.Scope) void {
    if (scope) |s| bld.endScope(s);
}

fn WithModuleScopeReturnType(comptime Ret: type) type {
    return switch (@typeInfo(Ret)) {
        .error_union => Ret,
        else => Builder.Error!Ret,
    };
}

/// Convenience wrapper that begins an inferred module scope for `T`, executes
/// `func(args)`, and then closes the scope.
///
/// This is primarily meant to remove the common:
///
/// - `const scope = try beginModuleScope(...);`
/// - `defer endModuleScope(...);`
///
/// boilerplate from `forward()` implementations.
///
/// Notes:
/// - If a user-defined module scope is already active, this becomes a no-op
///   wrapper (it will not start a nested inferred scope).
/// - `func` should return either a non-error type `R` (then this returns
///   `Builder.Error!R`) or an error union whose error set can include
///   `Builder.Error` (commonly `Builder.Error!R`).
pub fn withModuleScope(
    comptime T: type,
    bld: *Builder,
    explicit_name: ?[]const u8,
    comptime func: anytype,
    args: anytype,
) WithModuleScopeReturnType(@TypeOf(@call(.auto, func, args))) {
    const scope: ?Builder.Scope = try beginModuleScope(T, bld, explicit_name);
    defer endModuleScope(bld, scope);
    return @call(.auto, func, args);
}

/// Runtime-polymorphic module interface (vtable style).
///
/// Intended for dynamic composition and replacement paths where the module
/// implementation is not known at comptime.
pub const ModuleDyn = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const ForwardError = anyerror;

    pub const VTable = struct {
        name: *const fn (ctx: *const anyopaque) []const u8,
        forward: *const fn (ctx: *anyopaque, bld: *Builder, input: TensorRef) ForwardError!TensorRef,
        deinit: ?*const fn (ctx: *anyopaque, allocator: std.mem.Allocator) void = null,

        /// Set when `forward` already opens its own module scope, so `ModuleDyn`
        /// must not add a second one. `moduleDynFrom` sets it (the module it wraps
        /// scopes itself via `beginModuleScope`); hand-written vtables leave it
        /// false and get an auto-scope named after `name()`.
        self_scoping: bool = false,
    };

    const Self = @This();

    pub fn init(ctx: *anyopaque, vtable: *const VTable) Self {
        return .{ .ctx = ctx, .vtable = vtable };
    }

    pub fn name(self: *const Self) []const u8 {
        return self.vtable.name(self.ctx);
    }

    pub fn forward(self: *Self, bld: *Builder, input: TensorRef) ForwardError!TensorRef {
        if (self.vtable.self_scoping) return self.vtable.forward(self.ctx, bld, input);

        const scope: Builder.Scope = try bld.beginAutoScope(self.name());
        defer bld.endScope(scope);
        return self.vtable.forward(self.ctx, bld, input);
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        if (self.vtable.deinit) |deinit_fn| {
            deinit_fn(self.ctx, allocator);
        }
        self.* = undefined;
    }
};

/// Returns whether a type provides a `forward` declaration.
pub fn isForwardModuleType(comptime T: type) bool {
    const U: type = baseModuleType(T);
    return @hasDecl(U, "forward");
}

/// Compile-time validator for forward-compatible module-like types.
pub fn assertForwardModuleType(comptime T: type) void {
    if (!isForwardModuleType(T)) {
        @compileError("Expected a module type exposing `forward`");
    }
}

/// Convert a regular bound module/layer (e.g. `nn.Linear`) into `ModuleDyn`.
///
/// The dynamic module borrows the provided `module` pointer; caller owns the
/// pointed value and must keep it alive while the dynamic wrapper is used.
///
/// Module name defaults to `@typeName(T)`.
pub fn moduleDynFrom(comptime T: type, module: *T) ModuleDyn {
    const Adapter = struct {
        fn name(ctx: *const anyopaque) []const u8 {
            _ = ctx;
            return moduleTypeName(T);
        }

        fn forward(ctx: *anyopaque, bld: *Builder, input: TensorRef) ModuleDyn.ForwardError!TensorRef {
            const typed: *T = @ptrCast(@alignCast(ctx));
            return typed.forward(bld, input);
        }

        const vtable: ModuleDyn.VTable = .{
            .name = name,
            .forward = forward,
            .deinit = null,
            // `T.forward` opens its own scope; don't wrap it in a second one.
            .self_scoping = true,
        };
    };

    return ModuleDyn.init(module, &Adapter.vtable);
}

/// Returns whether a type follows the current static module protocol:
/// it exposes `bind` and `forward` declarations.
pub fn isModuleType(comptime T: type) bool {
    return @hasDecl(T, "bind") and @hasDecl(T, "forward");
}

/// Compile-time validator for module-like types.
pub fn assertModuleType(comptime T: type) void {
    if (!isModuleType(T)) {
        @compileError("Expected a module type exposing both `bind` and `forward`");
    }
}
