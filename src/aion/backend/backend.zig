// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const types = @import("types.zig");
const executable = @import("../runtime/executable.zig");
const tensor_store = @import("../runtime/tensor_store.zig");

// Local aliases (NOT re-exports). Use `types.zig` / `utils.zig` directly from other files.
const BackendKind = types.BackendKind;
const BackendCaps = types.BackendCaps;
const BackendError = types.BackendError;
const Layout = types.Layout;
const BufferViewConst = types.BufferViewConst;
const BufferViewMut = types.BufferViewMut;
const ElemwiseBinaryOp = types.ElemwiseBinaryOp;
const ReduceOp = types.ReduceOp;
const MatMulParams = types.MatMulParams;

pub const ExecuteProgramError = BackendError || tensor_store.StoreError;

pub const Backend = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        kind: *const fn (ctx: *anyopaque) BackendKind,
        name: *const fn (ctx: *anyopaque) []const u8,
        caps: *const fn (ctx: *anyopaque) BackendCaps,
        deinit: *const fn (ctx: *anyopaque) void,

        /// Execute a validated tiled program.
        ///
        /// Contract:
        /// - `prog` has been validated end-to-end by the compiler (Graph -> ExecutableProgram).
        /// - Implementation must not perform per-op argument checking; only backend/storage errors can occur.
        executeProgram: *const fn (ctx: *anyopaque, prog: *const executable.ExecutableProgram, store: tensor_store.TensorStore) ExecuteProgramError!void,
    };

    pub fn kind(self: Backend) BackendKind {
        return self.vtable.kind(self.ctx);
    }

    pub fn name(self: Backend) []const u8 {
        return self.vtable.name(self.ctx);
    }

    pub fn caps(self: Backend) BackendCaps {
        return self.vtable.caps(self.ctx);
    }

    pub fn deinit(self: Backend) void {
        self.vtable.deinit(self.ctx);
    }

    pub fn executeProgram(
        self: Backend,
        prog: *const executable.ExecutableProgram,
        store: tensor_store.TensorStore,
    ) ExecuteProgramError!void {
        return self.vtable.executeProgram(self.ctx, prog, store);
    }
};
