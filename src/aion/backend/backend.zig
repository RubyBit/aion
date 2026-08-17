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

/// An execution session bound to a single `TensorStore`.
///
/// A session binds a backend to one tensor store for repeated execution. Tensor
/// backing ownership remains in the store; sessions own only backend bookkeeping.
pub const Session = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// Execute a validated tiled program against this session's store.
        ///
        /// Contract:
        /// - `prog` has been validated end-to-end by the compiler (Graph -> ExecutableProgram).
        /// - Implementation must not perform per-op argument checking; only backend/storage errors can occur.
        execute: *const fn (ctx: *anyopaque, prog: *const executable.ExecutableProgram) ExecuteProgramError!void,
        /// Wait for submitted work and drop backend resources that may retain
        /// tensor backings. Called only before storage eviction.
        retireResources: *const fn (ctx: *anyopaque) void,
        /// Release backend-local session bookkeeping.
        deinit: *const fn (ctx: *anyopaque) void,
    };

    pub fn execute(self: Session, prog: *const executable.ExecutableProgram) ExecuteProgramError!void {
        return self.vtable.execute(self.ctx, prog);
    }

    pub fn deinit(self: Session) void {
        self.vtable.deinit(self.ctx);
    }

    pub fn retireResources(self: Session) void {
        self.vtable.retireResources(self.ctx);
    }
};

pub const Backend = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        kind: *const fn (ctx: *anyopaque) BackendKind,
        name: *const fn (ctx: *anyopaque) []const u8,
        caps: *const fn (ctx: *anyopaque) BackendCaps,
        deinit: *const fn (ctx: *anyopaque) void,

        /// Build an execution `Session` bound to `store`. The session owns any
        /// per-store state (device residency) and must be released via
        /// `Session.deinit`. The backend's device-global caches are shared, so
        /// multiple live sessions over one backend are fine. Session creation only
        /// allocates host bookkeeping (device buffers are lazy), so it faults with
        /// `StoreError` (OOM), not the full execution error set.
        createSession: *const fn (ctx: *anyopaque, store: tensor_store.TensorStore) tensor_store.StoreError!Session,
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

    pub fn createSession(self: Backend, store: tensor_store.TensorStore) tensor_store.StoreError!Session {
        return self.vtable.createSession(self.ctx, store);
    }

    /// Convenience one-shot execution: build an ephemeral session, run `prog`,
    /// tear it down. Correct but not residency-persistent — callers that execute
    /// repeatedly against the same store (production, benches) should hold a
    /// `Session` explicitly so device residency is reused across calls.
    pub fn executeProgram(
        self: Backend,
        prog: *const executable.ExecutableProgram,
        store: tensor_store.TensorStore,
    ) ExecuteProgramError!void {
        var session = try self.createSession(store);
        defer session.deinit();
        return session.execute(prog);
    }
};
