// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! A compiled program's claim on the tensors it names.
//!
//! Weight lifetime is a store-wide question — a `Context` shares one store between models
//! — so it cannot be decided by whichever model compiled last. What a model *can* say is
//! "this program exists" and "this program is gone", which is all of this file.
//!
//! It lives in the program layer because that is the layer that knows both sides: the
//! store speaks `TensorId` and nothing else, and `runtime/executable.zig` is deliberately
//! storage-agnostic.
//!
//! `tensor_placements` is the set of tensors a program names: collected from
//! `executable.tensorUses` over every step, pruned when a pass deletes steps, and patched
//! by `retargetProgramTensorIds` — so acquire and release always see the same ids.

const manager_mod = @import("../../storage/manager.zig");
const executable = @import("../../runtime/executable.zig");

const StorageManager = manager_mod.StorageManager;
const StorageError = manager_mod.StorageError;
const Program = executable.ExecutableProgram;

/// Claim every tensor this program names, and make each one readable.
///
/// A weight an optimization pass folded away keeps its metadata and loses its bytes,
/// which is sound only while every program reads the fused weight instead. A program that
/// reads the source directly gets it unfolded here, before placement — so a pass that
/// declines a group at one shape costs a copy rather than a program that cannot run.
pub fn acquire(mgr: *StorageManager, prog: *const Program) StorageError!void {
    for (prog.tensor_placements) |entry| try mgr.unfoldTensor(entry.id);
    for (prog.tensor_placements) |entry| mgr.retainTensor(entry.id);
    mgr.collectDerived();
}

/// Give up the claim. Anything that was only being kept alive for this program goes.
pub fn release(mgr: *StorageManager, prog: *const Program) void {
    for (prog.tensor_placements) |entry| mgr.releaseTensor(entry.id);
    mgr.collectDerived();
}
