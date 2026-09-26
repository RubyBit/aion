// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Where a compile is going.
//!
//! Which device, how its kernels want weights laid out, and which optional rewrites
//! it profits from are one decision, not three: the pass set is derived for a device
//! (see `opt.defaults`). Threading them separately is how a call site ends up passing
//! a layout for one target and a device index for another.
//!
//! Non-default passes exist to bisect a pass on a real model or ablate one in a bench.
//! That is a property of a compile, never of a `.aion` — one file has to be able to
//! compile to a different schedule per backend.

const opt = @import("opt.zig");
const types = @import("../backend/types.zig");
const manager_mod = @import("../storage/manager.zig");

const DeviceRef = manager_mod.DeviceRef;

/// Deliberately without field defaults: `passes` is *derived* from the device, so a
/// `Target{}` literal would read as "the obvious target" while meaning "no optimizations
/// at all". Every target goes through a constructor.
pub const Target = struct {
    /// The device this program will execute on, and whose memory its tensors live in.
    device: DeviceRef,
    /// How an `[n, k]` q8 weight groups its rows for this device's NT kernel (see
    /// `types.QuantBlockOrder`); `row_major` groups none.
    quant_block_order: types.QuantBlockOrder,
    /// Which optional rewrites to run.
    passes: opt.Policy,

    /// The target a device profits from: its weight layout plus its default pass set.
    pub fn init(device: DeviceRef, quant_block_order: types.QuantBlockOrder) Target {
        return .{ .device = device, .quant_block_order = quant_block_order, .passes = opt.defaults() };
    }

    /// A host target whose kernels read ungrouped weights: what tests and tools that
    /// do not ask a CPU backend for its order want.
    pub fn cpu() Target {
        return init(.{}, .row_major);
    }

    /// Override the pass set — bisecting a pass, or a bench ablation.
    pub fn withPasses(self: Target, passes: opt.Policy) Target {
        return .{ .device = self.device, .quant_block_order = self.quant_block_order, .passes = passes };
    }

    /// The backend kind that executes this target's steps.
    pub fn backendKind(self: Target) types.BackendKind {
        return if (self.device.kind == .cpu) .cpu else .webgpu;
    }
};
