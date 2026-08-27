// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//! Where a compile is going.
//!
//! Which device, how to tile for it, and which optional rewrites it profits from are one
//! decision, not three: the tiling is derived for a device, and so is the pass set (see
//! `opt.defaults`). Threading them separately is how a call site ends up passing a policy
//! for one target and a device index for another, and it is why the pass set had nowhere
//! to live but an options struct belonging to some caller.
//!
//! Non-default passes exist to bisect a pass on a real model or ablate one in a bench.
//! That is a property of a compile, never of a `.aion` — one file has to be able to
//! compile to a different schedule per backend.

const plan = @import("plan.zig");
const opt = @import("opt.zig");
const manager_mod = @import("../storage/manager.zig");

const DeviceRef = manager_mod.DeviceRef;
const TilePolicy = plan.TilePolicy;

/// Deliberately without field defaults: `passes` is *derived* from the device, so a
/// `Target{}` literal would read as "the obvious target" while meaning "no optimizations
/// at all". Every target goes through a constructor.
pub const Target = struct {
    /// The device this program will execute on, and whose memory its tensors live in.
    device: DeviceRef,
    /// How to tile for that device.
    tiles: TilePolicy,
    /// Which optional rewrites to run.
    passes: opt.Policy,

    /// The target a device profits from: its tiling plus its default pass set.
    pub fn init(device: DeviceRef, tiles: TilePolicy) Target {
        return .{ .device = device, .tiles = tiles, .passes = opt.defaults(tiles) };
    }

    /// A host target with `tiles`. The same as `init(.{}, tiles)`, named because most
    /// tests and every CPU model want exactly this.
    pub fn cpu(tiles: TilePolicy) Target {
        return init(.{}, tiles);
    }

    /// Override the pass set — bisecting a pass, or a bench ablation.
    pub fn withPasses(self: Target, passes: opt.Policy) Target {
        return .{ .device = self.device, .tiles = self.tiles, .passes = passes };
    }
};
