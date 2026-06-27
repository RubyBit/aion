// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Root source file for the per-ISA kernel tier objects (see
// `aion/backend/cpu/multiversion/tier_export.zig` and `build.zig`).
//
// Its only job is to anchor the tier object's module root at `src/` — the same
// root directory as `src/root.zig` — so the backend's relative imports (e.g.
// `../../types.zig`) resolve. Referencing `tier_export` forces analysis of its
// `comptime` block, which performs the `@export` of the tier accessor symbol.

comptime {
    _ = @import("aion/backend/cpu/multiversion/tier_export.zig");
}
