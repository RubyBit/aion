const std = @import("std");

const backend_mod = @import("../backend/backend.zig");
const types = @import("../backend/types.zig");
const storage = @import("../storage/storage.zig");

pub const Backend = backend_mod.Backend;
pub const BackendError = types.BackendError;
pub const MatMulParams = types.MatMulParams;

pub const StorageError = storage.StorageError;
pub const TiledTensor = storage.TiledTensor;

/// Errors surfaced by tiled execution.
///
/// Notes:
/// - Backend failures are propagated.
/// - Storage errors are propagated.
/// - `InvalidArgument` is used for cross-tensor tiling incompatibilities.
pub const ProgramError = error{InvalidArgument} || BackendError || StorageError;

fn require(cond: bool) ProgramError!void {
    if (!cond) return ProgramError.InvalidArgument;
}

/// Executes a tiled GEMM:
///   C[M,N] = alpha * (A[M,K] @ B[K,N]) + beta * C
///
/// Contract (v0):
/// - A, B, C are rank-2 `TiledTensor`.
/// - A and C dtypes are scalar (f32/f16). B can be scalar or quant.
/// - Tiles are physically packed (guaranteed by `TiledTensor`).
/// - Tile geometry must be compatible:
///   - A.tile_shape = [tm, tk]
///   - B.tile_shape = [tk, tn]
///   - C.tile_shape = [tm, tn]
///   and the corresponding tile counts must match.
///
/// Performance notes:
/// - We intentionally keep the executor single-threaded.
///   CPU backend already parallelizes matmul for sufficiently large tiles.
///   A planner will later pick tile sizes that trigger backend parallelism.
pub fn matmulTiled(
    backend: Backend,
    c: *TiledTensor,
    a: *const TiledTensor,
    b: *const TiledTensor,
    alpha: f32,
    beta: f32,
) ProgramError!void {
    // Shape/dtype validation.
    try require(a.rank == 2 and b.rank == 2 and c.rank == 2);

    const m: usize = a.shape[0];
    const k: usize = a.shape[1];
    const k_b: usize = b.shape[0];
    const n: usize = b.shape[1];

    try require(k == k_b);
    try require(c.shape[0] == m and c.shape[1] == n);

    // Backend contracts: C must be scalar, A must be scalar, B can be quant.
    try require(!a.dtype.info().is_quantized);
    try require(!c.dtype.info().is_quantized);

    // Enforce the canonical tiling geometry:
    // A: [tm, tk], B: [tk, tn], C: [tm, tn].
    try require(a.tile_shape[0] == c.tile_shape[0]);
    try require(b.tile_shape[1] == c.tile_shape[1]);
    try require(a.tile_shape[1] == b.tile_shape[0]);

    // Tile grid compatibility.
    try require(a.tile_counts[0] == c.tile_counts[0]);
    try require(b.tile_counts[1] == c.tile_counts[1]);
    try require(a.tile_counts[1] == b.tile_counts[0]);

    // If B is quantized, K tiles must stay aligned to its block size.
    // This should already be guaranteed by `TiledTensor.init()` for quant tensors,
    // but we double-check cross-tensor compatibility here.
    if (b.dtype.info().is_quantized) {
        const be: usize = b.dtype.info().block_elems;
        try require(k % be == 0);
        try require(a.tile_shape[1] % be == 0);
        // Allow last k-tile smaller, but it must be block-aligned too.
        const rem: usize = k % a.tile_shape[1];
        if (rem != 0) try require(rem % be == 0);
    }

    var ti_m: usize = 0;
    while (ti_m < c.tile_counts[0]) : (ti_m += 1) {
        var ti_n: usize = 0;
        while (ti_n < c.tile_counts[1]) : (ti_n += 1) {
            // Keep the C tile live across all K tiles to avoid repeated struct init.
            var c_tile = try c.acquireTileMut(ti_m, ti_n);

            const c_tile_view0 = c_tile.bufferView();

            const m_tile: usize = c_tile_view0.layout.shape[0];
            const n_tile: usize = c_tile_view0.layout.shape[1];

            // Apply beta exactly once (on the first K slice). Subsequent slices accumulate.
            var ti_k: usize = 0;
            while (ti_k < a.tile_counts[1]) : (ti_k += 1) {
                const a_tile = try a.acquireTileConst(ti_m, ti_k);
                const b_tile = try b.acquireTileConst(ti_k, ti_n);
                const a_view = a_tile.bufferView();
                const b_view = b_tile.bufferView();
                const c_view = c_tile.bufferView();

                // Sanity shape checks for boundary tiles.
                const k_tile: usize = a_view.layout.shape[1];
                try require(a_view.layout.shape[0] == m_tile);
                try require(b_view.layout.shape[0] == k_tile);
                try require(b_view.layout.shape[1] == n_tile);

                const beta_tile: f32 = if (ti_k == 0) beta else 1.0;
                const params: MatMulParams = .{
                    .m = m_tile,
                    .n = n_tile,
                    .k = k_tile,
                    .alpha = alpha,
                    .beta = beta_tile,
                };

                // Delegate computation (and any internal threading) to the backend.
                try backend.matmul(params, c_view, a_view, b_view);
            }
        }
    }
}
