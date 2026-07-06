// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const types = @import("../backend/types.zig");

pub const DType = types.DType;
pub const BackendKind = types.BackendKind;
pub const BackendCaps = types.BackendCaps;

/// Describes the backend a graph is being compiled for. Today only `.cpu` is
/// realized; this is the seam through which a GPU backend selects a different
/// `TilePolicy` (see `TilePolicy.forTarget`) without the lowering pipeline
/// changing. The compiled `TilePolicy` carries `target_kind`, so `lowerNode`
/// can branch per-target later via the policy it already receives — no extra
/// parameter threading required.
pub const CompileTarget = struct {
    kind: BackendKind = .cpu,
    caps: BackendCaps = .{},
};

pub const TilePolicy = struct {
    /// Backend this policy was derived for. Defaults to `.cpu` so a bare
    /// `TilePolicy{}` is unchanged from before this field existed.
    target_kind: BackendKind = .cpu,

    /// Default square tile side for rank-2 tensors.
    ///
    /// Note: transpose materialization is simplest when tiles are square.
    base_square_2d: usize = 64,

    /// Default tile length for rank-1 tensors.
    base_1d: usize = 256,

    /// Required alignment for quantized K blocks (ggml q4/q8 use 32).
    quant_k_block: usize = 32,

    /// Tile alignment in bytes for `TiledTensor` backing.
    tile_alignment: usize = 64,

    /// Max number of batch tiles to allow when retiling rank>2 scalar tensors.
    /// Bounds the compile-time `ReTileCopyScalar` copy-loop length; it does not
    /// affect the post-retile tensor's tiling (that is set by `want_tile`). Sized
    /// generously so ops that bridge a transiently fine-tiled producer (e.g. a
    /// `reshape` that one-row-tiles a long sequence dim before `RelPosMHA` re-tiles
    /// it to a single [T, D] panel) are not falsely rejected.
    batch_retile_max_tiles: usize = 8192,

    /// Tensors with total element count at or below this threshold are stored
    /// as a single tile, eliminating per-tile acquire/release overhead and
    /// enabling the memcpy fast-path in readTensorPackedF32/writeTensorPackedF32.
    small_tensor_threshold: usize = 128 * 1024,

    /// For rank>2 tensors, scalar retiling that changes the last two dims can
    /// be expensive and may increase peak memory use (a new tiled backing buffer
    /// is allocated). We still need to allow it for common layout-bridging cases
    /// (e.g. matmul outputs feeding into another matmul with a different K-tile),
    /// so this separate threshold bounds when such retiles are permitted.
    ///
    /// This is deliberately decoupled from `small_tensor_threshold`: raising the
    /// latter would also force many tensors into single-tile storage, which can
    /// harm parallelism. Here we only control whether the compiler may insert a
    /// `ReTileCopyScalar` that changes the last/2nd-last tile sizes for rank>2.
    retile_last2d_change_max_elems: usize = 32 * 1024 * 1024,

    // ---- Per-row / attention scratch caps ----
    //
    // These bound tile sizes along axes that the CPU kernels back with bounded
    // (often stack) scratch. They are lifted out of magic constants so a GPU
    // target (via `forTarget`) can relax them — GPU kernels stage scratch in
    // shared/global memory and tolerate much larger tiles. The CPU defaults here
    // reproduce the previous hard-coded values exactly (byte-identical programs).

    /// Max rows per tile for row-wise softmax/layernorm/rmsnorm (per-row scratch).
    /// Used by both the tiling chooser and the compile-time validation guard.
    softmax_row_cap: usize = 256,

    /// Attention query-row tile target (heuristic; favors tile-level parallelism).
    attn_q_tile: usize = 32,
    /// Attention key-block tile cap (chooser cap AND validation key-rows limit).
    attn_k_tile_cap: usize = 128,
    /// Attention value/output-column tile cap (chooser cap AND validation limit).
    attn_v_tile_cap: usize = 64,
    /// Attention head-dim tile cap (chooser).
    attn_head_tile_cap: usize = 128,
    /// Max attention query rows per tile accepted by the CPU kernel (validation).
    attn_max_q_rows: usize = 256,

    // /// Target number of output tiles for matvec-shaped matmuls (M <= 4).
    // ///
    // /// Matvec matmuls parallelize over N tiles; with too few tiles, only a fraction
    // /// of the thread pool does real work. We size `tn` so that we get at least this
    // /// many tiles along the N axis, rounded to SIMD-friendly multiples of 16. Tuned
    // /// for 32-core CPUs; leaves a small margin so every worker usually gets one tile.
    // matvec_min_n_tiles: usize = 32,

    // /// Minimum `tn` for matvec matmuls. Even when the shape would prefer smaller
    // /// tiles for more parallelism, we keep at least this many columns per tile so
    // /// per-call packing/dispatch overhead doesn't dominate the inner kernel.
    // matvec_min_tn: usize = 32,
};

/// Macro tile-side cap for GPU targets. GPU wants FEW, LARGE tiles (the opposite
/// of the CPU's many-small-tiles-for-thread-parallelism heuristic): one large tile
/// per output collapses a matmul to a single dispatch with K accumulated in-kernel.
const GPU_MACRO_TILE_CAP: usize = 4096;
/// K-tile cap for GPU targets — full-K in one tile when K is moderate, so there is
/// one dispatch per output tile (no per-k-tile global-memory C round-trip).
///
/// `GPU_MACRO_TILE_CAP * GPU_K_TILE_CAP * 4 bytes` is the largest f32 tile we
/// emit (2048*2048*4 = 16 MiB). Kept conservative: WebGPU's default
/// `maxStorageBufferBindingSize` is 128 MiB, but a discrete NVIDIA adapter under
/// wgpu-native rejected a 32 MiB binding here, so we stay well below until the
/// backend queries the real device limit (a planned follow-up) and can size tiles
/// to it. Until then, square shapes ≤2048 stay a single dispatch; larger ones tile
/// into ≤16 MiB pieces (still far fewer/larger than the CPU policy).
const GPU_K_TILE_CAP: usize = 4096;

/// Default tiling policy for a compile target. CPU returns the historical
/// defaults; GPU targets get large base tile sizes so typical tensors land in a
/// single tile (few, large dispatches). The matmul choosers additionally branch
/// on `target_kind` (see `chooseMatMulTiles`/`chooseMatMulTk`).
pub fn tilePolicyForTarget(target: CompileTarget) TilePolicy {
    return switch (target.kind) {
        .cpu => .{ .target_kind = target.kind },
        // GPU kernels stage scratch in shared/global memory and tolerate much
        // larger tiles than the CPU's cache-blocked kernels. Large bases push most
        // tensors to a single tile; the per-row/attention caps stay at their
        // defaults for now (matmul + elementwise are the realized GPU paths).
        .cuda, .metal, .vulkan, .webgpu => .{
            .target_kind = target.kind,
            .base_square_2d = GPU_MACRO_TILE_CAP,
            .base_1d = 1 << 24, // 16M elems → typical vectors are a single tile
            // Attention on GPU streams keys inside one kernel (shared-memory
            // online softmax), so a slice's whole q/k/v/out should land in ONE
            // tile — the caps that bound the CPU kernels' stack scratch don't
            // apply. The GPU attention exec binds one buffer per operand per
            // slice and rejects multi-tile last-two-dims with Unsupported.
            .attn_q_tile = GPU_MACRO_TILE_CAP,
            .attn_k_tile_cap = GPU_MACRO_TILE_CAP,
            .attn_v_tile_cap = GPU_MACRO_TILE_CAP,
            .attn_head_tile_cap = GPU_MACRO_TILE_CAP,
            .attn_max_q_rows = GPU_MACRO_TILE_CAP,
        },
    };
}

pub fn chooseTileShape1D(policy: TilePolicy, n: usize) [1]usize {
    const t: usize = @min(policy.base_1d, if (n == 0) 1 else n);
    return .{@max(@as(usize, 1), t)};
}

pub fn chooseTileShape2DSquare(policy: TilePolicy, m: usize, n: usize) [2]usize {
    // Default to square tiles for ease of transpose materialization, but avoid
    // pathological tiling for skinny matrices (e.g. [1, 256]) where a square
    // heuristic would yield 1x1 tiles and explode tile counts.
    //
    // For vector-like shapes, tile along the long axis using base_1d so common
    // batch==1 workloads don't spend their time in tile-loop overhead.
    if (m <= 1 and n <= 1) return .{ 1, 1 };
    if (m <= 1) {
        const tn: usize = @max(@as(usize, 1), @min(n, policy.base_1d));
        return .{ 1, tn };
    }
    if (n <= 1) {
        const tm: usize = @max(@as(usize, 1), @min(m, policy.base_1d));
        return .{ tm, 1 };
    }

    const side: usize = @max(@as(usize, 1), @min(policy.base_square_2d, @min(m, n)));
    return .{ side, side };
}

pub fn chooseSoftmaxTiles(policy: TilePolicy, m: usize, n: usize) struct { tm: usize, tn: usize } {
    // Softmax is row-wise and needs per-row scratch. Prefer modest row tiles.
    // Keep tm small to allow stack scratch in exec and increase parallelism.
    const tm_cap: usize = policy.softmax_row_cap;
    const tm: usize = @max(@as(usize, 1), @min(tm_cap, @min(m, policy.base_1d)));

    // Favor wide tiles along the reduction axis to reduce tile overhead.
    // Cap tn to base_square_2d to avoid giant tiles.
    const tn: usize = @max(@as(usize, 1), @min(n, policy.base_square_2d));
    return .{ .tm = tm, .tn = tn };
}

pub fn chooseNormTiles(policy: TilePolicy, m: usize, n: usize) struct { tm: usize, tn: usize } {
    // LayerNorm/RMSNorm are row-wise and need per-row scratch. Keep tm small.
    const tm_cap: usize = policy.softmax_row_cap;
    const tm: usize = @max(@as(usize, 1), @min(tm_cap, @min(m, policy.base_1d)));
    // Favor decent width along last dim; cap to base_square_2d like softmax.
    const tn: usize = @max(@as(usize, 1), @min(n, policy.base_square_2d));
    return .{ .tm = tm, .tn = tn };
}

pub fn chooseAttentionTiles(policy: TilePolicy, m: usize, n: usize, dk: usize, dv: usize) struct { tm: usize, tn: usize, tk: usize, tv: usize } {
    // GPU targets: one tile per slice dimension (the kernel streams keys and
    // holds no per-tile stack scratch), and no ×16 rounding — a split dv/dk
    // tile would force the exec into multi-buffer bindings for no benefit.
    if (policy.target_kind != .cpu) {
        const tm_g: usize = @max(@as(usize, 1), @min(m, @min(policy.attn_q_tile, policy.base_1d)));
        const tn_g: usize = @max(@as(usize, 1), @min(n, @min(policy.attn_k_tile_cap, policy.base_square_2d)));
        const tk_g: usize = @max(@as(usize, 1), @min(dk, policy.attn_head_tile_cap));
        const tv_g: usize = @max(@as(usize, 1), @min(dv, policy.attn_v_tile_cap));
        return .{ .tm = tm_g, .tn = tn_g, .tk = tk_g, .tv = tv_g };
    }

    // Attention is row-wise over queries, and needs per-query scratch.
    // IMPORTANT: we parallelize attention primarily across query tiles.
    // Keep tm modest so we have enough tile-level parallelism on many-core CPUs.
    const tm_cap: usize = policy.attn_q_tile;
    const tm: usize = @max(@as(usize, 1), @min(tm_cap, @min(m, policy.base_1d)));

    // Block keys modestly so per-row score scratch stays small.
    const tn_cap: usize = policy.attn_k_tile_cap;
    const tn: usize = @max(@as(usize, 1), @min(n, @min(tn_cap, policy.base_square_2d)));

    // Block value/output columns to bound the accumulator scratch.
    const tv_cap: usize = policy.attn_v_tile_cap;
    var tv_target: usize = @min(dv, @min(tv_cap, policy.base_square_2d));
    if (tv_target >= 16) {
        tv_target = roundDownToMultiple(tv_target, 16);
        if (tv_target == 0) tv_target = @min(@as(usize, 16), dv);
    }
    const tv: usize = @max(@as(usize, 1), @min(tv_target, dv));

    // Block head dim reasonably; keep it SIMD-friendly.
    var tk_target: usize = @min(policy.attn_head_tile_cap, dk);
    if (tk_target >= 16) {
        tk_target = roundDownToMultiple(tk_target, 16);
        if (tk_target == 0) tk_target = @min(@as(usize, 16), dk);
    }
    const tk: usize = @max(@as(usize, 1), @min(tk_target, dk));

    return .{ .tm = tm, .tn = tn, .tk = tk, .tv = tv };
}

pub fn chooseConv1DTiles(policy: TilePolicy, l: usize, c_out: usize) struct { tl: usize, tc: usize } {
    // Small outputs: single tile (avoids per-tile overhead).
    if (l * c_out <= policy.small_tensor_threshold) return .{ .tl = l, .tc = c_out };

    // Conv1D is typically memory-friendly along the length axis (NLC) and many hot
    // cases (e.g. pointwise) benefit from larger M to amortize matmul overhead.
    // Allow Conv1D to use a larger length tile than the generic base_1d.
    const tl_cap: usize = std.math.mul(usize, policy.base_1d, 2) catch policy.base_1d;
    const tl: usize = @max(@as(usize, 1), @min(l, tl_cap));

    var tc_target: usize = @max(@as(usize, 1), @min(c_out, policy.base_square_2d));
    if (tc_target >= 16) {
        tc_target = roundDownToMultiple(tc_target, 16);
        if (tc_target == 0) tc_target = @min(@as(usize, 16), c_out);
    }
    const tc: usize = @max(@as(usize, 1), @min(c_out, tc_target));
    return .{ .tl = tl, .tc = tc };
}

pub fn chooseConv2DTiles(policy: TilePolicy, h: usize, w: usize, c_out: usize) struct { th: usize, tw: usize, tc: usize } {
    const side: usize = @max(@as(usize, 1), @min(policy.base_square_2d, @min(h, w)));
    const th: usize = @max(@as(usize, 1), @min(h, side));
    const tw: usize = @max(@as(usize, 1), @min(w, side));

    var tc_target: usize = @max(@as(usize, 1), @min(c_out, policy.base_square_2d));
    if (tc_target >= 16) {
        tc_target = roundDownToMultiple(tc_target, 16);
        if (tc_target == 0) tc_target = @min(@as(usize, 16), c_out);
    }
    const tc: usize = @max(@as(usize, 1), @min(c_out, tc_target));
    return .{ .th = th, .tw = tw, .tc = tc };
}

pub fn roundDownToMultiple(x: usize, m: usize) usize {
    if (m == 0) return x;
    return x - (x % m);
}

pub fn roundUpToMultiple(x: usize, m: usize) usize {
    if (m == 0) return x;
    const r: usize = x % m;
    if (r == 0) return x;
    return x + (m - r);
}

pub fn chooseMatMulTk(policy: TilePolicy, k: usize, b_dtype: DType) usize {
    // GPU: keep K in a single (large, bounded) tile so the kernel accumulates all
    // of K within one dispatch instead of round-tripping C through global memory
    // per k-tile. Respect quant block alignment.
    if (policy.target_kind != .cpu) {
        const cap: usize = @min(k, GPU_K_TILE_CAP);
        if (!b_dtype.info().is_quantized) return @max(@as(usize, 1), cap);
        const be: usize = b_dtype.info().block_elems;
        return @max(be, roundDownToMultiple(cap, be));
    }

    // Heuristic: try to keep K tiles reasonably sized, but ensure quant alignment.
    if (b_dtype == .f16) {
        // For f16, prefer larger K tiles to reduce K-split overhead (especially
        // when the execution path accumulates in f32 and converts outputs).
        const base_f16: usize = @min(@as(usize, 512), @max(policy.quant_k_block, k));
        return @min(base_f16, k);
    }

    const base: usize = @min(@as(usize, 256), @max(policy.quant_k_block, k));
    if (!b_dtype.info().is_quantized) return @min(base, k);

    const be: usize = b_dtype.info().block_elems;
    const want: usize = roundDownToMultiple(@min(base, k), be);
    return @max(be, want);
}

pub fn chooseMatMulTiles(policy: TilePolicy, m: usize, n: usize, k: usize, b_dtype: DType) struct { tm: usize, tn: usize, tk: usize } {
    // Macro tiles: choose C tiles.
    //
    // Important trade-off:
    // - Larger tiles improve per-tile kernel efficiency and amortize packing.
    // - But too-large tiles reduce the number of output tiles, which limits parallelism
    //   for tiled program execution (we intentionally avoid intra-tile parallel writes).
    //
    // Heuristic (v1): for matrices >= 512, cap the tile side to 128 to ensure enough
    // tile-level parallelism on many-core CPUs.
    // Special-case very skinny matrices (e.g. matvec with m==1): a square tile would
    // degenerate to tn==1, which is disastrous for kernel efficiency and tile overhead.
    //
    // Heuristic: when one dimension is tiny, tile fully along that dim and use a wide
    // tile along the other dim (rounded to SIMD-friendly multiples of 16).
    // GPU: few, large tiles. One ≤cap×cap output tile (typically a single tile for
    // moderate shapes → one dispatch) with full-or-large K. No CPU 128-cap (that
    // exists to spread work across cores; the GPU parallelizes within a dispatch).
    if (policy.target_kind != .cpu) {
        const tm: usize = @max(@as(usize, 1), @min(m, GPU_MACRO_TILE_CAP));
        const tn: usize = @max(@as(usize, 1), @min(n, GPU_MACRO_TILE_CAP));
        const tk: usize = chooseMatMulTk(policy, k, b_dtype);
        return .{ .tm = tm, .tn = tn, .tk = tk };
    }

    if (m <= 4 and n >= 64) {
        const tm: usize = @max(@as(usize, 1), m);

        // Keep N tiles reasonably large to amortize per-tile overhead, but not so large
        // that we destroy parallelism. 256 matches CPU kernel NC.
        var tn_target: usize = @min(@as(usize, 256), n);
        if (tn_target >= 16) {
            tn_target = roundDownToMultiple(tn_target, 16);
            if (tn_target == 0) tn_target = 16;
        }
        const tn: usize = @max(@as(usize, 1), @min(tn_target, n));

        const tk: usize = chooseMatMulTk(policy, k, b_dtype);
        return .{ .tm = tm, .tn = tn, .tk = tk };
    }
    if (n <= 4 and m >= 64) {
        const tn: usize = @max(@as(usize, 1), n);
        var tm_target: usize = @min(@as(usize, 256), m);
        if (tm_target >= 16) {
            tm_target = roundDownToMultiple(tm_target, 16);
            if (tm_target == 0) tm_target = 16;
        }
        const tm: usize = @max(@as(usize, 1), @min(tm_target, m));
        const tk: usize = chooseMatMulTk(policy, k, b_dtype);
        return .{ .tm = tm, .tn = tn, .tk = tk };
    }

    // Default: square-ish tiles.
    const min_mn: usize = @min(m, n);
    var side: usize = @max(@as(usize, 1), @min(policy.base_square_2d, min_mn));
    if (side > 128 and min_mn >= 512) {
        // f16 matmul currently routes through packed f32 kernels; larger macro-tiles
        // reduce repeated conversion/packing overhead across many tiny C tiles.
        if (b_dtype == .f16) {
            side = @min(side, 256);
        } else {
            side = 128;
        }
    }
    const tm: usize = @min(side, m);
    const tn: usize = @min(side, n);
    const tk: usize = chooseMatMulTk(policy, k, b_dtype);
    return .{ .tm = tm, .tn = tn, .tk = tk };
}
