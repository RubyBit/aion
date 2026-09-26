// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");
const types = @import("../backend/types.zig");

pub const TensorId = u32;
pub const BlockId = u32;
const MAX_RANK: usize = 8;
const MAX_CONCAT_INPUTS: usize = 16;
pub const MAX_CONTROL_OUTPUTS: usize = 16;
pub const MAX_LOOP_CARRIED: usize = 16;

/// Physical execution location chosen by the placement pass.
///
/// Backend kind only: a program is compiled for one device and executed by the
/// session bound to it, so nothing here distinguishes two adapters of the same
/// kind. Running one model across several GPUs needs tensor ownership to be
/// per-device first — until that exists, an adapter index here would validate
/// nothing and imply support that is not present.
pub const Placement = struct {
    kind: types.BackendKind = .cpu,

    pub fn eql(a: Placement, b: Placement) bool {
        return a.kind == b.kind;
    }
};

pub const StepMatMul = struct { c: TensorId, a: TensorId, b: TensorId, alpha: f32, beta: f32 };
pub const ElementwiseBroadcastKind = enum(u8) {
    identical,
    scalar_a,
    scalar_b,
    contiguous_suffix_a,
    contiguous_suffix_b,
    generic,
};
pub const ElementwiseBroadcastPlan = struct {
    kind: ElementwiseBroadcastKind,
    output_rank: u8,
    a_broadcast_axes: u8,
    b_broadcast_axes: u8,
};
pub const StepElemwiseBinary = struct {
    op: types.ElemwiseBinaryOp,
    /// Read only when `op == .gate`: the activation applied to `a` before the multiply
    /// (`gate(a, b) = act(a) * b`). Meaningless for every other op.
    act: types.UnaryOp = .gelu,
    out: TensorId,
    a: TensorId,
    b: TensorId,
    broadcast: ElementwiseBroadcastPlan,
};
pub const StepUnary = struct { op: types.UnaryOp, out: TensorId, a: TensorId };
pub const StepSoftmax = struct { out: TensorId, a: TensorId, axis: i32 };
pub const StepConv1D = struct {
    out: TensorId,
    x: TensorId,
    w: TensorId,
    bias: ?TensorId,
    stride: usize,
    dilation: usize,
    pad_left: usize,
    pad_right: usize,
    pad_mode: types.PadMode,
    groups: usize,
};
pub const StepMaxPool2D = struct { out: TensorId, x: TensorId, opts: @import("../graph/window.zig").Pool2D };

pub const StepConv2D = struct {
    out: TensorId,
    x: TensorId,
    w: TensorId,
    bias: ?TensorId,
    stride_h: usize,
    stride_w: usize,
    dilation_h: usize,
    dilation_w: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_left: usize,
    pad_right: usize,
    pad_mode: types.PadMode,
    groups: usize,
};
pub const StepLayerNorm = struct { out: TensorId, x: TensorId, gamma: TensorId, beta: TensorId, eps: f32 };
/// `residual` folds a residual add into the apply pass: `out = residual + rmsnorm(x)*gamma
/// + beta`. It is a SCHEDULE, not a meaning — `opt/fuse_steps.zig` sets it from a
/// lowered norm-then-add pair, and no graph op or on-disk record mentions it, so one
/// `.aion` can still compile to a different schedule per backend.
///
/// When present, residual / x / out are identically shaped: the rowwise kernel contract,
/// enforced by `validateStep` rather than by the step being its own tag. A configuration of the norm, not a second
/// norm — which is what keeps residual, scale and the LayerNorm variants from becoming a
/// tag per combination.
pub const StepRMSNorm = struct {
    out: TensorId,
    x: TensorId,
    gamma: TensorId,
    beta: TensorId,
    eps: f32,
    residual: ?TensorId = null,
};
/// Grouped-query attention; see `graph.Op.Attention`.
///
/// Optional query positions and K/V lengths are independent.
pub const StepAttention = struct {
    out: TensorId,
    q: TensorId,
    k: TensorId,
    v: TensorId,
    query_positions: ?TensorId,
    kv_lengths: ?TensorId,
    scale: f32,
    window: @import("../graph/graph.zig").AttentionWindow,
    attn_logits_soft_cap: f32,
};
/// Relative-positional multi-head self-attention (Transformer-XL / Conformer).
///
/// Shapes:
/// - q, k, v, out: [B, H, T*, D] (q rows T_q; k/v rows T_kv)
/// - pos_emb:      [H, P, D]
/// - pos_bias_u/_v:[H, D]
/// - mask (opt):   [T_q, T_kv] additive
pub const StepRelPosMHA = struct {
    out: TensorId,
    q: TensorId,
    k: TensorId,
    v: TensorId,
    pos_emb: TensorId,
    pos_bias_u: TensorId,
    pos_bias_v: TensorId,
    mask: ?TensorId,
    scale: f32,
    window: @import("../graph/graph.zig").AttentionWindow = .full,
    relative_zero_index: usize,
    attn_logits_soft_cap: f32 = 0,
};
pub const StepReduceAll = struct { op: types.ReduceOp, out: TensorId, a: TensorId };
pub const StepReduceAxis = struct { op: types.ReduceOp, out: TensorId, a: TensorId, axis: usize };
/// ArgMax over the last axis (v1): out (i32) = index of max of a along axis.
pub const StepArgMax = struct { out: TensorId, a: TensorId, axis: usize };
/// The `k` best values along the last axis and the index each came from, both
/// sorted best-first with ties going to the lowest index.
pub const StepTopK = struct { values: TensorId, indices: TensorId, a: TensorId, k: usize, axis: usize, largest: bool };
/// In-place row scatter: buf[idx] = src. Output aliases buf (set in lowering).
pub const StepScatterRow = struct { buf: TensorId, idx: TensorId, src: TensorId };
pub const StepConcatScalar = struct { out: TensorId, axis: usize, input_count: u8, inputs: [MAX_CONCAT_INPUTS]TensorId };
pub const StepCopy = struct { dst: TensorId, src: TensorId };

/// Gather rows from a 2D table using i32 indices.
///
/// Shapes:
/// - table:   [V, D] (f16/f32)
/// - indices: [B, L] (i32)
/// - out:     [B, L, D]
pub const StepGatherRows = struct { out: TensorId, table: TensorId, indices: TensorId };

/// Rotary positional embedding over 1D positions.
///
/// Shapes:
/// - x:         [B, L, N, H] (f16/f32)
/// - positions: [B, L] (i32)
/// - out:       [B, L, N, H]
pub const StepRoPE1D = struct {
    out: TensorId,
    x: TensorId,
    positions: TensorId,
    base_frequency: f32,
    scale_factor: f32,
    rope_proportion: f32,
};

/// General row gather. The first implementation supports the canonical
/// `axis == batch_dims` forms used by embeddings and batched sequence pooling.
/// General gather: `out = data[:axis] ++ indices[batch_dims:] ++ data[axis+1:]`.
/// The row/embedding steps above cover the large (possibly chunked) tables; this one
/// covers every remaining axis/batch_dims/rank combination.
pub const StepGatherND = struct {
    out: TensorId,
    data: TensorId,
    indices: TensorId,
    axis: u8,
    batch_dims: u8,
};

pub const StepGather = struct {
    out: TensorId,
    data: TensorId,
    indices: TensorId,
    axis: u8,
    batch_dims: u8,
};

/// In-place KV cache append.
///
/// Shapes:
/// - cache:     [B, T, H_kv, D] (f16/f32), written in-place
/// - new_kv:    [B, new_len, H_kv, D] (f16/f32), read-only
/// - end_index: [B] (i32), per-batch append start offsets
pub const StepSequenceAppend = struct {
    cache: TensorId,
    new_kv: TensorId,
    /// The append position, consumed as a device buffer by any backend that
    /// implements this op. Cache capacity is settled before specialization
    /// (see `Model.settleSequenceCaches`), so nothing reads it on the host.
    end_index: TensorId,
};

/// Elementwise scalar-dtype cast (f16 <-> f32 in v1).
pub const StepCast = struct { out: TensorId, x: TensorId, to_dtype: types.DType };

/// Matmul with B conceptually transposed: C[m,n] = sum_k A[m,k] * B[n,k].
///
/// Shapes:
/// - a: f32 with trailing axis `K` (any leading rank; C's leading axes match A's).
/// - b: q8_0 `[N, K]` with `quant_axis == 1` (per-row blocks).
/// - c: f32 `[..., N]`.
pub const StepMatMulNT = struct {
    c: TensorId,
    a: TensorId,
    b: TensorId,
    alpha: f32,
    beta: f32,
};

pub const StepLSTMCellFused = struct {
    out_state: TensorId,
    x: TensorId,
    h_prev: TensorId,
    c_prev: TensorId,
    w_ih: TensorId,
    w_hh: TensorId,
    b_ih: ?TensorId,
    b_hh: ?TensorId,
};

/// Branch control flow over a scalar i32 predicate (`0=false`, non-zero=true`).
///
/// Both branch blocks are expected to materialize `output_count` tensors with
/// identical dtype/shape/layout. After the selected block runs, selected branch
/// result buffers are swapped into `outputs` when the store supports no-copy
/// swapping. This preserves SSA-style branch-local outputs without copying data.
pub const StepIf = struct {
    cond: TensorId,
    then_block: BlockId,
    else_block: BlockId,
    output_count: u8,
    outputs: [MAX_CONTROL_OUTPUTS]TensorId,
    then_outputs: [MAX_CONTROL_OUTPUTS]TensorId,
    else_outputs: [MAX_CONTROL_OUTPUTS]TensorId,
};

/// Loop control flow with fixed storage and shape-invariant carried tensors.
///
/// The body writes next-state tensors into `body_carried_outputs`. After each
/// body iteration, those tensors are swapped with `carried`, so the next
/// iteration sees the new state without copying the tensor contents. Swapping
/// the buffers also leaves the body output tensors available as scratch storage
/// for the next iteration.
pub const StepLoop = struct {
    trip_count: ?TensorId,
    static_max_trip_count: usize,
    cond: ?TensorId,
    check_before: bool,
    body_block: BlockId,
    carried_count: u8,
    carried: [MAX_LOOP_CARRIED]TensorId,
    body_carried_outputs: [MAX_LOOP_CARRIED]TensorId,
};

pub const StepRFFT = struct {
    out: TensorId,
    x: TensorId,
    /// Real FFT length (power of two); equals the input's last dimension.
    n_fft: usize,
};

pub const StepSTFT = struct {
    out: TensorId,
    signal: TensorId,
    window: TensorId,
    n_fft: usize,
    hop_length: usize,
    center: bool,
    num_frames: usize,
};

/// View materializations (scalar only in v0).
pub const StepReshapeScalar = struct { dst: TensorId, src: TensorId };
pub const StepTranspose2DScalar = struct { dst: TensorId, src: TensorId };
pub const StepSliceNDScalar = struct { dst: TensorId, src: TensorId, rank: u8, starts: [MAX_RANK]usize };

/// An explicit value crossing between placements.  Transfer lowering decides
/// whether this becomes a copy, a unified-memory alias + synchronization, or
/// disappears because both placements are identical.
pub const StepTransfer = struct {
    dst: TensorId,
    src: TensorId,
    source: Placement,
    destination: Placement,
};

pub const Step = union(enum) {
    MatMul: StepMatMul,
    ElemwiseBinary: StepElemwiseBinary,
    Unary: StepUnary,
    Softmax: StepSoftmax,
    Conv1D: StepConv1D,
    Conv2D: StepConv2D,
    MaxPool2D: StepMaxPool2D,
    LayerNorm: StepLayerNorm,
    RMSNorm: StepRMSNorm,
    Attention: StepAttention,
    RelPosMHA: StepRelPosMHA,
    ArgMax: StepArgMax,
    TopK: StepTopK,
    ScatterRow: StepScatterRow,
    ReduceAll: StepReduceAll,
    ReduceAxis: StepReduceAxis,
    ConcatScalar: StepConcatScalar,
    Copy: StepCopy,

    GatherRows: StepGatherRows,
    Gather: StepGather,
    GatherND: StepGatherND,

    RoPE1D: StepRoPE1D,

    SequenceAppend: StepSequenceAppend,

    LSTMCellFused: StepLSTMCellFused,

    If: StepIf,
    Loop: StepLoop,

    RFFT: StepRFFT,
    STFT: StepSTFT,

    ReshapeScalar: StepReshapeScalar,
    Transpose2DScalar: StepTranspose2DScalar,
    SliceNDScalar: StepSliceNDScalar,

    Cast: StepCast,
    MatMulNT: StepMatMulNT,
    Transfer: StepTransfer,
};

/// A scheduled operation, the placement at which it executes, and which of its
/// operands the executing backend reads on the host.
///
/// `host_operands` is a bitmask over `tensorUses(&op).slice()`: a set bit means
/// the runtime resolves that operand's value on the CPU while stepping the
/// schedule — control-flow predicates, and nothing else. Such an operand must be
/// CPU-placed; every other operand must match `placement`. The compiler sets the
/// mask from `controlHostOperands` and inserts the transfers that make it true,
/// so a host read can never silently observe a stale device value. Kernels get
/// no say: an operand a backend cannot consume on the device makes the op
/// unsupported there, rather than quietly moving work to the CPU.
pub const PlacedStep = struct {
    origin: ?@import("../diagnostic.zig").Origin = null,
    placement: Placement = .{},
    host_operands: u64 = 0,
    op: Step,

    pub fn readsOperandOnHost(self: PlacedStep, index: usize) bool {
        return index < 64 and (self.host_operands & (@as(u64, 1) << @intCast(index))) != 0;
    }
};

pub const Access = enum { read, write, read_write };
pub const TensorUse = struct { id: *TensorId, access: Access };

/// Fixed-capacity operand description.  The largest payload is If (predicate
/// plus three lists of 16 tensors).  Capacity matches the `host_operands` mask
/// width so every operand is addressable by it.
pub const TensorUses = struct {
    items: [64]TensorUse = undefined,
    len: usize = 0,

    fn add(self: *TensorUses, id: *TensorId, access: Access) void {
        std.debug.assert(self.len < self.items.len);
        self.items[self.len] = .{ .id = id, .access = access };
        self.len += 1;
    }

    fn addOptional(self: *TensorUses, id: *?TensorId, access: Access) void {
        if (id.* != null) self.add(&id.*.?, access);
    }

    pub fn slice(self: *const TensorUses) []const TensorUse {
        return self.items[0..self.len];
    }
};

/// Complete, explicit operand roles, as pointers into the step so one walk
/// serves reading, placement validation, and operand rewriting.  Deliberately
/// not reflection: TensorId and BlockId are both u32, so field walking cannot
/// tell a storage dependency from a control-flow block id.
pub fn tensorUses(step: *Step) TensorUses {
    var out: TensorUses = .{};
    switch (step.*) {
        .MatMul => |*s| {
            out.add(&s.c, if (s.beta == 0) .write else .read_write);
            out.add(&s.a, .read);
            out.add(&s.b, .read);
        },
        .ElemwiseBinary => |*s| {
            out.add(&s.out, .write);
            out.add(&s.a, .read);
            out.add(&s.b, .read);
        },
        .Unary => |*s| {
            out.add(&s.out, .write);
            out.add(&s.a, .read);
        },
        .Softmax => |*s| {
            out.add(&s.out, .write);
            out.add(&s.a, .read);
        },
        .Conv1D => |*s| {
            out.add(&s.out, .write);
            out.add(&s.x, .read);
            out.add(&s.w, .read);
            out.addOptional(&s.bias, .read);
        },
        .MaxPool2D => |*s| {
            out.add(&s.out, .write);
            out.add(&s.x, .read);
        },
        .Conv2D => |*s| {
            out.add(&s.out, .write);
            out.add(&s.x, .read);
            out.add(&s.w, .read);
            out.addOptional(&s.bias, .read);
        },
        .LayerNorm => |*s| {
            out.add(&s.out, .write);
            out.add(&s.x, .read);
            out.add(&s.gamma, .read);
            out.add(&s.beta, .read);
        },
        .RMSNorm => |*s| {
            out.add(&s.out, .write);
            out.add(&s.x, .read);
            out.add(&s.gamma, .read);
            out.add(&s.beta, .read);
            // Appended last on purpose: `host_operands` is a positional mask, so an
            // optional operand may not shift the index of any operand before it.
            out.addOptional(&s.residual, .read);
        },
        .Attention => |*s| {
            out.add(&s.out, .write);
            out.add(&s.q, .read);
            out.add(&s.k, .read);
            out.add(&s.v, .read);
            out.addOptional(&s.query_positions, .read);
            out.addOptional(&s.kv_lengths, .read);
        },
        .RelPosMHA => |*s| {
            out.add(&s.out, .write);
            out.add(&s.q, .read);
            out.add(&s.k, .read);
            out.add(&s.v, .read);
            out.add(&s.pos_emb, .read);
            out.add(&s.pos_bias_u, .read);
            out.add(&s.pos_bias_v, .read);
            out.addOptional(&s.mask, .read);
        },
        .ArgMax => |*s| {
            out.add(&s.out, .write);
            out.add(&s.a, .read);
        },
        .TopK => |*s| {
            out.add(&s.values, .write);
            out.add(&s.indices, .write);
            out.add(&s.a, .read);
        },
        .ScatterRow => |*s| {
            out.add(&s.buf, .read_write);
            out.add(&s.idx, .read);
            out.add(&s.src, .read);
        },
        .ReduceAll => |*s| {
            out.add(&s.out, .write);
            out.add(&s.a, .read);
        },
        .ReduceAxis => |*s| {
            out.add(&s.out, .write);
            out.add(&s.a, .read);
        },
        .ConcatScalar => |*s| {
            out.add(&s.out, .write);
            for (s.inputs[0..s.input_count]) |*id| out.add(id, .read);
        },
        .Copy => |*s| {
            out.add(&s.dst, .write);
            out.add(&s.src, .read);
        },
        .GatherRows => |*s| {
            out.add(&s.out, .write);
            out.add(&s.table, .read);
            out.add(&s.indices, .read);
        },
        .GatherND => |*s| {
            out.add(&s.out, .write);
            out.add(&s.data, .read);
            out.add(&s.indices, .read);
        },
        .Gather => |*s| {
            out.add(&s.out, .write);
            out.add(&s.data, .read);
            out.add(&s.indices, .read);
        },
        .RoPE1D => |*s| {
            out.add(&s.out, .write);
            out.add(&s.x, .read);
            out.add(&s.positions, .read);
        },
        .SequenceAppend => |*s| {
            out.add(&s.cache, .read_write);
            out.add(&s.new_kv, .read);
            out.add(&s.end_index, .read);
        },
        .LSTMCellFused => |*s| {
            out.add(&s.out_state, .write);
            out.add(&s.x, .read);
            out.add(&s.h_prev, .read);
            out.add(&s.c_prev, .read);
            out.add(&s.w_ih, .read);
            out.add(&s.w_hh, .read);
            out.addOptional(&s.b_ih, .read);
            out.addOptional(&s.b_hh, .read);
        },
        .If => |*s| {
            out.add(&s.cond, .read);
            const n: usize = @intCast(s.output_count);
            for (s.outputs[0..n]) |*id| out.add(id, .write);
            for (s.then_outputs[0..n]) |*id| out.add(id, .read);
            for (s.else_outputs[0..n]) |*id| out.add(id, .read);
        },
        .Loop => |*s| {
            out.addOptional(&s.trip_count, .read);
            out.addOptional(&s.cond, .read);
            const n: usize = @intCast(s.carried_count);
            for (s.carried[0..n]) |*id| out.add(id, .read_write);
            for (s.body_carried_outputs[0..n]) |*id| out.add(id, .read_write);
        },
        .RFFT => |*s| {
            out.add(&s.out, .write);
            out.add(&s.x, .read);
        },
        .STFT => |*s| {
            out.add(&s.out, .write);
            out.add(&s.signal, .read);
            out.add(&s.window, .read);
        },
        .ReshapeScalar => |*s| {
            out.add(&s.dst, .write);
            out.add(&s.src, .read);
        },
        .Transpose2DScalar => |*s| {
            out.add(&s.dst, .write);
            out.add(&s.src, .read);
        },
        .SliceNDScalar => |*s| {
            out.add(&s.dst, .write);
            out.add(&s.src, .read);
        },
        .Cast => |*s| {
            out.add(&s.out, .write);
            out.add(&s.x, .read);
        },
        .MatMulNT => |*s| {
            out.add(&s.c, if (s.beta == 0) .write else .read_write);
            out.add(&s.a, .read);
            out.add(&s.b, .read);
        },
        .Transfer => |*s| {
            out.add(&s.dst, .write);
            out.add(&s.src, .read);
        },
    }
    return out;
}

/// Bit for `field` within `uses`, by pointer identity — so a caller names the
/// operand it means instead of tracking its position in the walk.
pub fn operandMask(uses: TensorUses, field: *const TensorId) u64 {
    for (uses.slice(), 0..) |use, i| {
        if (use.id == field) return @as(u64, 1) << @intCast(i);
    }
    return 0;
}

/// Operands the runtime resolves on the CPU regardless of target: control-flow
/// predicates are evaluated by the executor stepping the schedule, never by a
/// kernel. This is the only source of host reads — see `PlacedStep`.
pub fn controlHostOperands(step: *Step) u64 {
    const uses = tensorUses(step);
    return switch (step.*) {
        .If => |*s| operandMask(uses, &s.cond),
        .Loop => |*s| (if (s.trip_count != null) operandMask(uses, &s.trip_count.?) else 0) |
            (if (s.cond != null) operandMask(uses, &s.cond.?) else 0),
        else => 0,
    };
}

pub const TensorPlacement = struct { id: TensorId, placement: Placement };
pub const PlacementError = error{ InvalidProgram, MissingPlacement, PlacementMismatch, MissingTransfer };

/// Validated executable schedule.
///
/// Contract:
/// - Produced by `graph/program.compileGraph` after full graph + storage validation.
/// - Execution must assume correctness and avoid per-step argument checks.
/// - Runtime can still fail due to backend errors (e.g. Unsupported) or storage errors.
pub const Block = struct {
    steps: []PlacedStep,
};

/// One physical allocation shared by non-overlapping logical workspace tensors.
/// Logical layout remains on each TensorId; the storage manager owns the backing
/// through the owner id and resolves every member to it.
pub const WorkspaceSlot = struct {
    owner: TensorId,
    members: []TensorId,
    /// Bytes of the one buffer, host or device: the largest member.
    bytes: usize,
    placement: Placement,
    /// True only when the compiler populated the owner (for example Iota/Dim)
    /// and materialization must preserve those bytes.
    preserve_contents: bool = false,
};

pub const ExecutableProgram = struct {
    allocator: std.mem.Allocator,
    steps: []PlacedStep,
    blocks: []Block = &[_]Block{},
    outputs: []TensorId,
    /// The placement every non-`Transfer` step executes at. Constant for the
    /// program's lifetime, so a backend checks it once instead of walking the
    /// schedule to re-derive what placement already decided.
    target: Placement = .{},
    tensor_placements: []TensorPlacement = &[_]TensorPlacement{},
    /// Storage allocated specifically for this compiled specialization. External
    /// inputs, parameters, and model state are excluded. The owning model uses
    /// this list to reclaim an evicted specialization's workspace.
    owned_tensors: []TensorId = &[_]TensorId{},
    workspace_slots: []WorkspaceSlot = &[_]WorkspaceSlot{},
    workspace_bytes: usize = 0,

    const Self = @This();

    /// Placements are sorted by tensor id at compile time, so this is a binary
    /// search rather than the scan a per-operand lookup would otherwise repeat.
    pub fn placementOf(self: *const Self, id: TensorId) ?Placement {
        var lo: usize = 0;
        var hi: usize = self.tensor_placements.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const entry = self.tensor_placements[mid];
            if (entry.id == id) return entry.placement;
            if (entry.id < id) lo = mid + 1 else hi = mid;
        }
        return null;
    }

    /// Validate the placed schedule recursively. Transfer is the only operation
    /// allowed to mention values at two different placements.
    pub fn validatePlacements(self: *const Self) PlacementError!void {
        try self.validateStepList(self.steps);
        for (self.blocks) |block| try self.validateStepList(block.steps);
    }

    fn validateStepList(self: *const Self, steps: []PlacedStep) PlacementError!void {
        for (steps) |*placed| {
            switch (placed.op) {
                .Transfer => |transfer| {
                    if (!placed.placement.eql(transfer.destination)) return error.PlacementMismatch;
                    const src = self.placementOf(transfer.src) orelse return error.MissingPlacement;
                    const dst = self.placementOf(transfer.dst) orelse return error.MissingPlacement;
                    if (!src.eql(transfer.source) or !dst.eql(transfer.destination)) return error.PlacementMismatch;
                },
                else => {
                    const uses = tensorUses(&placed.op);
                    for (uses.slice(), 0..) |use, i| {
                        const actual = self.placementOf(use.id.*) orelse return error.MissingPlacement;
                        if (placed.readsOperandOnHost(i)) {
                            // The value is consumed on the CPU, so it must live
                            // there — either host-owned, or transferred back.
                            if (actual.kind != .cpu) return error.PlacementMismatch;
                        } else if (use.access != .read) {
                            // A step writes only where it executes.
                            if (!actual.eql(placed.placement)) return error.MissingTransfer;
                        }
                        // A read-only operand may sit at the step's placement or
                        // on the host, where it is uploaded on use: nothing else
                        // writes it, so that read cannot go stale.
                    }
                },
            }
        }
    }

    pub fn deinit(self: *Self) void {
        self.allocator.free(self.steps);
        for (self.blocks) |block| {
            if (block.steps.len != 0) self.allocator.free(block.steps);
        }
        if (self.blocks.len != 0) self.allocator.free(self.blocks);
        self.allocator.free(self.outputs);
        if (self.tensor_placements.len != 0) self.allocator.free(self.tensor_placements);
        if (self.owned_tensors.len != 0) self.allocator.free(self.owned_tensors);
        for (self.workspace_slots) |slot| self.allocator.free(slot.members);
        if (self.workspace_slots.len != 0) self.allocator.free(self.workspace_slots);
        self.* = undefined;
    }
};

test "placement verifier requires a step to write where it executes" {
    const cpu: Placement = .{};
    const gpu: Placement = .{ .kind = .webgpu };
    var steps = [_]PlacedStep{.{
        .placement = gpu,
        .op = .{ .Unary = .{ .op = .relu, .out = 2, .a = 1 } },
    }};
    // `a` is read-only and host-owned, which is a legal upload; `out` is written
    // on the GPU and so cannot be CPU-placed.
    var placements = [_]TensorPlacement{
        .{ .id = 1, .placement = cpu },
        .{ .id = 2, .placement = cpu },
    };
    const program: ExecutableProgram = .{
        .allocator = std.testing.allocator,
        .steps = &steps,
        .outputs = @constCast(&[_]TensorId{2}),
        .tensor_placements = &placements,
    };
    try std.testing.expectError(error.MissingTransfer, program.validatePlacements());
}

test "placement verifier accepts an explicit transfer" {
    const cpu: Placement = .{};
    const gpu: Placement = .{ .kind = .webgpu };
    var steps = [_]PlacedStep{.{
        .placement = gpu,
        .op = .{ .Transfer = .{ .dst = 2, .src = 1, .source = cpu, .destination = gpu } },
    }};
    var placements = [_]TensorPlacement{
        .{ .id = 1, .placement = cpu },
        .{ .id = 2, .placement = gpu },
    };
    const program: ExecutableProgram = .{
        .allocator = std.testing.allocator,
        .steps = &steps,
        .outputs = @constCast(&[_]TensorId{2}),
        .tensor_placements = &placements,
    };
    try program.validatePlacements();
}

test "placement verifier forbids GPU executor host-control reads" {
    const gpu: Placement = .{ .kind = .webgpu };
    var steps = [_]PlacedStep{.{
        .placement = gpu,
        .host_operands = 1, // the If predicate is resolved on the CPU
        .op = .{ .If = .{
            .cond = 1,
            .then_block = 0,
            .else_block = 1,
            .output_count = 0,
            .outputs = @splat(0),
            .then_outputs = @splat(0),
            .else_outputs = @splat(0),
        } },
    }};
    var placements = [_]TensorPlacement{.{ .id = 1, .placement = gpu }};
    const program: ExecutableProgram = .{
        .allocator = std.testing.allocator,
        .steps = &steps,
        .outputs = @constCast(&[_]TensorId{}),
        .tensor_placements = &placements,
    };
    try std.testing.expectError(error.PlacementMismatch, program.validatePlacements());
}
