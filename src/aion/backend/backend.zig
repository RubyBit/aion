const std = @import("std");
const types = @import("types.zig");
const utils = @import("utils.zig");

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

pub const Backend = struct {
    ctx: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        kind: *const fn (ctx: *anyopaque) BackendKind,
        name: *const fn (ctx: *anyopaque) []const u8,
        caps: *const fn (ctx: *anyopaque) BackendCaps,
        deinit: *const fn (ctx: *anyopaque) void,

        /// v0 contract: out/a/b must be packed row-major.
        elemwiseBinary: *const fn (ctx: *anyopaque, op: ElemwiseBinaryOp, out: BufferViewMut, a: BufferViewConst, b: BufferViewConst) BackendError!void,

        /// v0 contract:
        /// - out/a must be packed row-major scalar tensors of same shape
        /// - b must be packed row-major scalar, rank-1 with shape = [last_dim]
        /// - broadcast is over all leading dimensions (batch/seq/etc.)
        ///
        /// Computes: out[..., j] = op(a[..., j], b[j])
        broadcastLastDimBinary: *const fn (ctx: *anyopaque, op: ElemwiseBinaryOp, out: BufferViewMut, a: BufferViewConst, b: BufferViewConst) BackendError!void,

        /// v0 contract: out/a must be packed row-major scalar tensors of same shape.
        relu: *const fn (ctx: *anyopaque, out: BufferViewMut, a: BufferViewConst) BackendError!void,

        /// v0 contract:
        /// - a must be packed row-major scalar
        /// - out must be packed row-major scalar with exactly 1 element
        /// - reduction is over all elements of a
        reduce: *const fn (ctx: *anyopaque, op: ReduceOp, out: BufferViewMut, a: BufferViewConst) BackendError!void,

        /// v0 contact: a/b/c must be packed row-major.
        matmul: *const fn (ctx: *anyopaque, params: MatMulParams, c: BufferViewMut, a: BufferViewConst, b: BufferViewConst) BackendError!void,

        /// Data movement primitive for pack/unpack nodes.
        /// Recommended v0 behavior:
        /// - Require same dtype
        /// - Allow packed<->packed always
        /// - Optionally allow strided<->packed if implemented (else return Unsupported)
        copy: *const fn (ctx: *anyopaque, dst: BufferViewMut, src: BufferViewConst) BackendError!void,
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

    pub fn elemwiseBinary(
        self: Backend,
        op: ElemwiseBinaryOp,
        out: BufferViewMut,
        a: BufferViewConst,
        b: BufferViewConst,
    ) BackendError!void {
        // Elemwise requires scalar (non-quant) types
        try utils.requirePackedScalar(out);
        try utils.requirePackedScalar(a);
        try utils.requirePackedScalar(b);
        try utils.requireSameShape(out.layout, a.layout);
        try utils.requireSameShape(out.layout, b.layout);
        if (out.dtype != a.dtype or out.dtype != b.dtype) return BackendError.InvalidArgument;
        return self.vtable.elemwiseBinary(self.ctx, op, out, a, b);
    }

    pub fn broadcastLastDimBinary(
        self: Backend,
        op: ElemwiseBinaryOp,
        out: BufferViewMut,
        a: BufferViewConst,
        b: BufferViewConst,
    ) BackendError!void {
        try utils.requirePackedScalar(out);
        try utils.requirePackedScalar(a);
        try utils.requirePackedScalar(b);

        try utils.requireSameShape(out.layout, a.layout);
        if (out.dtype != a.dtype or out.dtype != b.dtype) return BackendError.InvalidArgument;

        // b is rank-1 and matches last dimension of out.
        if (b.layout.rank != 1 or b.layout.shape.len != 1) return BackendError.InvalidArgument;
        if (out.layout.rank == 0 or out.layout.shape.len == 0) return BackendError.InvalidArgument;
        const last_dim: usize = out.layout.shape[out.layout.shape.len - 1];
        if (b.layout.shape[0] != last_dim) return BackendError.InvalidArgument;

        return self.vtable.broadcastLastDimBinary(self.ctx, op, out, a, b);
    }

    pub fn relu(
        self: Backend,
        out: BufferViewMut,
        a: BufferViewConst,
    ) BackendError!void {
        try utils.requirePackedScalar(out);
        try utils.requirePackedScalar(a);
        try utils.requireSameShape(out.layout, a.layout);
        if (out.dtype != a.dtype) return BackendError.InvalidArgument;
        return self.vtable.relu(self.ctx, out, a);
    }

    pub fn reduce(
        self: Backend,
        op: ReduceOp,
        out: BufferViewMut,
        a: BufferViewConst,
    ) BackendError!void {
        try utils.requirePackedScalar(out);
        try utils.requirePackedScalar(a);
        const out_elems: usize = try utils.elemCount(out.layout.shape);
        if (out_elems != 1) return BackendError.InvalidArgument;
        return self.vtable.reduce(self.ctx, op, out, a);
    }

    pub fn matmul(
        self: Backend,
        params: MatMulParams,
        c: BufferViewMut,
        a: BufferViewConst,
        b: BufferViewConst,
    ) BackendError!void {
        // C must be scalar (output is always dequantized)
        try utils.requirePackedScalar(c);
        // A must be scalar (activations)
        try utils.requirePackedScalar(a);
        // B can be scalar OR quantized (weights)
        try utils.requirePacked(b);
        try utils.requireMatMulShapes(params, c.layout, a.layout, b.layout);
        return self.vtable.matmul(self.ctx, params, c, a, b);
    }

    pub fn copy(
        self: Backend,
        dst: BufferViewMut,
        src: BufferViewConst,
    ) BackendError!void {
        try dst.layout.validate();
        try src.layout.validate();
        if (dst.dtype != src.dtype) return BackendError.InvalidArgument;
        try utils.requireSameShape(dst.layout, src.layout);
        // require packed -> v0: CPU backend only supports packed<->packed copies.
        try utils.requirePacked(dst);
        try utils.requirePacked(src);
        return self.vtable.copy(self.ctx, dst, src);
    }
};
