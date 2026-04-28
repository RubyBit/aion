// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const fast_math = @import("../kernels/fast_math.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

const MAX_HIDDEN: usize = 1024;
const MAX_INPUT: usize = 2048;

fn loadRowF32(store: tensor_store.TensorStore, meta: tensor_store.TensorMeta, id: tensor_store.TensorId, row: usize, out: []f32) ExecuteProgramError!void {
    if (meta.dtype != .f32) return BackendError.InvalidArgument;
    if (meta.rank != 2) return BackendError.InvalidArgument;
    if (row >= meta.shape[0]) return BackendError.InvalidArgument;
    if (out.len != meta.shape[1]) return BackendError.InvalidArgument;

    const ti0: usize = row / meta.tile_shape[0];
    const in0: usize = row - ti0 * meta.tile_shape[0];

    var col_off: usize = 0;
    var ti1: usize = 0;
    while (ti1 < meta.tile_counts[1]) : (ti1 += 1) {
        const tile = try store.acquireTileConst(id, ti0, ti1);
        defer store.releaseConst(tile.token);

        const view = tile.bufferView();
        if (view.dtype != .f32) return BackendError.InvalidArgument;
        if (view.layout.rank != 2) return BackendError.InvalidArgument;

        const m_tile: usize = view.layout.shape[0];
        const n_tile: usize = view.layout.shape[1];
        if (in0 >= m_tile) return BackendError.InvalidArgument;
        if (col_off + n_tile > out.len) return BackendError.InvalidArgument;

        const stride0_bytes_i: isize = view.layout.strides_bytes[0];
        const stride1_bytes_i: isize = view.layout.strides_bytes[1];
        if (stride0_bytes_i <= 0) return BackendError.InvalidArgument;
        if (stride1_bytes_i != 4) return BackendError.InvalidArgument;
        const stride0_bytes: usize = @intCast(stride0_bytes_i);

        const row_bytes: usize = n_tile * 4;
        const src_off: usize = in0 * stride0_bytes;
        if (src_off + row_bytes > view.bytes.len) return BackendError.InvalidArgument;

        const dst: []u8 = std.mem.sliceAsBytes(out[col_off .. col_off + n_tile]);
        @memcpy(dst, view.bytes[src_off .. src_off + row_bytes]);
        col_off += n_tile;
    }

    if (col_off != out.len) return BackendError.InvalidArgument;
}

fn loadVecF32(store: tensor_store.TensorStore, meta: tensor_store.TensorMeta, id: tensor_store.TensorId, out: []f32) ExecuteProgramError!void {
    if (meta.dtype != .f32) return BackendError.InvalidArgument;
    if (meta.rank != 1) return BackendError.InvalidArgument;
    if (out.len != meta.shape[0]) return BackendError.InvalidArgument;

    var off: usize = 0;
    var ti0: usize = 0;
    while (ti0 < meta.tile_counts[0]) : (ti0 += 1) {
        const tile = try store.acquireTileConst(id, ti0, 0);
        defer store.releaseConst(tile.token);

        const view = tile.bufferView();
        if (view.dtype != .f32) return BackendError.InvalidArgument;
        if (view.layout.rank != 1) return BackendError.InvalidArgument;
        const stride0_bytes_i: isize = view.layout.strides_bytes[0];
        if (stride0_bytes_i != 4) return BackendError.InvalidArgument;
        const n_tile: usize = view.layout.shape[0];
        if (off + n_tile > out.len) return BackendError.InvalidArgument;

        const bytes: usize = n_tile * 4;
        if (bytes > view.bytes.len) return BackendError.InvalidArgument;

        const dst: []u8 = std.mem.sliceAsBytes(out[off .. off + n_tile]);
        @memcpy(dst, view.bytes[0..bytes]);
        off += n_tile;
    }

    if (off != out.len) return BackendError.InvalidArgument;
}

fn storeRowStateF32(
    store: tensor_store.TensorStore,
    meta: tensor_store.TensorMeta,
    id: tensor_store.TensorId,
    row: usize,
    h: []const f32,
    c: []const f32,
) ExecuteProgramError!void {
    if (meta.dtype != .f32) return BackendError.InvalidArgument;
    if (meta.rank != 2) return BackendError.InvalidArgument;
    if (row >= meta.shape[0]) return BackendError.InvalidArgument;
    if (meta.shape[1] != h.len + c.len) return BackendError.InvalidArgument;

    const ti0: usize = row / meta.tile_shape[0];
    const in0: usize = row - ti0 * meta.tile_shape[0];

    var col_off: usize = 0;
    var ti1: usize = 0;
    while (ti1 < meta.tile_counts[1]) : (ti1 += 1) {
        var tile = try store.acquireTileMut(id, ti0, ti1);
        defer store.releaseMut(tile.token);

        const view = tile.bufferView();
        if (view.dtype != .f32) return BackendError.InvalidArgument;
        if (view.layout.rank != 2) return BackendError.InvalidArgument;

        const m_tile: usize = view.layout.shape[0];
        const n_tile: usize = view.layout.shape[1];
        if (in0 >= m_tile) return BackendError.InvalidArgument;

        const stride0_bytes_i: isize = view.layout.strides_bytes[0];
        const stride1_bytes_i: isize = view.layout.strides_bytes[1];
        if (stride0_bytes_i <= 0) return BackendError.InvalidArgument;
        if (stride1_bytes_i != 4) return BackendError.InvalidArgument;
        const stride0_bytes: usize = @intCast(stride0_bytes_i);

        const row_bytes: usize = n_tile * 4;
        const dst_off: usize = in0 * stride0_bytes;
        if (dst_off + row_bytes > view.bytes.len) return BackendError.InvalidArgument;

        // Fill the output row segment from either h or c depending on position.
        var tmp: [MAX_HIDDEN * 2]f32 = undefined;
        if (n_tile > tmp.len) return BackendError.InvalidArgument;

        var j: usize = 0;
        while (j < n_tile) : (j += 1) {
            const g: usize = col_off + j;
            tmp[j] = if (g < h.len) h[g] else c[g - h.len];
        }

        const src: []const u8 = std.mem.sliceAsBytes(tmp[0..n_tile]);
        @memcpy(view.bytes[dst_off .. dst_off + row_bytes], src);
        col_off += n_tile;
    }

    if (col_off != meta.shape[1]) return BackendError.InvalidArgument;
}

fn accumMatVecTiled(
    store: tensor_store.TensorStore,
    w_meta: tensor_store.TensorMeta,
    w_id: tensor_store.TensorId,
    x: []const f32,
    out: []f32,
) ExecuteProgramError!void {
    if (w_meta.dtype != .f32) return BackendError.InvalidArgument;
    if (w_meta.rank != 2) return BackendError.InvalidArgument;
    if (x.len != w_meta.shape[0]) return BackendError.InvalidArgument;
    if (out.len != w_meta.shape[1]) return BackendError.InvalidArgument;

    var ti0: usize = 0;
    while (ti0 < w_meta.tile_counts[0]) : (ti0 += 1) {
        var ti1: usize = 0;
        while (ti1 < w_meta.tile_counts[1]) : (ti1 += 1) {
            const tile = try store.acquireTileConst(w_id, ti0, ti1);
            defer store.releaseConst(tile.token);

            const view = tile.bufferView();
            if (view.dtype != .f32) return BackendError.InvalidArgument;
            if (view.layout.rank != 2) return BackendError.InvalidArgument;

            const stride0_bytes_i: isize = view.layout.strides_bytes[0];
            const stride1_bytes_i: isize = view.layout.strides_bytes[1];
            if (stride0_bytes_i <= 0) return BackendError.InvalidArgument;
            if (stride1_bytes_i != 4) return BackendError.InvalidArgument;
            const stride0_bytes: usize = @intCast(stride0_bytes_i);
            const stride1_bytes: usize = @intCast(stride1_bytes_i);

            const base_r: usize = ti0 * w_meta.tile_shape[0];
            const base_c: usize = ti1 * w_meta.tile_shape[1];

            const m_tile: usize = view.layout.shape[0];
            const n_tile: usize = view.layout.shape[1];
            if (base_c + n_tile > out.len) return BackendError.InvalidArgument;

            const needed_bytes: usize = (m_tile - 1) * stride0_bytes + (n_tile - 1) * stride1_bytes + 4;
            if (needed_bytes > view.bytes.len) return BackendError.InvalidArgument;

            var lr: usize = 0;
            while (lr < m_tile) : (lr += 1) {
                const gr: usize = base_r + lr;
                if (gr >= x.len) break;
                const xv: f32 = x[gr];
                if (xv == 0.0) continue;

                const row_base_off: usize = lr * stride0_bytes;
                var lc: usize = 0;
                while (lc < n_tile) : (lc += 1) {
                    const off_bytes: usize = row_base_off + lc * stride1_bytes;
                    const p: *align(1) const f32 = @ptrCast(view.bytes.ptr + off_bytes);
                    out[base_c + lc] += xv * p.*;
                }
            }
        }
    }
}

pub fn execLSTMCellFused(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepLSTMCellFused,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    _ = pool;
    _ = thread_count;

    const out_meta = try store.meta(s.out_state);
    const x_meta = try store.meta(s.x);
    const h_meta = try store.meta(s.h_prev);
    const c_meta = try store.meta(s.c_prev);
    const wih_meta = try store.meta(s.w_ih);
    const whh_meta = try store.meta(s.w_hh);

    // v0: only f32 fused implementation.
    if (out_meta.dtype != .f32) return BackendError.Unsupported;

    if (out_meta.rank != 2 or x_meta.rank != 2 or h_meta.rank != 2 or c_meta.rank != 2) return BackendError.InvalidArgument;
    if (wih_meta.rank != 2 or whh_meta.rank != 2) return BackendError.InvalidArgument;

    const batch: usize = x_meta.shape[0];
    const input_size: usize = x_meta.shape[1];
    const hidden: usize = h_meta.shape[1];

    if (batch == 0 or input_size == 0 or hidden == 0) return BackendError.InvalidArgument;
    if (input_size > MAX_INPUT or hidden > MAX_HIDDEN) return BackendError.InvalidArgument;

    const gate_dim: usize = hidden * 4;
    if (wih_meta.shape[0] != input_size or wih_meta.shape[1] != gate_dim) return BackendError.InvalidArgument;
    if (whh_meta.shape[0] != hidden or whh_meta.shape[1] != gate_dim) return BackendError.InvalidArgument;

    if (h_meta.shape[0] != batch or c_meta.shape[0] != batch) return BackendError.InvalidArgument;
    if (c_meta.shape[1] != hidden) return BackendError.InvalidArgument;

    if (out_meta.shape[0] != batch or out_meta.shape[1] != hidden * 2) return BackendError.InvalidArgument;

    var b_ih_buf: [MAX_HIDDEN * 4]f32 = undefined;
    var b_hh_buf: [MAX_HIDDEN * 4]f32 = undefined;
    const has_bias: bool = (s.b_ih != null);
    if (has_bias != (s.b_hh != null)) return BackendError.InvalidArgument;

    if (has_bias) {
        const bih_id: tensor_store.TensorId = s.b_ih.?;
        const bhh_id: tensor_store.TensorId = s.b_hh.?;
        const bih_meta = try store.meta(bih_id);
        const bhh_meta = try store.meta(bhh_id);
        if (bih_meta.dtype != .f32 or bhh_meta.dtype != .f32) return BackendError.InvalidArgument;
        if (bih_meta.rank != 1 or bhh_meta.rank != 1) return BackendError.InvalidArgument;
        if (bih_meta.shape[0] != gate_dim or bhh_meta.shape[0] != gate_dim) return BackendError.InvalidArgument;

        try loadVecF32(store, bih_meta, bih_id, b_ih_buf[0..gate_dim]);
        try loadVecF32(store, bhh_meta, bhh_id, b_hh_buf[0..gate_dim]);
    }

    var x_row_buf: [MAX_INPUT]f32 = undefined;
    var h_row_buf: [MAX_HIDDEN]f32 = undefined;
    var c_row_buf: [MAX_HIDDEN]f32 = undefined;
    var gates_buf: [MAX_HIDDEN * 4]f32 = undefined;
    var h_out_buf: [MAX_HIDDEN]f32 = undefined;
    var c_out_buf: [MAX_HIDDEN]f32 = undefined;

    var b: usize = 0;
    while (b < batch) : (b += 1) {
        try loadRowF32(store, x_meta, s.x, b, x_row_buf[0..input_size]);
        try loadRowF32(store, h_meta, s.h_prev, b, h_row_buf[0..hidden]);
        try loadRowF32(store, c_meta, s.c_prev, b, c_row_buf[0..hidden]);

        // gates = bias (optional)
        var j: usize = 0;
        while (j < gate_dim) : (j += 1) {
            gates_buf[j] = if (has_bias) (b_ih_buf[j] + b_hh_buf[j]) else 0.0;
        }

        // gates += x @ w_ih
        try accumMatVecTiled(store, wih_meta, s.w_ih, x_row_buf[0..input_size], gates_buf[0..gate_dim]);
        // gates += h_prev @ w_hh
        try accumMatVecTiled(store, whh_meta, s.w_hh, h_row_buf[0..hidden], gates_buf[0..gate_dim]);

        // Compute new state.
        const h_off0: usize = 0;
        const h_off1: usize = hidden;
        const h_off2: usize = hidden * 2;
        const h_off3: usize = hidden * 3;

        j = 0;
        while (j < hidden) : (j += 1) {
            const i_lin: f32 = gates_buf[h_off0 + j];
            const f_lin: f32 = gates_buf[h_off1 + j];
            const g_lin: f32 = gates_buf[h_off2 + j];
            const o_lin: f32 = gates_buf[h_off3 + j];

            const i_gate: f32 = fast_math.sigmoidApproxF32(i_lin);
            const f_gate: f32 = fast_math.sigmoidApproxF32(f_lin);
            const g_gate: f32 = fast_math.tanhApproxF32(g_lin);
            const o_gate: f32 = fast_math.sigmoidApproxF32(o_lin);

            const c_t: f32 = f_gate * c_row_buf[j] + i_gate * g_gate;
            const h_t: f32 = o_gate * fast_math.tanhApproxF32(c_t);

            c_out_buf[j] = c_t;
            h_out_buf[j] = h_t;
        }

        try storeRowStateF32(store, out_meta, s.out_state, b, h_out_buf[0..hidden], c_out_buf[0..hidden]);
    }
}
