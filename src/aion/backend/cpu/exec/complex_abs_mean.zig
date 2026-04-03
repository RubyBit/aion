const std = @import("std");

const backend_mod = @import("../../backend.zig");
const types = @import("../../types.zig");
const thread_pool = @import("../../../runtime/thread_pool.zig");
const tensor_store = @import("../../../runtime/tensor_store.zig");
const executable = @import("../../../runtime/executable.zig");

const BackendError = types.BackendError;
const ExecuteProgramError = backend_mod.ExecuteProgramError;

const MAX_RANK: usize = 8;

const TileCacheConst = struct {
    valid: bool = false,
    tile_index: usize = 0,
    tile: tensor_store.TileRefConst = undefined,

    fn deinit(self: *TileCacheConst, store: tensor_store.TensorStore) void {
        if (self.valid) {
            store.releaseConst(self.tile.token);
            self.valid = false;
        }
    }
};

const TileCacheMut = struct {
    valid: bool = false,
    tile_index: usize = 0,
    tile: tensor_store.TileRefMut = undefined,

    fn deinit(self: *TileCacheMut, store: tensor_store.TensorStore) void {
        if (self.valid) {
            store.releaseMut(self.tile.token);
            self.valid = false;
        }
    }
};

fn ensureConstTile(cache: *TileCacheConst, store: tensor_store.TensorStore, id: tensor_store.TensorId, tile_index: usize) ExecuteProgramError!void {
    if (cache.valid and cache.tile_index == tile_index) return;
    if (cache.valid) {
        store.releaseConst(cache.tile.token);
        cache.valid = false;
    }
    cache.tile = try store.acquireTileConstLinear(id, tile_index);
    cache.tile_index = tile_index;
    cache.valid = true;
}

fn ensureMutTile(cache: *TileCacheMut, store: tensor_store.TensorStore, id: tensor_store.TensorId, tile_index: usize) ExecuteProgramError!void {
    if (cache.valid and cache.tile_index == tile_index) return;
    if (cache.valid) {
        store.releaseMut(cache.tile.token);
        cache.valid = false;
    }
    cache.tile = try store.acquireTileMutLinear(id, tile_index);
    cache.tile_index = tile_index;
    cache.valid = true;
}

fn readScalarF32At(
    store: tensor_store.TensorStore,
    meta: tensor_store.TensorMeta,
    id: tensor_store.TensorId,
    coords: []const usize,
    cache: *TileCacheConst,
) ExecuteProgramError!f32 {
    if (coords.len != @as(usize, meta.rank)) return BackendError.InvalidArgument;
    if (meta.rank == 0 or meta.rank > MAX_RANK) return BackendError.InvalidArgument;

    var tile_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
    var local_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;

    const rank: usize = @as(usize, meta.rank);
    var d: usize = 0;
    while (d < rank) : (d += 1) {
        if (coords[d] >= meta.shape[d]) return BackendError.InvalidArgument;
        tile_coords[d] = coords[d] / meta.tile_shape[d];
        local_coords[d] = coords[d] - tile_coords[d] * meta.tile_shape[d];
    }

    const tile_index: usize = try tensor_store.encodeTileIndex(meta, tile_coords[0..rank]);
    try ensureConstTile(cache, store, id, tile_index);

    const v = cache.tile.bufferView();
    if (v.dtype != meta.dtype) return BackendError.InvalidArgument;
    if (@as(usize, v.layout.rank) != rank) return BackendError.InvalidArgument;

    var off_bytes: usize = 0;
    d = 0;
    while (d < rank) : (d += 1) {
        const stride_i: isize = v.layout.strides_bytes[d];
        if (stride_i <= 0) return BackendError.InvalidArgument;
        const stride: usize = @intCast(stride_i);
        off_bytes = std.math.add(usize, off_bytes, local_coords[d] * stride) catch return BackendError.InvalidArgument;
    }

    switch (meta.dtype) {
        .f32 => {
            if (off_bytes + 4 > v.bytes.len) return BackendError.InvalidArgument;
            return @as(*align(1) const f32, @ptrCast(v.bytes.ptr + off_bytes)).*;
        },
        .f16 => {
            if (off_bytes + 2 > v.bytes.len) return BackendError.InvalidArgument;
            const x: f16 = @as(*align(1) const f16, @ptrCast(v.bytes.ptr + off_bytes)).*;
            return @floatCast(x);
        },
        else => return BackendError.InvalidArgument,
    }
}

fn writeScalarFromF32At(
    store: tensor_store.TensorStore,
    meta: tensor_store.TensorMeta,
    id: tensor_store.TensorId,
    coords: []const usize,
    v_in: f32,
    cache: *TileCacheMut,
) ExecuteProgramError!void {
    if (coords.len != @as(usize, meta.rank)) return BackendError.InvalidArgument;
    if (meta.rank == 0 or meta.rank > MAX_RANK) return BackendError.InvalidArgument;

    var tile_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;
    var local_coords: [MAX_RANK]usize = .{0} ** MAX_RANK;

    const rank: usize = @as(usize, meta.rank);
    var d: usize = 0;
    while (d < rank) : (d += 1) {
        if (coords[d] >= meta.shape[d]) return BackendError.InvalidArgument;
        tile_coords[d] = coords[d] / meta.tile_shape[d];
        local_coords[d] = coords[d] - tile_coords[d] * meta.tile_shape[d];
    }

    const tile_index: usize = try tensor_store.encodeTileIndex(meta, tile_coords[0..rank]);
    try ensureMutTile(cache, store, id, tile_index);

    const v = cache.tile.bufferView();
    if (v.dtype != meta.dtype) return BackendError.InvalidArgument;
    if (@as(usize, v.layout.rank) != rank) return BackendError.InvalidArgument;

    var off_bytes: usize = 0;
    d = 0;
    while (d < rank) : (d += 1) {
        const stride_i: isize = v.layout.strides_bytes[d];
        if (stride_i <= 0) return BackendError.InvalidArgument;
        const stride: usize = @intCast(stride_i);
        off_bytes = std.math.add(usize, off_bytes, local_coords[d] * stride) catch return BackendError.InvalidArgument;
    }

    switch (meta.dtype) {
        .f32 => {
            if (off_bytes + 4 > v.bytes.len) return BackendError.InvalidArgument;
            @as(*align(1) f32, @ptrCast(v.bytes.ptr + off_bytes)).* = v_in;
        },
        .f16 => {
            if (off_bytes + 2 > v.bytes.len) return BackendError.InvalidArgument;
            @as(*align(1) f16, @ptrCast(v.bytes.ptr + off_bytes)).* = @floatCast(v_in);
        },
        else => return BackendError.InvalidArgument,
    }
}

pub fn execComplexAbsMean(
    pool: ?*thread_pool.ThreadPool,
    thread_count: usize,
    s: executable.StepComplexAbsMean,
    store: tensor_store.TensorStore,
) ExecuteProgramError!void {
    _ = pool;
    _ = thread_count;

    const out_meta = try store.meta(s.out);
    const stft_meta = try store.meta(s.stft);

    if (out_meta.dtype != stft_meta.dtype) return BackendError.InvalidArgument;
    if (stft_meta.rank != 3 or out_meta.rank != 2) return BackendError.InvalidArgument;

    const batch: usize = stft_meta.shape[0];
    const time: usize = stft_meta.shape[1];
    const chans2: usize = stft_meta.shape[2];
    if (batch == 0 or time == 0 or chans2 == 0) return BackendError.InvalidArgument;
    if (chans2 % 2 != 0) return BackendError.InvalidArgument;

    const cutoff: usize = chans2 / 2;
    if (s.out_channels == 0 or s.out_channels > cutoff) return BackendError.InvalidArgument;

    if (out_meta.shape[0] != batch) return BackendError.InvalidArgument;
    if (out_meta.shape[1] != s.out_channels) return BackendError.InvalidArgument;

    // v0: scalar dtypes only.
    if (out_meta.dtype != .f32 and out_meta.dtype != .f16) return BackendError.Unsupported;

    var real_cache: TileCacheConst = .{};
    defer real_cache.deinit(store);
    var imag_cache: TileCacheConst = .{};
    defer imag_cache.deinit(store);
    var out_cache: TileCacheMut = .{};
    defer out_cache.deinit(store);

    const inv_time: f32 = 1.0 / @as(f32, @floatFromInt(time));

    var b: usize = 0;
    while (b < batch) : (b += 1) {
        var c: usize = 0;
        while (c < s.out_channels) : (c += 1) {
            var acc: f32 = 0.0;

            var t: usize = 0;
            while (t < time) : (t += 1) {
                const re: f32 = try readScalarF32At(store, stft_meta, s.stft, &[_]usize{ b, t, c }, &real_cache);
                const im: f32 = try readScalarF32At(store, stft_meta, s.stft, &[_]usize{ b, t, c + cutoff }, &imag_cache);
                const mag2: f32 = re * re + im * im;
                acc += @sqrt(mag2);
            }

            const out_v: f32 = acc * inv_time;
            try writeScalarFromF32At(store, out_meta, s.out, &[_]usize{ b, c }, out_v, &out_cache);
        }
    }
}
