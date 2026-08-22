// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const api = @import("api/api.zig");
const types = @import("backend/types.zig");
const pkg_types = @import("storage/aion_file/types.zig");

pub const AionStatus = enum(c_int) {
    AION_OK = 0,
    AION_INVALID_ARGUMENT = 1,
    AION_OUT_OF_MEMORY = 2,
    AION_UNSUPPORTED = 3,
    AION_INTERNAL_ERROR = 4,
};

pub const AionDType = enum(c_int) {
    AION_DTYPE_F32 = 0,
    AION_DTYPE_F16 = 1,
    AION_DTYPE_I8 = 2,
    AION_DTYPE_Q4_0 = 3,
    AION_DTYPE_Q8_0 = 4,
    AION_DTYPE_I32 = 5,
};

pub const AionDeviceKind = enum(c_int) {
    AION_DEVICE_CPU = 0,
    AION_DEVICE_GPU = 1,
};

pub const AionGpuPower = enum(c_int) {
    AION_GPU_POWER_DEFAULT = 0,
    AION_GPU_POWER_LOW = 1,
    AION_GPU_POWER_HIGH = 2,
};

pub const AionGpuBackend = enum(c_int) {
    AION_GPU_BACKEND_ANY = 0,
    AION_GPU_BACKEND_VULKAN = 1,
    AION_GPU_BACKEND_D3D12 = 2,
    AION_GPU_BACKEND_METAL = 3,
    AION_GPU_BACKEND_GL = 4,
};

/// Per-GPU creation options mirrored from `api.GpuOptions`.
/// `adapter_index < 0` means "auto" (no explicit adapter).
pub const AionGpuOptions = extern struct {
    power: AionGpuPower = .AION_GPU_POWER_DEFAULT,
    backend: AionGpuBackend = .AION_GPU_BACKEND_ANY,
    adapter_index: i32 = -1,
};

pub const AionContext = struct {
    allocator: std.mem.Allocator = std.heap.page_allocator,
    ctx: api.Context = undefined,

    last_err_buf: [512]u8 = @splat(0),
    last_err_len: usize = 0,

    fn init(thread_count: usize, gpus: []const api.GpuOptions) !AionContext {
        var out: AionContext = .{};
        out.ctx = try api.Context.init(out.allocator, .{ .thread_count = thread_count, .gpus = gpus });
        out.clearLastError();
        return out;
    }

    fn deinit(self: *AionContext) void {
        self.ctx.deinit();
        self.* = undefined;
    }

    fn clearLastError(self: *AionContext) void {
        self.last_err_len = 0;
        self.last_err_buf[0] = 0;
    }

    fn setLastError(self: *AionContext, comptime prefix: []const u8, err: anyerror) void {
        // Best-effort: format "<prefix>: <error>" into a fixed buffer.
        const msg = std.fmt.bufPrint(self.last_err_buf[0..], "{s}: {s}", .{ prefix, @errorName(err) }) catch blk: {
            // Fallback: just prefix.
            const m2 = std.fmt.bufPrint(self.last_err_buf[0..], "{s}", .{prefix}) catch {
                self.last_err_buf[0] = 0;
                self.last_err_len = 0;
                return;
            };
            break :blk m2;
        };
        self.last_err_len = msg.len;
        if (self.last_err_len < self.last_err_buf.len) self.last_err_buf[self.last_err_len] = 0;
    }
};

pub const AionTensor = struct {
    owner: *AionContext,
    tensor: api.Tensor,
};

pub const AionLoadedModel = struct {
    owner: *AionContext,
    model: api.LoadedModel,
};

fn mapError(err: anyerror) AionStatus {
    return switch (err) {
        error.InvalidArgument => .AION_INVALID_ARGUMENT,
        error.OutOfMemory => .AION_OUT_OF_MEMORY,
        error.Unsupported => .AION_UNSUPPORTED,
        error.UnsupportedVersion => .AION_UNSUPPORTED,
        error.UnsupportedFeature => .AION_UNSUPPORTED,
        error.BackendUnavailable => .AION_UNSUPPORTED,
        error.InvalidFormat => .AION_INVALID_ARGUMENT,
        error.IoFailure => .AION_INTERNAL_ERROR,
        else => .AION_INTERNAL_ERROR,
    };
}

fn dtypeToC(dt: types.DType) AionDType {
    return switch (dt) {
        .f32 => .AION_DTYPE_F32,
        .f16 => .AION_DTYPE_F16,
        .i8 => .AION_DTYPE_I8,
        .q4_0 => .AION_DTYPE_Q4_0,
        .q8_0 => .AION_DTYPE_Q8_0,
        .i32 => .AION_DTYPE_I32,
    };
}

fn dtypeFromC(dt: AionDType) ?types.DType {
    return switch (dt) {
        .AION_DTYPE_F32 => .f32,
        .AION_DTYPE_F16 => .f16,
        .AION_DTYPE_I8 => .i8,
        .AION_DTYPE_Q4_0 => .q4_0,
        .AION_DTYPE_Q8_0 => .q8_0,
        .AION_DTYPE_I32 => .i32,
    };
}

fn deviceFromC(kind: AionDeviceKind, index: u32) api.DeviceSelector {
    return switch (kind) {
        .AION_DEVICE_CPU => .cpu,
        .AION_DEVICE_GPU => .{ .gpu = @as(usize, index) },
    };
}

fn gpuOptionsFromC(c: AionGpuOptions) api.GpuOptions {
    return .{
        .power = switch (c.power) {
            .AION_GPU_POWER_DEFAULT => .default,
            .AION_GPU_POWER_LOW => .low,
            .AION_GPU_POWER_HIGH => .high,
        },
        .backend = switch (c.backend) {
            .AION_GPU_BACKEND_ANY => .any,
            .AION_GPU_BACKEND_VULKAN => .vulkan,
            .AION_GPU_BACKEND_D3D12 => .d3d12,
            .AION_GPU_BACKEND_METAL => .metal,
            .AION_GPU_BACKEND_GL => .gl,
        },
        .adapter_index = if (c.adapter_index < 0) null else @as(usize, @intCast(c.adapter_index)),
    };
}

fn requireAligned(ptr: *const anyopaque, alignment: usize) bool {
    if (alignment <= 1) return true;
    const raw_addr: usize = @intFromPtr(ptr);
    return (raw_addr & (alignment - 1)) == 0;
}

fn copyStringToBuf(s: []const u8, buf: [*c]u8, cap: usize, out_len: ?*usize) void {
    if (out_len) |p| p.* = s.len;
    if (buf == null or cap == 0) return;

    const dst: [*c]u8 = buf;
    const n: usize = if (cap <= 1) 0 else @min(s.len, cap - 1);
    if (n != 0) {
        @memcpy(dst[0..n], s[0..n]);
    }
    dst[n] = 0;
}

fn getShapeSlice(rank: usize, shape_ptr: [*c]const usize) ?[]const usize {
    // Note: The current high-level Aion API does not allow rank-0 tensors.
    // Scalars are represented as shape {1}.
    if (shape_ptr == null) return null;
    return shape_ptr[0..rank];
}

pub export fn aion_version_major() callconv(.c) u32 {
    return 0;
}

pub export fn aion_version_minor() callconv(.c) u32 {
    return 0;
}

pub export fn aion_version_patch() callconv(.c) u32 {
    return 1;
}

pub export fn aion_status_string(status: AionStatus) callconv(.c) [*c]const u8 {
    return switch (status) {
        .AION_OK => "AION_OK",
        .AION_INVALID_ARGUMENT => "AION_INVALID_ARGUMENT",
        .AION_OUT_OF_MEMORY => "AION_OUT_OF_MEMORY",
        .AION_UNSUPPORTED => "AION_UNSUPPORTED",
        .AION_INTERNAL_ERROR => "AION_INTERNAL_ERROR",
    };
}

pub export fn aion_context_last_error_message(
    ctx_opt: ?*const AionContext,
    buf: [*c]u8,
    cap: usize,
    out_len: ?*usize,
) callconv(.c) AionStatus {
    const ctx: *const AionContext = ctx_opt orelse return .AION_INVALID_ARGUMENT;
    copyStringToBuf(ctx.last_err_buf[0..ctx.last_err_len], buf, cap, out_len);
    return .AION_OK;
}

pub export fn aion_context_create_cpu(thread_count: usize, out_ctx: ?*?*AionContext) callconv(.c) AionStatus {
    if (out_ctx == null) return .AION_INVALID_ARGUMENT;
    out_ctx.?.* = null;
    if (thread_count == 0) return .AION_INVALID_ARGUMENT;

    const ctx_ptr: *AionContext = std.heap.page_allocator.create(AionContext) catch return .AION_OUT_OF_MEMORY;
    errdefer std.heap.page_allocator.destroy(ctx_ptr);

    ctx_ptr.* = AionContext.init(thread_count, &.{}) catch |e| {
        const st = mapError(e);
        // No context to store the error yet.
        return st;
    };

    out_ctx.?.* = ctx_ptr;
    return .AION_OK;
}

/// Maximum number of GPUs accepted by `aion_context_create` in one call.
const MAX_GPUS: usize = 16;

pub export fn aion_context_create(
    thread_count: usize,
    gpus_ptr: [*c]const AionGpuOptions,
    gpu_count: usize,
    out_ctx: ?*?*AionContext,
) callconv(.c) AionStatus {
    if (out_ctx == null) return .AION_INVALID_ARGUMENT;
    out_ctx.?.* = null;
    if (thread_count == 0) return .AION_INVALID_ARGUMENT;
    if (gpu_count > MAX_GPUS) return .AION_INVALID_ARGUMENT;
    if (gpu_count != 0 and gpus_ptr == null) return .AION_INVALID_ARGUMENT;

    var gpu_buf: [MAX_GPUS]api.GpuOptions = undefined;
    var i: usize = 0;
    while (i < gpu_count) : (i += 1) {
        gpu_buf[i] = gpuOptionsFromC(gpus_ptr[i]);
    }
    const gpus: []const api.GpuOptions = gpu_buf[0..gpu_count];

    const ctx_ptr: *AionContext = std.heap.page_allocator.create(AionContext) catch return .AION_OUT_OF_MEMORY;
    errdefer std.heap.page_allocator.destroy(ctx_ptr);

    ctx_ptr.* = AionContext.init(thread_count, gpus) catch |e| {
        // No context to store the error yet.
        return mapError(e);
    };

    out_ctx.?.* = ctx_ptr;
    return .AION_OK;
}

pub export fn aion_context_destroy(ctx_opt: ?*AionContext) callconv(.c) void {
    const ctx: *AionContext = ctx_opt orelse return;
    ctx.deinit();
    std.heap.page_allocator.destroy(ctx);
}

pub export fn aion_tensor_create_empty(
    ctx_opt: ?*AionContext,
    dtype_c: AionDType,
    rank: usize,
    shape_ptr: [*c]const usize,
    out_tensor: ?*?*AionTensor,
) callconv(.c) AionStatus {
    const ctx: *AionContext = ctx_opt orelse return .AION_INVALID_ARGUMENT;
    ctx.clearLastError();

    if (out_tensor == null) return .AION_INVALID_ARGUMENT;
    out_tensor.?.* = null;

    const shape: []const usize = getShapeSlice(rank, shape_ptr) orelse return .AION_INVALID_ARGUMENT;
    const dt: types.DType = dtypeFromC(dtype_c) orelse return .AION_INVALID_ARGUMENT;

    const t: api.Tensor = ctx.ctx.tensor(dt, shape) catch |e| {
        ctx.setLastError("tensor_create_empty", e);
        return mapError(e);
    };

    const handle: *AionTensor = std.heap.page_allocator.create(AionTensor) catch {
        ctx.setLastError("tensor_handle_alloc", error.OutOfMemory);
        return .AION_OUT_OF_MEMORY;
    };
    handle.* = .{ .owner = ctx, .tensor = t };
    out_tensor.?.* = handle;
    return .AION_OK;
}

pub export fn aion_tensor_create_empty_tiled(
    ctx_opt: ?*AionContext,
    dtype_c: AionDType,
    rank: usize,
    shape_ptr: [*c]const usize,
    tile_shape_ptr: [*c]const usize,
    out_tensor: ?*?*AionTensor,
) callconv(.c) AionStatus {
    const ctx: *AionContext = ctx_opt orelse return .AION_INVALID_ARGUMENT;
    ctx.clearLastError();

    if (out_tensor == null) return .AION_INVALID_ARGUMENT;
    out_tensor.?.* = null;

    const shape: []const usize = getShapeSlice(rank, shape_ptr) orelse return .AION_INVALID_ARGUMENT;
    const tile_shape: []const usize = getShapeSlice(rank, tile_shape_ptr) orelse return .AION_INVALID_ARGUMENT;
    const dt: types.DType = dtypeFromC(dtype_c) orelse return .AION_INVALID_ARGUMENT;

    const t: api.Tensor = ctx.ctx.tensorTiled(dt, shape, tile_shape) catch |e| {
        ctx.setLastError("tensor_create_empty_tiled", e);
        return mapError(e);
    };

    const handle: *AionTensor = std.heap.page_allocator.create(AionTensor) catch {
        ctx.setLastError("tensor_handle_alloc", error.OutOfMemory);
        return .AION_OUT_OF_MEMORY;
    };
    handle.* = .{ .owner = ctx, .tensor = t };
    out_tensor.?.* = handle;
    return .AION_OK;
}

pub export fn aion_tensor_create(
    ctx_opt: ?*AionContext,
    dtype_c: AionDType,
    rank: usize,
    shape_ptr: [*c]const usize,
    values_ptr: ?*const anyopaque,
    values_len: usize,
    out_tensor: ?*?*AionTensor,
) callconv(.c) AionStatus {
    const ctx: *AionContext = ctx_opt orelse return .AION_INVALID_ARGUMENT;
    ctx.clearLastError();

    if (out_tensor == null) return .AION_INVALID_ARGUMENT;
    out_tensor.?.* = null;

    const shape: []const usize = getShapeSlice(rank, shape_ptr) orelse return .AION_INVALID_ARGUMENT;
    const dt: types.DType = dtypeFromC(dtype_c) orelse return .AION_INVALID_ARGUMENT;

    // If no values are provided, behave like create_empty (but keep this API convenient for FFI).
    const t: api.Tensor = blk: {
        if (values_ptr == null) {
            if (values_len != 0) return .AION_INVALID_ARGUMENT;
            break :blk ctx.ctx.tensor(dt, shape) catch |e| {
                ctx.setLastError("tensor_create", e);
                return mapError(e);
            };
        }
        if (values_len == 0) {
            break :blk ctx.ctx.tensor(dt, shape) catch |e| {
                ctx.setLastError("tensor_create", e);
                return mapError(e);
            };
        }

        if (dt.info().is_quantized) return .AION_UNSUPPORTED;

        const elem_bytes: usize = dt.info().block_bytes;
        if (!requireAligned(values_ptr.?, elem_bytes)) return .AION_INVALID_ARGUMENT;

        const total_bytes: usize = std.math.mul(usize, values_len, elem_bytes) catch return .AION_INVALID_ARGUMENT;
        const bytes_ptr: [*]const u8 = @ptrCast(values_ptr.?);
        const @"packed": []const u8 = bytes_ptr[0..total_bytes];

        break :blk ctx.ctx.fromPackedScalar(dt, shape, @"packed") catch |e| {
            ctx.setLastError("tensor_create", e);
            return mapError(e);
        };
    };

    const handle: *AionTensor = std.heap.page_allocator.create(AionTensor) catch {
        ctx.setLastError("tensor_handle_alloc", error.OutOfMemory);
        return .AION_OUT_OF_MEMORY;
    };
    handle.* = .{ .owner = ctx, .tensor = t };
    out_tensor.?.* = handle;
    return .AION_OK;
}

pub export fn aion_tensor_destroy(t_opt: ?*AionTensor) callconv(.c) void {
    const t: *AionTensor = t_opt orelse return;
    // Underlying tensor storage is owned by the context; this only frees the handle.
    std.heap.page_allocator.destroy(t);
}

/// Migrate a tensor to `(kind, index)`.
///
/// Move semantics: the source-device copy is freed. After moving off the CPU,
/// host `aion_tensor_read`/`aion_tensor_write` fail until migrated back to the
/// CPU with `aion_tensor_to(t, AION_DEVICE_CPU, 0)`.
pub export fn aion_tensor_to(
    t_opt: ?*AionTensor,
    kind: AionDeviceKind,
    index: u32,
) callconv(.c) AionStatus {
    const t: *AionTensor = t_opt orelse return .AION_INVALID_ARGUMENT;
    const ctx: *AionContext = t.owner;
    ctx.clearLastError();

    const sel = deviceFromC(kind, index);
    t.tensor.to(sel) catch |e| {
        ctx.setLastError("tensor_to", e);
        return mapError(e);
    };
    return .AION_OK;
}

/// Report the device a tensor is currently resident on. Either output pointer
/// may be NULL if that field is not needed.
pub export fn aion_tensor_device(
    t_opt: ?*const AionTensor,
    out_kind: ?*AionDeviceKind,
    out_index: ?*u32,
) callconv(.c) AionStatus {
    const t: *const AionTensor = t_opt orelse return .AION_INVALID_ARGUMENT;
    const ctx: *AionContext = t.owner;
    ctx.clearLastError();

    const ref = t.tensor.device() catch |e| {
        ctx.setLastError("tensor_device", e);
        return mapError(e);
    };
    if (out_kind) |p| p.* = switch (ref.kind) {
        .cpu => .AION_DEVICE_CPU,
        .gpu => .AION_DEVICE_GPU,
    };
    if (out_index) |p| p.* = @as(u32, ref.index);
    return .AION_OK;
}

pub export fn aion_tensor_dtype(t_opt: ?*const AionTensor) callconv(.c) AionDType {
    const t: *const AionTensor = t_opt orelse return .AION_DTYPE_F32;
    return dtypeToC(t.tensor.getDType());
}

pub export fn aion_tensor_rank(t_opt: ?*const AionTensor) callconv(.c) usize {
    const t: *const AionTensor = t_opt orelse return 0;
    return t.tensor.getShape().len;
}

pub export fn aion_tensor_shape(t_opt: ?*const AionTensor, out_dims: [*c]usize, out_rank: usize) callconv(.c) AionStatus {
    const t: *const AionTensor = t_opt orelse return .AION_INVALID_ARGUMENT;
    const shape: []const usize = t.tensor.getShape();
    if (out_rank != shape.len) return .AION_INVALID_ARGUMENT;
    if (out_dims == null) return .AION_INVALID_ARGUMENT;
    @memcpy(out_dims[0..out_rank], shape);
    return .AION_OK;
}

pub export fn aion_tensor_read(
    t_opt: ?*const AionTensor,
    dtype_c: AionDType,
    out_values: ?*anyopaque,
    out_len: usize,
) callconv(.c) AionStatus {
    const t: *const AionTensor = t_opt orelse return .AION_INVALID_ARGUMENT;
    const ctx: *AionContext = t.owner;
    ctx.clearLastError();

    if (out_values == null) return .AION_INVALID_ARGUMENT;
    const dt: types.DType = dtypeFromC(dtype_c) orelse return .AION_INVALID_ARGUMENT;
    if (t.tensor.getDType() != dt) return .AION_INVALID_ARGUMENT;

    const want: usize = t.tensor.elemCount() catch |e| {
        ctx.setLastError("tensor_elem_count", e);
        return mapError(e);
    };
    if (out_len != want) return .AION_INVALID_ARGUMENT;

    if (dt.info().is_quantized) return .AION_UNSUPPORTED;
    const elem_bytes: usize = dt.info().block_bytes;
    if (!requireAligned(out_values.?, elem_bytes)) return .AION_INVALID_ARGUMENT;

    const total_bytes: usize = std.math.mul(usize, out_len, elem_bytes) catch return .AION_INVALID_ARGUMENT;
    const bytes_ptr: [*]u8 = @ptrCast(out_values.?);
    const out_bytes: []u8 = bytes_ptr[0..total_bytes];

    t.tensor.readPackedScalar(out_bytes) catch |e| {
        ctx.setLastError("tensor_read", e);
        return mapError(e);
    };
    return .AION_OK;
}

pub export fn aion_tensor_write(
    t_opt: ?*AionTensor,
    dtype_c: AionDType,
    values_ptr: ?*const anyopaque,
    values_len: usize,
) callconv(.c) AionStatus {
    const t: *AionTensor = t_opt orelse return .AION_INVALID_ARGUMENT;
    const ctx: *AionContext = t.owner;
    ctx.clearLastError();

    if (values_ptr == null) return .AION_INVALID_ARGUMENT;
    const dt: types.DType = dtypeFromC(dtype_c) orelse return .AION_INVALID_ARGUMENT;
    if (t.tensor.getDType() != dt) return .AION_INVALID_ARGUMENT;

    const want: usize = t.tensor.elemCount() catch |e| {
        ctx.setLastError("tensor_elem_count", e);
        return mapError(e);
    };
    if (values_len != want) return .AION_INVALID_ARGUMENT;

    if (dt.info().is_quantized) return .AION_UNSUPPORTED;
    const elem_bytes: usize = dt.info().block_bytes;
    if (!requireAligned(values_ptr.?, elem_bytes)) return .AION_INVALID_ARGUMENT;

    const total_bytes: usize = std.math.mul(usize, values_len, elem_bytes) catch return .AION_INVALID_ARGUMENT;
    const bytes_ptr: [*]const u8 = @ptrCast(values_ptr.?);
    const @"packed": []const u8 = bytes_ptr[0..total_bytes];

    t.tensor.writePackedScalar(@"packed") catch |e| {
        ctx.setLastError("tensor_write", e);
        return mapError(e);
    };
    return .AION_OK;
}

pub export fn aion_tensor_read_scalar(
    t_opt: ?*const AionTensor,
    dtype_c: AionDType,
    out_value: ?*anyopaque,
) callconv(.c) AionStatus {
    const t: *const AionTensor = t_opt orelse return .AION_INVALID_ARGUMENT;
    const ctx: *AionContext = t.owner;
    ctx.clearLastError();

    if (out_value == null) return .AION_INVALID_ARGUMENT;
    const dt: types.DType = dtypeFromC(dtype_c) orelse return .AION_INVALID_ARGUMENT;
    if (t.tensor.getDType() != dt) return .AION_INVALID_ARGUMENT;

    const want: usize = t.tensor.elemCount() catch |e| {
        ctx.setLastError("tensor_elem_count", e);
        return mapError(e);
    };
    if (want != 1) return .AION_INVALID_ARGUMENT;

    if (dt.info().is_quantized) return .AION_UNSUPPORTED;
    const elem_bytes: usize = dt.info().block_bytes;
    if (!requireAligned(out_value.?, elem_bytes)) return .AION_INVALID_ARGUMENT;

    var tmp: [8]u8 = undefined;
    if (elem_bytes > tmp.len) return .AION_UNSUPPORTED;
    const tmp_slice: []u8 = tmp[0..elem_bytes];

    t.tensor.readPackedScalar(tmp_slice) catch |e| {
        ctx.setLastError("tensor_read_scalar", e);
        return mapError(e);
    };
    const out_bytes_ptr: [*]u8 = @ptrCast(out_value.?);
    @memcpy(out_bytes_ptr[0..elem_bytes], tmp_slice);
    return .AION_OK;
}

fn loadModelImpl(
    ctx_opt: ?*AionContext,
    path: ?[*:0]const u8,
    absolute: bool,
    opts: api.LoadModelOptions,
    out_model: ?*?*AionLoadedModel,
    comptime what: []const u8,
) AionStatus {
    const ctx: *AionContext = ctx_opt orelse return .AION_INVALID_ARGUMENT;
    ctx.clearLastError();

    if (out_model == null) return .AION_INVALID_ARGUMENT;
    out_model.?.* = null;
    if (path == null) return .AION_INVALID_ARGUMENT;

    const p: []const u8 = std.mem.span(path.?);
    const lm: api.LoadedModel = (if (absolute)
        ctx.ctx.loadModelPathAbsolute(p, opts)
    else
        ctx.ctx.loadModelPath(p, opts)) catch |e| {
        ctx.setLastError(what, e);
        return mapError(e);
    };

    const handle: *AionLoadedModel = std.heap.page_allocator.create(AionLoadedModel) catch {
        ctx.setLastError("loaded_model_handle_alloc", error.OutOfMemory);
        return .AION_OUT_OF_MEMORY;
    };
    handle.* = .{ .owner = ctx, .model = lm };
    out_model.?.* = handle;
    return .AION_OK;
}

/// Load options. Zero-init (or a NULL pointer) means defaults: CPU, auto-init
/// inputs, auto positions, no cache auto-sizing. `cache_capacity_tokens > 0`
/// sizes every role-declared sequence cache with a free capacity symbol;
/// `cache_growable != 0` additionally starts those caches small
/// (`cache_initial_capacity_tokens`, 0 = default 8) and grows them on demand
/// (growth factor `cache_growth_numerator/denominator`, 0/0 = default 3/2).
pub const AionLoadModelOptions = extern struct {
    device_kind: u32 = 0, // AionDeviceKind
    device_index: u32 = 0,
    auto_init_inputs: u8 = 1,
    auto_positions: u8 = 1,
    _pad: [6]u8 = @splat(0),
    cache_capacity_tokens: u64 = 0,
    cache_growable: u8 = 0,
    _pad1: [7]u8 = @splat(0),
    cache_initial_capacity_tokens: u64 = 0,
    cache_growth_numerator: u64 = 0,
    cache_growth_denominator: u64 = 0,
    plan_cache_budget_bytes: u64 = 0,
};

fn loadOptionsFromC(c_opts: ?*const AionLoadModelOptions) ?api.LoadModelOptions {
    const c = c_opts orelse return api.LoadModelOptions{};
    const kind = std.enums.fromInt(AionDeviceKind, c.device_kind) orelse return null;
    var opts: api.LoadModelOptions = .{
        .device = deviceFromC(kind, c.device_index),
        .auto_init_inputs = c.auto_init_inputs != 0,
        .auto_positions = c.auto_positions != 0,
    };
    if (c.cache_capacity_tokens > 0) {
        opts.cache.capacity_tokens = @intCast(c.cache_capacity_tokens);
        if (c.cache_growable != 0) {
            var growth: api.CacheGrowth = .{};
            if (c.cache_initial_capacity_tokens > 0) growth.initial_capacity_tokens = @intCast(c.cache_initial_capacity_tokens);
            if (c.cache_growth_numerator > 0 and c.cache_growth_denominator > 0) {
                growth.growth_numerator = @intCast(c.cache_growth_numerator);
                growth.growth_denominator = @intCast(c.cache_growth_denominator);
            }
            opts.cache.growable = growth;
        }
    }
    if (c.plan_cache_budget_bytes > 0) opts.plan_cache_budget_bytes = @intCast(c.plan_cache_budget_bytes);
    return opts;
}

pub export fn aion_loaded_model_load_path(
    ctx_opt: ?*AionContext,
    path: ?[*:0]const u8,
    opts_opt: ?*const AionLoadModelOptions,
    out_model: ?*?*AionLoadedModel,
) callconv(.c) AionStatus {
    const opts = loadOptionsFromC(opts_opt) orelse return .AION_INVALID_ARGUMENT;
    return loadModelImpl(ctx_opt, path, false, opts, out_model, "load_model_path");
}

pub export fn aion_loaded_model_load_path_absolute(
    ctx_opt: ?*AionContext,
    path: ?[*:0]const u8,
    opts_opt: ?*const AionLoadModelOptions,
    out_model: ?*?*AionLoadedModel,
) callconv(.c) AionStatus {
    const opts = loadOptionsFromC(opts_opt) orelse return .AION_INVALID_ARGUMENT;
    return loadModelImpl(ctx_opt, path, true, opts, out_model, "load_model_path_absolute");
}

pub export fn aion_loaded_model_destroy(m_opt: ?*AionLoadedModel) callconv(.c) void {
    const m: *AionLoadedModel = m_opt orelse return;
    m.model.deinit();
    std.heap.page_allocator.destroy(m);
}

pub export fn aion_loaded_model_input_count(m_opt: ?*const AionLoadedModel) callconv(.c) usize {
    const m: *const AionLoadedModel = m_opt orelse return 0;
    return m.model.inputNames().len;
}

pub export fn aion_loaded_model_output_count(m_opt: ?*const AionLoadedModel) callconv(.c) usize {
    const m: *const AionLoadedModel = m_opt orelse return 0;
    return m.model.outputNames().len;
}

pub export fn aion_loaded_model_input_name(
    m_opt: ?*const AionLoadedModel,
    index: usize,
    buf: [*c]u8,
    cap: usize,
    out_len: ?*usize,
) callconv(.c) AionStatus {
    const m: *const AionLoadedModel = m_opt orelse return .AION_INVALID_ARGUMENT;
    const sigs = m.model.inputNames();
    if (index >= sigs.len) return .AION_INVALID_ARGUMENT;
    copyStringToBuf(sigs[index].name, buf, cap, out_len);
    return .AION_OK;
}

pub export fn aion_loaded_model_output_name(
    m_opt: ?*const AionLoadedModel,
    index: usize,
    buf: [*c]u8,
    cap: usize,
    out_len: ?*usize,
) callconv(.c) AionStatus {
    const m: *const AionLoadedModel = m_opt orelse return .AION_INVALID_ARGUMENT;
    const sigs = m.model.outputNames();
    if (index >= sigs.len) return .AION_INVALID_ARGUMENT;
    copyStringToBuf(sigs[index].name, buf, cap, out_len);
    return .AION_OK;
}

pub export fn aion_loaded_model_input_dtype(m_opt: ?*const AionLoadedModel, index: usize, out_dtype: ?*AionDType) callconv(.c) AionStatus {
    const m: *const AionLoadedModel = m_opt orelse return .AION_INVALID_ARGUMENT;
    if (out_dtype == null) return .AION_INVALID_ARGUMENT;
    const sigs = m.model.inputNames();
    if (index >= sigs.len) return .AION_INVALID_ARGUMENT;
    out_dtype.?.* = dtypeToC(sigs[index].dtype);
    return .AION_OK;
}

pub export fn aion_loaded_model_input_rank(m_opt: ?*const AionLoadedModel, index: usize, out_rank: ?*usize) callconv(.c) AionStatus {
    const m: *const AionLoadedModel = m_opt orelse return .AION_INVALID_ARGUMENT;
    if (out_rank == null) return .AION_INVALID_ARGUMENT;
    const sigs = m.model.inputNames();
    if (index >= sigs.len) return .AION_INVALID_ARGUMENT;
    out_rank.?.* = @as(usize, sigs[index].rank);
    return .AION_OK;
}

pub export fn aion_loaded_model_output_dtype(m_opt: ?*const AionLoadedModel, index: usize, out_dtype: ?*AionDType) callconv(.c) AionStatus {
    const m: *const AionLoadedModel = m_opt orelse return .AION_INVALID_ARGUMENT;
    if (out_dtype == null) return .AION_INVALID_ARGUMENT;
    const sigs = m.model.outputNames();
    if (index >= sigs.len) return .AION_INVALID_ARGUMENT;
    out_dtype.?.* = dtypeToC(sigs[index].dtype);
    return .AION_OK;
}

pub export fn aion_loaded_model_output_rank(m_opt: ?*const AionLoadedModel, index: usize, out_rank: ?*usize) callconv(.c) AionStatus {
    const m: *const AionLoadedModel = m_opt orelse return .AION_INVALID_ARGUMENT;
    if (out_rank == null) return .AION_INVALID_ARGUMENT;
    const sigs = m.model.outputNames();
    if (index >= sigs.len) return .AION_INVALID_ARGUMENT;
    out_rank.?.* = @as(usize, sigs[index].rank);
    return .AION_OK;
}

pub export fn aion_loaded_model_bind_input(
    m_opt: ?*AionLoadedModel,
    name: ?[*:0]const u8,
    t_opt: ?*const AionTensor,
) callconv(.c) AionStatus {
    const m: *AionLoadedModel = m_opt orelse return .AION_INVALID_ARGUMENT;
    const ctx: *AionContext = m.owner;
    ctx.clearLastError();

    if (name == null) return .AION_INVALID_ARGUMENT;
    const t: *const AionTensor = t_opt orelse return .AION_INVALID_ARGUMENT;

    const n: []const u8 = std.mem.span(name.?);
    m.model.bindInput(n, t.tensor) catch |e| {
        ctx.setLastError("bind_input", e);
        return mapError(e);
    };
    return .AION_OK;
}

pub export fn aion_loaded_model_run(m_opt: ?*AionLoadedModel) callconv(.c) AionStatus {
    const m: *AionLoadedModel = m_opt orelse return .AION_INVALID_ARGUMENT;
    const ctx: *AionContext = m.owner;
    ctx.clearLastError();

    m.model.run() catch |e| {
        ctx.setLastError("loaded_model_run", e);
        return mapError(e);
    };
    return .AION_OK;
}

/// Tokens consumed so far by position auto-management (0 when disabled).
pub export fn aion_loaded_model_position(m_opt: ?*const AionLoadedModel, out_tokens: ?*u64) callconv(.c) AionStatus {
    const m: *const AionLoadedModel = m_opt orelse return .AION_INVALID_ARGUMENT;
    if (out_tokens == null) return .AION_INVALID_ARGUMENT;
    out_tokens.?.* = m.model.currentPosition();
    return .AION_OK;
}

/// Overwrite the auto-tracked position (session restore / rollback).
pub export fn aion_loaded_model_set_position(m_opt: ?*AionLoadedModel, tokens: u64) callconv(.c) AionStatus {
    const m: *AionLoadedModel = m_opt orelse return .AION_INVALID_ARGUMENT;
    m.model.setPosition(tokens);
    return .AION_OK;
}

/// Whether output `index` is io-aliased recurrent state (a carry the runtime
/// writes back into its input slot on every run). Lets bindings skip copying
/// such outputs out by default.
pub export fn aion_loaded_model_output_is_state(
    m_opt: ?*const AionLoadedModel,
    index: usize,
    out_is_state: ?*u8,
) callconv(.c) AionStatus {
    const m: *const AionLoadedModel = m_opt orelse return .AION_INVALID_ARGUMENT;
    if (out_is_state == null) return .AION_INVALID_ARGUMENT;
    if (index >= m.model.outputNames().len) return .AION_INVALID_ARGUMENT;
    out_is_state.?.* = @intFromBool(m.model.outputIsAliasedState(index));
    return .AION_OK;
}

pub export fn aion_loaded_model_reset_state(m_opt: ?*AionLoadedModel) callconv(.c) AionStatus {
    const m: *AionLoadedModel = m_opt orelse return .AION_INVALID_ARGUMENT;
    const ctx: *AionContext = m.owner;
    ctx.clearLastError();

    m.model.resetState() catch |e| {
        ctx.setLastError("loaded_model_reset_state", e);
        return mapError(e);
    };
    return .AION_OK;
}

/// Declare a sequence-cache policy for an io-aliased recurrent-state input by
/// name. `kind`: 0 = none (fixed), 1 = growable, 2 = ring. Growable uses
/// `initial_capacity_tokens` / `growth_numerator` / `growth_denominator` /
/// `max_capacity_tokens`; ring uses `ring_window_tokens`. Lets a caller allocate
/// a small initial cache and have the runtime grow its slot on demand up to the
/// bound (device-resident growth included). Unused fields are ignored per kind.
pub export fn aion_loaded_model_set_state_input_policy(
    m_opt: ?*AionLoadedModel,
    name: ?[*:0]const u8,
    kind: u32,
    initial_capacity_tokens: u64,
    growth_numerator: u64,
    growth_denominator: u64,
    max_capacity_tokens: u64,
    ring_window_tokens: u64,
) callconv(.c) AionStatus {
    const m: *AionLoadedModel = m_opt orelse return .AION_INVALID_ARGUMENT;
    const ctx: *AionContext = m.owner;
    ctx.clearLastError();
    if (name == null) return .AION_INVALID_ARGUMENT;
    const n: []const u8 = std.mem.span(name.?);

    const policy: api.SequenceCachePolicy = switch (kind) {
        1 => .{ .growable = .{
            .initial_capacity_tokens = @intCast(initial_capacity_tokens),
            .growth_numerator = @intCast(growth_numerator),
            .growth_denominator = @intCast(growth_denominator),
            .max_capacity_tokens = @intCast(max_capacity_tokens),
        } },
        2 => .{ .ring = .{ .window_tokens = @intCast(ring_window_tokens) } },
        else => .{ .none = {} },
    };
    m.model.setStateInputPolicy(n, policy) catch |e| {
        ctx.setLastError("set_state_input_policy", e);
        return mapError(e);
    };
    return .AION_OK;
}

pub export fn aion_loaded_model_output_tensor(
    m_opt: ?*AionLoadedModel,
    name: ?[*:0]const u8,
    out_tensor: ?*?*AionTensor,
) callconv(.c) AionStatus {
    const m: *AionLoadedModel = m_opt orelse return .AION_INVALID_ARGUMENT;
    const ctx: *AionContext = m.owner;
    ctx.clearLastError();

    if (out_tensor == null) return .AION_INVALID_ARGUMENT;
    out_tensor.?.* = null;
    if (name == null) return .AION_INVALID_ARGUMENT;

    const n: []const u8 = std.mem.span(name.?);
    const t: api.Tensor = m.model.outputTensor(n) catch |e| {
        ctx.setLastError("output_tensor", e);
        return mapError(e);
    };

    const handle: *AionTensor = std.heap.page_allocator.create(AionTensor) catch {
        ctx.setLastError("tensor_handle_alloc", error.OutOfMemory);
        return .AION_OUT_OF_MEMORY;
    };
    handle.* = .{ .owner = ctx, .tensor = t };
    out_tensor.?.* = handle;
    return .AION_OK;
}

// ============================================================================
// Model authoring (Builder → compile/export)
//
// Mirrors the Zig `api.Builder` + `Context.compile`/`exportModel`. Values are
// plain `AionValueId` (u32) scoped to a builder — not heap handles. All ~32 ops
// go through ONE entry point, `aion_builder_op`, taking an `AionOpSpec` whose
// attribute union mirrors the core `graph.zig` `Op` union field-for-field.
// ============================================================================

/// Opaque id of a value produced/consumed inside a builder. Mirrors
/// `api.TensorRef` (which is just a `u32`).
pub const AionValueId = u32;

pub const AionUnaryOp = enum(c_int) {
    AION_UNARY_RELU = 0,
    AION_UNARY_GELU = 1,
    AION_UNARY_SILU = 2,
    AION_UNARY_SIGMOID = 3,
    AION_UNARY_TANH = 4,
    AION_UNARY_SQRT = 5,
    AION_UNARY_LOG = 6,
};

pub const AionBinaryOp = enum(c_int) {
    AION_BINARY_ADD = 0,
    AION_BINARY_SUB = 1,
    AION_BINARY_MUL = 2,
    AION_BINARY_DIV = 3,
    AION_BINARY_EQ = 4,
    AION_BINARY_NE = 5,
    AION_BINARY_LT = 6,
    AION_BINARY_GT = 7,
    AION_BINARY_LE = 8,
    AION_BINARY_GE = 9,
    /// Gated activation: `elemwise.act`(a) * b — GEGLU/SwiGLU/GLU/ReGLU.
    AION_BINARY_GATE = 10,
};

pub const AionReduceOp = enum(c_int) {
    AION_REDUCE_SUM = 0,
    AION_REDUCE_MEAN = 1,
};

pub const AionPadMode = enum(c_int) {
    AION_PAD_ZERO = 0,
    AION_PAD_REFLECT = 1,
};

pub const AionInputRoleKind = enum(c_int) {
    AION_ROLE_SEQUENCE_CACHE = 1,
    AION_ROLE_CACHE_WRITE_INDEX = 2,
    AION_ROLE_CACHE_VISIBLE_END = 3,
    AION_ROLE_POSITIONS = 4,
    AION_ROLE_TOKENS = 5,
    AION_ROLE_STATE = 6,
};

/// Op selector for `aion_builder_op`. Each value pairs with the like-named
/// member of `AionOpAttr` (ops with no attributes ignore `attr`).
pub const AionOp = enum(c_int) {
    AION_OP_MATMUL = 0,
    AION_OP_MATMUL_NT = 1,
    AION_OP_ELEMWISE = 2,
    AION_OP_UNARY = 3,
    AION_OP_SOFTMAX = 4,
    AION_OP_LAYERNORM = 5,
    AION_OP_RMSNORM = 6,
    AION_OP_ATTENTION = 7,
    AION_OP_RELPOS_MHA = 8,
    AION_OP_CONV1D = 9,
    AION_OP_CONV2D = 10,
    AION_OP_COPY = 11,
    AION_OP_ROPE1D = 12,
    AION_OP_SEQUENCE_APPEND = 13,
    AION_OP_REDUCE = 14,
    AION_OP_CONCAT = 15,
    AION_OP_RESHAPE = 16,
    AION_OP_SQUEEZE = 17,
    AION_OP_UNSQUEEZE = 18,
    AION_OP_TRANSPOSE2D = 19,
    AION_OP_SLICE = 20,
    AION_OP_LSTM_CELL = 21,
    AION_OP_RFFT = 22,
    AION_OP_STFT = 23,
    AION_OP_CAST = 24,
    AION_OP_ARGMAX = 25,
    AION_OP_SCATTER_ROW = 26,
    // 27 was AION_OP_GELU_MUL, retired: a gated activation is AION_OP_ELEMWISE with
    // `elemwise.op = AION_BINARY_GATE` and `elemwise.act` naming the activation.
    AION_OP_GATHER = 28,
    AION_OP_DIM = 29,
    AION_OP_IOTA = 30,
};

/// Per-op attributes. Only the member matching `AionOpSpec.op` is read.
pub const AionOpAttr = extern union {
    matmul: extern struct { alpha: f32, beta: f32 },
    elemwise: extern struct { op: AionBinaryOp, act: AionUnaryOp = .AION_UNARY_GELU },
    unary: extern struct { op: AionUnaryOp },
    softmax: extern struct { axis: i32 },
    norm: extern struct { eps: f32, normalized_shape: [*c]const usize, normalized_shape_len: usize },
    attention: extern struct {
        scale: f32,
        causal: u8,
        sliding_window: usize,
        attn_logits_soft_cap: f32,
        has_query_positions: u8,
        has_kv_lengths: u8,
    },
    relpos_mha: extern struct { scale: f32, chunk_size: usize = 0, chunk_left: usize = 0 },
    conv1d: extern struct {
        stride: usize,
        dilation: usize,
        pad_left: usize,
        pad_right: usize,
        groups: usize,
        pad_mode: AionPadMode,
    },
    conv2d: extern struct {
        stride_h: usize,
        stride_w: usize,
        dilation_h: usize,
        dilation_w: usize,
        pad_top: usize,
        pad_bottom: usize,
        pad_left: usize,
        pad_right: usize,
        groups: usize,
        pad_mode: AionPadMode,
    },
    rope1d: extern struct { base_frequency: f32, scale_factor: f32, rope_proportion: f32 },
    reduce: extern struct { op: AionReduceOp, axis: i32, has_axis: u8 },
    concat: extern struct { axis: i32 },
    reshape: extern struct { shape: [*c]const usize, shape_len: usize, shape_symbols: [*c]const [*c]const u8 },
    squeeze: extern struct { axis: i32, has_axis: u8 },
    unsqueeze: extern struct { axis: i32 },
    slice: extern struct { starts: [*c]const usize, lens: [*c]const usize, len: usize, len_symbols: [*c]const [*c]const u8 },
    stft: extern struct { n_fft: usize, hop_length: usize, center: u8 },
    cast: extern struct { to_dtype: AionDType },
    argmax: extern struct { axis: i32 },
    gather: extern struct { axis: i32, batch_dims: usize },
};

/// A single op to append to a builder: which op, its input value ids (like a
/// graph `Node`'s inputs), and its attributes.
pub const AionOpSpec = extern struct {
    op: AionOp,
    inputs: [*c]const AionValueId,
    inputs_len: usize,
    attr: AionOpAttr,
};

pub const AionBuilder = struct {
    owner: *AionContext,
    builder: api.Builder,

    outputs: std.ArrayList(api.NamedTensorRef) = .empty,
    aliases: std.ArrayList(api.OutputAlias) = .empty,
    roles: std.ArrayList(api.InputRoleDecl) = .empty,
    metadata: std.ArrayList(api.ExportMetadata) = .empty,

    /// Stable storage for strings duped out of caller-owned C strings that must
    /// outlive the accumulate call (metadata, dim-symbol names, role symbols).
    str_arena: std.heap.ArenaAllocator,

    fn alloc(self: *AionBuilder) std.mem.Allocator {
        return self.owner.allocator;
    }

    fn dupeStr(self: *AionBuilder, s: []const u8) ![]const u8 {
        const buf = try self.str_arena.allocator().alloc(u8, s.len);
        @memcpy(buf, s);
        return buf;
    }
};

fn unaryFromC(op: AionUnaryOp) types.UnaryOp {
    return switch (op) {
        .AION_UNARY_RELU => .relu,
        .AION_UNARY_GELU => .gelu,
        .AION_UNARY_SILU => .silu,
        .AION_UNARY_SIGMOID => .sigmoid,
        .AION_UNARY_TANH => .tanh,
        .AION_UNARY_SQRT => .sqrt,
        .AION_UNARY_LOG => .log,
    };
}

fn binaryFromC(op: AionBinaryOp) types.ElemwiseBinaryOp {
    return switch (op) {
        .AION_BINARY_ADD => .add,
        .AION_BINARY_SUB => .sub,
        .AION_BINARY_MUL => .mul,
        .AION_BINARY_DIV => .div,
        .AION_BINARY_EQ => .eq,
        .AION_BINARY_NE => .ne,
        .AION_BINARY_LT => .lt,
        .AION_BINARY_GT => .gt,
        .AION_BINARY_LE => .le,
        .AION_BINARY_GE => .ge,
        .AION_BINARY_GATE => .gate,
    };
}

fn reduceFromC(op: AionReduceOp) types.ReduceOp {
    return switch (op) {
        .AION_REDUCE_SUM => .sum,
        .AION_REDUCE_MEAN => .mean,
    };
}

fn padModeFromC(m: AionPadMode) types.PadMode {
    return switch (m) {
        .AION_PAD_ZERO => .zero,
        .AION_PAD_REFLECT => .reflect,
    };
}

/// A C string pointer (possibly null) from a symbol array -> optional slice.
fn cStrOrNull(s: [*c]const u8) ?[]const u8 {
    if (s == null) return null;
    return std.mem.span(@as([*:0]const u8, @ptrCast(s)));
}

fn roleKindFromC(k: AionInputRoleKind) pkg_types.InputRoleKind {
    return switch (k) {
        .AION_ROLE_SEQUENCE_CACHE => .sequence_cache,
        .AION_ROLE_CACHE_WRITE_INDEX => .cache_write_index,
        .AION_ROLE_CACHE_VISIBLE_END => .cache_visible_end,
        .AION_ROLE_POSITIONS => .positions,
        .AION_ROLE_TOKENS => .tokens,
        .AION_ROLE_STATE => .state,
    };
}

fn vref(id: AionValueId) api.TensorRef {
    return .{ .value = id };
}

pub export fn aion_builder_create(ctx_opt: ?*AionContext, out_builder: ?*?*AionBuilder) callconv(.c) AionStatus {
    const ctx: *AionContext = ctx_opt orelse return .AION_INVALID_ARGUMENT;
    ctx.clearLastError();
    if (out_builder == null) return .AION_INVALID_ARGUMENT;
    out_builder.?.* = null;

    const b: *AionBuilder = std.heap.page_allocator.create(AionBuilder) catch {
        ctx.setLastError("builder_alloc", error.OutOfMemory);
        return .AION_OUT_OF_MEMORY;
    };
    b.* = .{
        .owner = ctx,
        .builder = api.Builder.init(&ctx.ctx),
        .str_arena = std.heap.ArenaAllocator.init(ctx.allocator),
    };
    out_builder.?.* = b;
    return .AION_OK;
}

pub export fn aion_builder_destroy(b_opt: ?*AionBuilder) callconv(.c) void {
    const b: *AionBuilder = b_opt orelse return;
    const a = b.owner.allocator;
    b.outputs.deinit(a);
    b.aliases.deinit(a);
    b.roles.deinit(a);
    b.metadata.deinit(a);
    b.str_arena.deinit();
    b.builder.deinit();
    std.heap.page_allocator.destroy(b);
}

pub export fn aion_builder_input(
    b_opt: ?*AionBuilder,
    dtype_c: AionDType,
    rank: usize,
    shape_ptr: [*c]const usize,
    out_value: ?*AionValueId,
) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    b.owner.clearLastError();
    if (out_value == null) return .AION_INVALID_ARGUMENT;
    const shape: []const usize = getShapeSlice(rank, shape_ptr) orelse return .AION_INVALID_ARGUMENT;
    const dt: types.DType = dtypeFromC(dtype_c) orelse return .AION_INVALID_ARGUMENT;
    const ref = b.builder.input(dt, shape) catch |e| {
        b.owner.setLastError("builder_input", e);
        return mapError(e);
    };
    out_value.?.* = ref.value;
    return .AION_OK;
}

pub export fn aion_builder_param(
    b_opt: ?*AionBuilder,
    t_opt: ?*const AionTensor,
    out_value: ?*AionValueId,
) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    b.owner.clearLastError();
    if (out_value == null) return .AION_INVALID_ARGUMENT;
    const t: *const AionTensor = t_opt orelse return .AION_INVALID_ARGUMENT;
    const ref = b.builder.param(t.tensor) catch |e| {
        b.owner.setLastError("builder_param", e);
        return mapError(e);
    };
    out_value.?.* = ref.value;
    return .AION_OK;
}

pub export fn aion_builder_name(b_opt: ?*AionBuilder, value: AionValueId, name: ?[*:0]const u8) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    b.owner.clearLastError();
    if (name == null) return .AION_INVALID_ARGUMENT;
    _ = b.builder.name(vref(value), std.mem.span(name.?)) catch |e| {
        b.owner.setLastError("builder_name", e);
        return mapError(e);
    };
    return .AION_OK;
}

/// Bind a weight under a semantic, scope-qualified name (`layers.3/attn/weight`).
///
/// Unlike `aion_builder_param`, whose generated name is positional and shifts when
/// construction order changes, this produces the stable key that load/swap-by-name
/// depends on.
pub export fn aion_builder_param_named(
    b_opt: ?*AionBuilder,
    tensor: ?*const AionTensor,
    name: ?[*:0]const u8,
    out_value: ?*AionValueId,
) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    b.owner.clearLastError();
    const t: *const AionTensor = tensor orelse return .AION_INVALID_ARGUMENT;
    if (name == null or out_value == null) return .AION_INVALID_ARGUMENT;

    const ref = b.builder.paramNamed(t.tensor, std.mem.span(name.?)) catch |e| {
        b.owner.setLastError("builder_param_named", e);
        return mapError(e);
    };
    out_value.?.* = ref.value;
    return .AION_OK;
}

// --- scopes ------------------------------------------------------------------
// Scopes nest, so a module tree produces `state_dict`-style parameter paths.
// `begin_*` hands back the depth it opened at; pass that to `end_scope`, which
// closes every scope at or below it (so an early return cannot leak a level).

pub export fn aion_builder_begin_scope(
    b_opt: ?*AionBuilder,
    name: ?[*:0]const u8,
    out_depth: ?*usize,
) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    b.owner.clearLastError();
    if (name == null) return .AION_INVALID_ARGUMENT;

    const scope = b.builder.beginScope(std.mem.span(name.?)) catch |e| {
        b.owner.setLastError("builder_begin_scope", e);
        return mapError(e);
    };
    if (out_depth) |o| o.* = scope.depth;
    return .AION_OK;
}

/// Open a scope named `{base}#{n}`, numbered per parent scope. The resolved
/// segment is written to `buf` when provided.
pub export fn aion_builder_begin_auto_scope(
    b_opt: ?*AionBuilder,
    base: ?[*:0]const u8,
    out_depth: ?*usize,
    buf: [*c]u8,
    cap: usize,
    out_len: ?*usize,
) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    b.owner.clearLastError();
    if (base == null) return .AION_INVALID_ARGUMENT;

    const scope = b.builder.beginAutoScope(std.mem.span(base.?)) catch |e| {
        b.owner.setLastError("builder_begin_auto_scope", e);
        return mapError(e);
    };
    if (out_depth) |o| o.* = scope.depth;
    copyStringToBuf(scope.name, buf, cap, out_len);
    return .AION_OK;
}

pub export fn aion_builder_end_scope(b_opt: ?*AionBuilder, depth: usize) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    b.builder.endScopeAtDepth(depth);
    return .AION_OK;
}

pub export fn aion_builder_scope_path(
    b_opt: ?*const AionBuilder,
    buf: [*c]u8,
    cap: usize,
    out_len: ?*usize,
) callconv(.c) AionStatus {
    const b: *const AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    copyStringToBuf(b.builder.scopePath(), buf, cap, out_len);
    return .AION_OK;
}

// --- synthesized constants ----------------------------------------------------
// Several ops require an operand where the maths wants a number. These bind (and
// cache) that operand, so a model that needs the same identity vector or scalar in
// many layers pays for one.

/// A cached one-element f32 constant, for scalar affine through the broadcast op.
pub export fn aion_builder_constant(
    b_opt: ?*AionBuilder,
    value: f32,
    out_value: ?*AionValueId,
) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    b.owner.clearLastError();
    if (out_value == null) return .AION_INVALID_ARGUMENT;

    const ref = b.builder.constant(value) catch |e| {
        b.owner.setLastError("builder_constant", e);
        return mapError(e);
    };
    out_value.?.* = ref.value;
    return .AION_OK;
}

/// A cached `[dim]` f32 vector filled with `fill` — the identity `gamma` (1.0) or
/// `beta` (0.0) of a norm.
pub export fn aion_builder_filled_vec(
    b_opt: ?*AionBuilder,
    dim: usize,
    fill: f32,
    out_value: ?*AionValueId,
) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    b.owner.clearLastError();
    if (out_value == null) return .AION_INVALID_ARGUMENT;

    const ref = (if (fill == 0.0) b.builder.zeros(dim) else if (fill == 1.0) b.builder.ones(dim) else {
        b.owner.setLastError("builder_filled_vec", error.InvalidArgument);
        return .AION_INVALID_ARGUMENT;
    }) catch |e| {
        b.owner.setLastError("builder_filled_vec", e);
        return mapError(e);
    };
    out_value.?.* = ref.value;
    return .AION_OK;
}

/// 0 = not a bound parameter, 1 = a user-supplied weight, 2 = a synthesized
/// constant. Distinguishes model state from operands the Builder had to invent.
pub export fn aion_builder_param_kind(
    b_opt: ?*const AionBuilder,
    value: AionValueId,
    out_kind: ?*u32,
) callconv(.c) AionStatus {
    const b: *const AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    if (out_kind == null) return .AION_INVALID_ARGUMENT;
    out_kind.?.* = switch (b.builder.paramKind(vref(value)) orelse {
        out_kind.?.* = 0;
        return .AION_OK;
    }) {
        .user => 1,
        .synthesized => 2,
    };
    return .AION_OK;
}

/// The debug name attached to a value, or empty when it has none.
///
/// This is the name that lands in the package and that load/swap-by-name uses, so
/// reading it back is how a caller sees a model's parameter paths without
/// recomputing them.
pub export fn aion_builder_value_name(
    b_opt: ?*const AionBuilder,
    value: AionValueId,
    buf: [*c]u8,
    cap: usize,
    out_len: ?*usize,
) callconv(.c) AionStatus {
    const b: *const AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    copyStringToBuf(b.builder.valueName(vref(value)) orelse "", buf, cap, out_len);
    return .AION_OK;
}

/// Whether the graph binds a parameter under `name`, of either kind.
pub export fn aion_builder_has_param_named(
    b_opt: ?*const AionBuilder,
    name: ?[*:0]const u8,
    out_found: ?*u8,
) callconv(.c) AionStatus {
    const b: *const AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    if (name == null or out_found == null) return .AION_INVALID_ARGUMENT;
    out_found.?.* = if (b.builder.hasParamNamed(std.mem.span(name.?))) 1 else 0;
    return .AION_OK;
}

/// Authoring-time shape introspection (available thanks to eager per-op
/// inference). Shapes reflect the authoring *placeholder* sizes: an axis declared
/// dynamic reports the size it was declared with, propagated through derived
/// values. Mirrors the tensor rank/shape/dtype readers.
pub export fn aion_builder_value_rank(b_opt: ?*const AionBuilder, value: AionValueId, out_rank: ?*usize) callconv(.c) AionStatus {
    const b: *const AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    if (out_rank == null) return .AION_INVALID_ARGUMENT;
    const shape = b.builder.knownShape(vref(value)) orelse return .AION_INVALID_ARGUMENT;
    out_rank.?.* = shape.len;
    return .AION_OK;
}

pub export fn aion_builder_value_shape(b_opt: ?*const AionBuilder, value: AionValueId, out_dims: [*c]usize, out_rank: usize) callconv(.c) AionStatus {
    const b: *const AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    if (out_dims == null) return .AION_INVALID_ARGUMENT;
    const shape = b.builder.knownShape(vref(value)) orelse return .AION_INVALID_ARGUMENT;
    if (out_rank != shape.len) return .AION_INVALID_ARGUMENT;
    @memcpy(out_dims[0..out_rank], shape);
    return .AION_OK;
}

/// The dim symbol on `value`'s `axis`, if that axis is free.
///
/// Writes the NUL-terminated name into `buf` (truncated to `cap`) and its full
/// length into `out_len`, like the other string readers. An axis with a fixed size
/// yields an empty name and length 0 — a normal answer, not an error.
///
/// This is how a binding builds a shape that keeps its input's free axes without
/// re-tracking symbols on its own side: ask, then pass the answer back to a view op.
pub export fn aion_builder_value_dim_symbol(
    b_opt: ?*const AionBuilder,
    value: AionValueId,
    axis: usize,
    buf: [*c]u8,
    cap: usize,
    out_len: ?*usize,
) callconv(.c) AionStatus {
    const b: *const AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    const dim = b.builder.dimAt(vref(value), axis) catch return .AION_INVALID_ARGUMENT;
    copyStringToBuf(switch (dim) {
        .size => "",
        .symbol => |s| s,
    }, buf, cap, out_len);
    return .AION_OK;
}

/// The authoring placeholder size of a declared dim symbol.
///
/// A free axis still needs a concrete size to author with; this is where a caller
/// that wants to build a symbolic view dim gets one, so the placeholder is only
/// ever recorded in one place — the builder that took the declaration.
pub export fn aion_builder_symbol_size(
    b_opt: ?*const AionBuilder,
    name: ?[*:0]const u8,
    out_size: ?*usize,
) callconv(.c) AionStatus {
    const b: *const AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    if (name == null or out_size == null) return .AION_INVALID_ARGUMENT;
    // Undeclared is a caller error, same as it is on the Zig side.
    out_size.?.* = b.builder.symbolSize(std.mem.span(name.?)) orelse return .AION_INVALID_ARGUMENT;
    return .AION_OK;
}

pub export fn aion_builder_value_dtype(b_opt: ?*const AionBuilder, value: AionValueId, out_dtype: ?*AionDType) callconv(.c) AionStatus {
    const b: *const AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    if (out_dtype == null) return .AION_INVALID_ARGUMENT;
    const dt = b.builder.dtypeOf(vref(value)) orelse return .AION_INVALID_ARGUMENT;
    out_dtype.?.* = dtypeToC(dt);
    return .AION_OK;
}

fn builderOpImpl(b: *AionBuilder, spec: *const AionOpSpec) api.Builder.Error!AionValueId {
    const bld = &b.builder;
    const n = spec.inputs_len;
    // Bounds helper: require at least `k` inputs.
    const need = struct {
        fn f(cnt: usize, k: usize, ptr: [*c]const AionValueId) api.Builder.Error!void {
            if (cnt < k or (k > 0 and ptr == null)) return api.Builder.Error.InvalidArgument;
        }
    }.f;
    const in = struct {
        fn f(s: *const AionOpSpec, i: usize) api.TensorRef {
            return .{ .value = s.inputs[i] };
        }
    }.f;

    const out: api.TensorRef = switch (spec.op) {
        .AION_OP_MATMUL => blk: {
            try need(n, 2, spec.inputs);
            break :blk try bld.matmul(in(spec, 0), in(spec, 1), spec.attr.matmul.alpha, spec.attr.matmul.beta);
        },
        .AION_OP_MATMUL_NT => blk: {
            try need(n, 2, spec.inputs);
            break :blk try bld.matmulNT(in(spec, 0), in(spec, 1), spec.attr.matmul.alpha, spec.attr.matmul.beta);
        },
        .AION_OP_ELEMWISE => blk: {
            try need(n, 2, spec.inputs);
            // `act` is read only for a gate, so every other op ignores whatever is there.
            break :blk if (spec.attr.elemwise.op == .AION_BINARY_GATE)
                try bld.gate(unaryFromC(spec.attr.elemwise.act), in(spec, 0), in(spec, 1))
            else
                try bld.elemwiseBinary(binaryFromC(spec.attr.elemwise.op), in(spec, 0), in(spec, 1));
        },
        .AION_OP_UNARY => blk: {
            try need(n, 1, spec.inputs);
            break :blk try bld.unary(unaryFromC(spec.attr.unary.op), in(spec, 0));
        },
        .AION_OP_SOFTMAX => blk: {
            try need(n, 1, spec.inputs);
            break :blk try bld.softmax(in(spec, 0), spec.attr.softmax.axis);
        },
        .AION_OP_LAYERNORM, .AION_OP_RMSNORM => blk: {
            try need(n, 3, spec.inputs);
            if (spec.attr.norm.normalized_shape == null) return api.Builder.Error.InvalidArgument;
            const ns = spec.attr.norm.normalized_shape[0..spec.attr.norm.normalized_shape_len];
            break :blk if (spec.op == .AION_OP_LAYERNORM)
                try bld.layernorm(in(spec, 0), in(spec, 1), in(spec, 2), spec.attr.norm.eps, ns)
            else
                try bld.rmsnorm(in(spec, 0), in(spec, 1), in(spec, 2), spec.attr.norm.eps, ns);
        },
        .AION_OP_ATTENTION => blk: {
            try need(n, 3, spec.inputs);
            const a = spec.attr.attention;
            var control_idx: usize = 3;
            const query_positions: ?api.TensorRef = if (a.has_query_positions != 0) blk2: {
                if (control_idx >= n) return api.Builder.Error.InvalidArgument;
                defer control_idx += 1;
                break :blk2 in(spec, control_idx);
            } else null;
            const kv_lengths: ?api.TensorRef = if (a.has_kv_lengths != 0) blk2: {
                if (control_idx >= n) return api.Builder.Error.InvalidArgument;
                defer control_idx += 1;
                break :blk2 in(spec, control_idx);
            } else null;
            if (control_idx != n) return api.Builder.Error.InvalidArgument;
            break :blk try bld.attention(
                in(spec, 0),
                in(spec, 1),
                in(spec, 2),
                query_positions,
                kv_lengths,
                a.scale,
                a.causal != 0,
                a.sliding_window,
                a.attn_logits_soft_cap,
            );
        },
        .AION_OP_RELPOS_MHA => blk: {
            try need(n, 6, spec.inputs);
            const mask: ?api.TensorRef = if (n >= 7) in(spec, 6) else null;
            break :blk try bld.relPosMHA(
                in(spec, 0),
                in(spec, 1),
                in(spec, 2),
                in(spec, 3),
                in(spec, 4),
                in(spec, 5),
                mask,
                spec.attr.relpos_mha.scale,
                .{
                    .size = spec.attr.relpos_mha.chunk_size,
                    .left = spec.attr.relpos_mha.chunk_left,
                },
            );
        },
        .AION_OP_CONV1D => blk: {
            try need(n, 2, spec.inputs);
            const bias: ?api.TensorRef = if (n >= 3) in(spec, 2) else null;
            break :blk try bld.conv1dPadMode(
                in(spec, 0),
                in(spec, 1),
                bias,
                spec.attr.conv1d.stride,
                spec.attr.conv1d.dilation,
                spec.attr.conv1d.pad_left,
                spec.attr.conv1d.pad_right,
                padModeFromC(spec.attr.conv1d.pad_mode),
                spec.attr.conv1d.groups,
            );
        },
        .AION_OP_CONV2D => blk: {
            try need(n, 2, spec.inputs);
            const bias: ?api.TensorRef = if (n >= 3) in(spec, 2) else null;
            const c = spec.attr.conv2d;
            break :blk try bld.conv2dPadMode(
                in(spec, 0),
                in(spec, 1),
                bias,
                c.stride_h,
                c.stride_w,
                c.dilation_h,
                c.dilation_w,
                c.pad_top,
                c.pad_bottom,
                c.pad_left,
                c.pad_right,
                padModeFromC(c.pad_mode),
                c.groups,
            );
        },
        .AION_OP_COPY => blk: {
            try need(n, 1, spec.inputs);
            break :blk try bld.copy(in(spec, 0));
        },
        .AION_OP_ROPE1D => blk: {
            try need(n, 2, spec.inputs);
            break :blk try bld.rope1d(in(spec, 0), in(spec, 1), spec.attr.rope1d.base_frequency, spec.attr.rope1d.scale_factor, spec.attr.rope1d.rope_proportion);
        },
        .AION_OP_SEQUENCE_APPEND => blk: {
            try need(n, 3, spec.inputs);
            break :blk try bld.sequenceAppend(in(spec, 0), in(spec, 1), in(spec, 2));
        },
        .AION_OP_REDUCE => blk: {
            try need(n, 1, spec.inputs);
            break :blk if (spec.attr.reduce.has_axis != 0)
                try bld.reduceAxis(reduceFromC(spec.attr.reduce.op), in(spec, 0), spec.attr.reduce.axis)
            else
                try bld.reduce(reduceFromC(spec.attr.reduce.op), in(spec, 0));
        },
        .AION_OP_CONCAT => blk: {
            try need(n, 1, spec.inputs);
            if (n > 16) return api.Builder.Error.InvalidArgument;
            var refs: [16]api.TensorRef = undefined;
            var i: usize = 0;
            while (i < n) : (i += 1) refs[i] = in(spec, i);
            break :blk try bld.concat(refs[0..n], spec.attr.concat.axis);
        },
        .AION_OP_RESHAPE => blk: {
            try need(n, 1, spec.inputs);
            if (spec.attr.reshape.shape == null) return api.Builder.Error.InvalidArgument;
            const shp = spec.attr.reshape.shape[0..spec.attr.reshape.shape_len];
            const syms = spec.attr.reshape.shape_symbols;
            if (syms != null) {
                if (shp.len > 8) return api.Builder.Error.InvalidArgument;
                var buf: [8]?[]const u8 = undefined;
                for (0..shp.len) |i| buf[i] = cStrOrNull(syms[i]);
                break :blk try bld.reshapeSym(in(spec, 0), shp, buf[0..shp.len]);
            }
            break :blk try bld.reshape(in(spec, 0), shp);
        },
        .AION_OP_SQUEEZE => blk: {
            try need(n, 1, spec.inputs);
            const axis: ?i32 = if (spec.attr.squeeze.has_axis != 0) spec.attr.squeeze.axis else null;
            break :blk try bld.squeeze(in(spec, 0), axis);
        },
        .AION_OP_UNSQUEEZE => blk: {
            try need(n, 1, spec.inputs);
            break :blk try bld.unsqueeze(in(spec, 0), spec.attr.unsqueeze.axis);
        },
        .AION_OP_TRANSPOSE2D => blk: {
            try need(n, 1, spec.inputs);
            break :blk try bld.transpose2d(in(spec, 0));
        },
        .AION_OP_SLICE => blk: {
            try need(n, 1, spec.inputs);
            if (spec.attr.slice.starts == null or spec.attr.slice.lens == null) return api.Builder.Error.InvalidArgument;
            const len = spec.attr.slice.len;
            const starts = spec.attr.slice.starts[0..len];
            const lens = spec.attr.slice.lens[0..len];
            const syms = spec.attr.slice.len_symbols;
            if (syms != null) {
                if (len > 8) return api.Builder.Error.InvalidArgument;
                var buf: [8]?[]const u8 = undefined;
                for (0..len) |i| buf[i] = cStrOrNull(syms[i]);
                break :blk try bld.sliceSym(in(spec, 0), starts, lens, buf[0..len]);
            }
            break :blk try bld.slice(in(spec, 0), starts, lens);
        },
        .AION_OP_LSTM_CELL => blk: {
            try need(n, 5, spec.inputs);
            const b_ih: ?api.TensorRef = if (n >= 7) in(spec, 5) else null;
            const b_hh: ?api.TensorRef = if (n >= 7) in(spec, 6) else null;
            break :blk try bld.lstmCell(in(spec, 0), in(spec, 1), in(spec, 2), in(spec, 3), in(spec, 4), b_ih, b_hh);
        },
        .AION_OP_RFFT => blk: {
            try need(n, 1, spec.inputs);
            break :blk try bld.rfft(in(spec, 0));
        },
        .AION_OP_STFT => blk: {
            try need(n, 2, spec.inputs);
            break :blk try bld.stft(in(spec, 0), in(spec, 1), spec.attr.stft.n_fft, spec.attr.stft.hop_length, spec.attr.stft.center != 0);
        },
        .AION_OP_CAST => blk: {
            try need(n, 1, spec.inputs);
            const to = dtypeFromC(spec.attr.cast.to_dtype) orelse return api.Builder.Error.InvalidArgument;
            break :blk try bld.cast(in(spec, 0), to);
        },
        .AION_OP_ARGMAX => blk: {
            try need(n, 1, spec.inputs);
            break :blk try bld.argmax(in(spec, 0), spec.attr.argmax.axis);
        },
        .AION_OP_SCATTER_ROW => blk: {
            try need(n, 3, spec.inputs);
            break :blk try bld.scatterRow(in(spec, 0), in(spec, 1), in(spec, 2));
        },
        .AION_OP_GATHER => blk: {
            try need(n, 2, spec.inputs);
            break :blk try bld.gather(in(spec, 0), in(spec, 1), spec.attr.gather.axis, spec.attr.gather.batch_dims);
        },
        .AION_OP_DIM => blk: {
            try need(n, 1, spec.inputs);
            break :blk try bld.dimSize(in(spec, 0), spec.attr.argmax.axis);
        },
        .AION_OP_IOTA => blk: {
            try need(n, 1, spec.inputs);
            break :blk try bld.iota(in(spec, 0), spec.attr.argmax.axis);
        },
    };
    return out.value;
}

pub export fn aion_builder_op(b_opt: ?*AionBuilder, spec_opt: ?*const AionOpSpec, out_value: ?*AionValueId) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    b.owner.clearLastError();
    if (out_value == null) return .AION_INVALID_ARGUMENT;
    const spec: *const AionOpSpec = spec_opt orelse return .AION_INVALID_ARGUMENT;
    const vid = builderOpImpl(b, spec) catch |e| {
        b.owner.setLastError("builder_op", e);
        return mapError(e);
    };
    out_value.?.* = vid;
    return .AION_OK;
}

/// Region id returned by `aion_builder_end_region`, consumed by if/loop.
pub const AionRegionId = u32;

pub export fn aion_builder_begin_region(b_opt: ?*AionBuilder) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    b.owner.clearLastError();
    b.builder.beginRegion() catch |e| {
        b.owner.setLastError("begin_region", e);
        return mapError(e);
    };
    return .AION_OK;
}

pub export fn aion_builder_end_region(
    b_opt: ?*AionBuilder,
    outputs_ptr: [*c]const AionValueId,
    outputs_len: usize,
    out_region: ?*AionRegionId,
) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    b.owner.clearLastError();
    if (out_region == null) return .AION_INVALID_ARGUMENT;
    if (outputs_len == 0 or outputs_len > 16 or outputs_ptr == null) return .AION_INVALID_ARGUMENT;

    var refs: [16]api.TensorRef = undefined;
    var i: usize = 0;
    while (i < outputs_len) : (i += 1) refs[i] = vref(outputs_ptr[i]);
    const rid = b.builder.endRegion(refs[0..outputs_len]) catch |e| {
        b.owner.setLastError("end_region", e);
        return mapError(e);
    };
    out_region.?.* = rid;
    return .AION_OK;
}

pub export fn aion_builder_if(
    b_opt: ?*AionBuilder,
    cond: AionValueId,
    then_region: AionRegionId,
    else_region: AionRegionId,
    out_value: ?*AionValueId,
) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    b.owner.clearLastError();
    if (out_value == null) return .AION_INVALID_ARGUMENT;
    const ref = b.builder.ifThenElse(vref(cond), then_region, else_region) catch |e| {
        b.owner.setLastError("builder_if", e);
        return mapError(e);
    };
    out_value.?.* = ref.value;
    return .AION_OK;
}

/// Loop. `out_values` must hold `carried_len` ids (the final carried values);
/// single-carry loops pass `carried_len == 1`. `has_cond_carry == 0` means no
/// continue-predicate carry.
pub export fn aion_builder_loop(
    b_opt: ?*AionBuilder,
    carried_ptr: [*c]const AionValueId,
    carried_len: usize,
    body_region: AionRegionId,
    trip: usize,
    cond_carry: usize,
    has_cond_carry: u8,
    check_before: u8,
    out_values: [*c]AionValueId,
) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    b.owner.clearLastError();
    if (carried_ptr == null or out_values == null) return .AION_INVALID_ARGUMENT;
    if (carried_len == 0 or carried_len > 16) return .AION_INVALID_ARGUMENT;

    var inits: [16]api.TensorRef = undefined;
    var outs: [16]api.TensorRef = undefined;
    var i: usize = 0;
    while (i < carried_len) : (i += 1) inits[i] = vref(carried_ptr[i]);

    const cc: ?usize = if (has_cond_carry != 0) cond_carry else null;
    b.builder.loopMulti(inits[0..carried_len], body_region, trip, cc, check_before != 0, outs[0..carried_len]) catch |e| {
        b.owner.setLastError("builder_loop", e);
        return mapError(e);
    };
    i = 0;
    while (i < carried_len) : (i += 1) out_values[i] = outs[i].value;
    return .AION_OK;
}

pub export fn aion_builder_mark_output(b_opt: ?*AionBuilder, value: AionValueId, name: ?[*:0]const u8) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    b.owner.clearLastError();
    if (name == null) return .AION_INVALID_ARGUMENT;
    // Attach the name so `compile` (which reads names off the builder) sees it,
    // and record a NamedTensorRef (reusing the builder's duped name) for export.
    _ = b.builder.name(vref(value), std.mem.span(name.?)) catch |e| {
        b.owner.setLastError("mark_output", e);
        return mapError(e);
    };
    const stored = b.builder.valueName(vref(value)) orelse {
        b.owner.setLastError("mark_output", error.InvalidArgument);
        return .AION_INTERNAL_ERROR;
    };
    b.outputs.append(b.alloc(), .{ .name = stored, .tensor = vref(value) }) catch {
        b.owner.setLastError("mark_output", error.OutOfMemory);
        return .AION_OUT_OF_MEMORY;
    };
    return .AION_OK;
}

/// Forget every output marked so far.
///
/// `mark_output` accumulates, so without this "compile exactly these outputs" is
/// not expressible: a second compile of the same builder carries the first one's
/// outputs too. Callers that pass an explicit output set clear first, which is
/// also what lets a single value be compiled and evaluated on a builder that is
/// still being authored.
pub export fn aion_builder_clear_outputs(b_opt: ?*AionBuilder) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    b.owner.clearLastError();
    b.outputs.clearRetainingCapacity();
    return .AION_OK;
}

pub export fn aion_builder_add_output_alias(b_opt: ?*AionBuilder, input_value: AionValueId, output_value: AionValueId) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    b.owner.clearLastError();
    b.aliases.append(b.alloc(), .{ .input = vref(input_value), .output = vref(output_value) }) catch {
        b.owner.setLastError("add_output_alias", error.OutOfMemory);
        return .AION_OUT_OF_MEMORY;
    };
    return .AION_OK;
}

pub export fn aion_builder_add_input_role(
    b_opt: ?*AionBuilder,
    value: AionValueId,
    kind: AionInputRoleKind,
    axis: i32,
    has_axis: u8,
    capacity_symbol: ?[*:0]const u8,
    zero_init: u8,
    allow_growable: u8,
    allow_ring: u8,
) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    b.owner.clearLastError();
    const sym: ?[]const u8 = if (capacity_symbol) |s| (b.dupeStr(std.mem.span(s)) catch {
        b.owner.setLastError("add_input_role", error.OutOfMemory);
        return .AION_OUT_OF_MEMORY;
    }) else null;
    b.roles.append(b.alloc(), .{
        .input = vref(value),
        .kind = roleKindFromC(kind),
        .axis = if (has_axis != 0) @as(u8, @intCast(axis)) else null,
        .capacity_symbol = sym,
        .zero_init = zero_init != 0,
        .allow_growable = allow_growable != 0,
        .allow_ring = allow_ring != 0,
    }) catch {
        b.owner.setLastError("add_input_role", error.OutOfMemory);
        return .AION_OUT_OF_MEMORY;
    };
    return .AION_OK;
}

pub export fn aion_builder_add_dim_symbol(b_opt: ?*AionBuilder, value: AionValueId, axis: usize, name: ?[*:0]const u8) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    b.owner.clearLastError();
    if (name == null) return .AION_INVALID_ARGUMENT;
    // The builder owns the declaration (and dupes the name into its graph arena).
    b.builder.symbolicDim(vref(value), axis, std.mem.span(name.?)) catch |e| {
        b.owner.setLastError("add_dim_symbol", e);
        return mapError(e);
    };
    return .AION_OK;
}

pub export fn aion_builder_add_metadata(b_opt: ?*AionBuilder, key: ?[*:0]const u8, value: ?[*:0]const u8) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    b.owner.clearLastError();
    if (key == null or value == null) return .AION_INVALID_ARGUMENT;
    const k = b.dupeStr(std.mem.span(key.?)) catch return .AION_OUT_OF_MEMORY;
    const v = b.dupeStr(std.mem.span(value.?)) catch return .AION_OUT_OF_MEMORY;
    b.metadata.append(b.alloc(), .{ .key = k, .value = v }) catch {
        b.owner.setLastError("add_metadata", error.OutOfMemory);
        return .AION_OUT_OF_MEMORY;
    };
    return .AION_OK;
}

/// Collect the accumulated bare output refs (for `compile`). Caller frees.
fn collectOutputRefs(b: *AionBuilder) ![]api.TensorRef {
    const refs = try b.alloc().alloc(api.TensorRef, b.outputs.items.len);
    for (b.outputs.items, 0..) |o, i| refs[i] = o.tensor;
    return refs;
}

pub export fn aion_builder_compile(
    b_opt: ?*AionBuilder,
    device_kind: AionDeviceKind,
    device_index: u32,
    out_model: ?*?*AionLoadedModel,
) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    const ctx = b.owner;
    ctx.clearLastError();
    if (out_model == null) return .AION_INVALID_ARGUMENT;
    out_model.?.* = null;
    if (b.outputs.items.len == 0) return .AION_INVALID_ARGUMENT;

    const refs = collectOutputRefs(b) catch {
        ctx.setLastError("builder_compile", error.OutOfMemory);
        return .AION_OUT_OF_MEMORY;
    };
    defer b.alloc().free(refs);

    // Symbolic dims are declared on the builder (`symbolicDim`) and read there.
    const opts: api.ExportModelOptions = .{
        .metadata = b.metadata.items,
        .output_aliases = b.aliases.items,
        .input_roles = b.roles.items,
    };
    const model = ctx.ctx.compileOn(deviceFromC(device_kind, device_index), &b.builder, refs, opts) catch |e| {
        ctx.setLastError("builder_compile", e);
        return mapError(e);
    };

    const handle: *AionLoadedModel = std.heap.page_allocator.create(AionLoadedModel) catch {
        ctx.setLastError("loaded_model_handle_alloc", error.OutOfMemory);
        return .AION_OUT_OF_MEMORY;
    };
    handle.* = .{ .owner = ctx, .model = model };
    out_model.?.* = handle;
    return .AION_OK;
}

fn builderExportImpl(b: *AionBuilder, path: ?[*:0]const u8, absolute: bool) AionStatus {
    const ctx = b.owner;
    ctx.clearLastError();
    if (path == null) return .AION_INVALID_ARGUMENT;
    if (b.outputs.items.len == 0) return .AION_INVALID_ARGUMENT;

    const opts: api.ExportModelOptions = .{
        .metadata = b.metadata.items,
        .output_aliases = b.aliases.items,
        .input_roles = b.roles.items,
    };
    const p: []const u8 = std.mem.span(path.?);
    (if (absolute)
        ctx.ctx.exportModelPathAbsolute(p, &b.builder, b.outputs.items, opts)
    else
        ctx.ctx.exportModelPath(p, &b.builder, b.outputs.items, opts)) catch |e| {
        ctx.setLastError("builder_export", e);
        return mapError(e);
    };
    return .AION_OK;
}

pub export fn aion_builder_export_path(b_opt: ?*AionBuilder, path: ?[*:0]const u8) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    return builderExportImpl(b, path, false);
}

pub export fn aion_builder_export_path_absolute(b_opt: ?*AionBuilder, path: ?[*:0]const u8) callconv(.c) AionStatus {
    const b: *AionBuilder = b_opt orelse return .AION_INVALID_ARGUMENT;
    return builderExportImpl(b, path, true);
}

/// Quantize row-major f32 values into a packed-quant tensor (q8_0 today).
pub export fn aion_tensor_quantize(
    ctx_opt: ?*AionContext,
    dtype_c: AionDType,
    rank: usize,
    shape_ptr: [*c]const usize,
    quant_axis: usize,
    values_ptr: [*c]const f32,
    values_len: usize,
    out_tensor: ?*?*AionTensor,
) callconv(.c) AionStatus {
    const ctx: *AionContext = ctx_opt orelse return .AION_INVALID_ARGUMENT;
    ctx.clearLastError();
    if (out_tensor == null) return .AION_INVALID_ARGUMENT;
    out_tensor.?.* = null;
    if (values_ptr == null) return .AION_INVALID_ARGUMENT;

    const shape: []const usize = getShapeSlice(rank, shape_ptr) orelse return .AION_INVALID_ARGUMENT;
    const dt: types.DType = dtypeFromC(dtype_c) orelse return .AION_INVALID_ARGUMENT;
    const values: []const f32 = values_ptr[0..values_len];

    const t: api.Tensor = ctx.ctx.fromF32Quantized(dt, shape, quant_axis, values) catch |e| {
        ctx.setLastError("tensor_quantize", e);
        return mapError(e);
    };
    const handle: *AionTensor = std.heap.page_allocator.create(AionTensor) catch {
        ctx.setLastError("tensor_handle_alloc", error.OutOfMemory);
        return .AION_OUT_OF_MEMORY;
    };
    handle.* = .{ .owner = ctx, .tensor = t };
    out_tensor.?.* = handle;
    return .AION_OK;
}

/// Create a quantized tensor directly from pre-packed quant bytes.
pub export fn aion_tensor_create_quant(
    ctx_opt: ?*AionContext,
    dtype_c: AionDType,
    rank: usize,
    shape_ptr: [*c]const usize,
    quant_axis: usize,
    packed_ptr: [*c]const u8,
    packed_len: usize,
    out_tensor: ?*?*AionTensor,
) callconv(.c) AionStatus {
    const ctx: *AionContext = ctx_opt orelse return .AION_INVALID_ARGUMENT;
    ctx.clearLastError();
    if (out_tensor == null) return .AION_INVALID_ARGUMENT;
    out_tensor.?.* = null;
    if (packed_ptr == null) return .AION_INVALID_ARGUMENT;

    const shape: []const usize = getShapeSlice(rank, shape_ptr) orelse return .AION_INVALID_ARGUMENT;
    const dt: types.DType = dtypeFromC(dtype_c) orelse return .AION_INVALID_ARGUMENT;
    if (!dt.info().is_quantized) return .AION_INVALID_ARGUMENT;
    const packed_bytes: []const u8 = packed_ptr[0..packed_len];

    const t: api.Tensor = ctx.ctx.fromPackedQuant(dt, shape, quant_axis, packed_bytes) catch |e| {
        ctx.setLastError("tensor_create_quant", e);
        return mapError(e);
    };
    const handle: *AionTensor = std.heap.page_allocator.create(AionTensor) catch {
        ctx.setLastError("tensor_handle_alloc", error.OutOfMemory);
        return .AION_OUT_OF_MEMORY;
    };
    handle.* = .{ .owner = ctx, .tensor = t };
    out_tensor.?.* = handle;
    return .AION_OK;
}

comptime {
    // Ensure these names stay in sync with the package errors we map.
    _ = pkg_types.PackageError;
}
