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
};

pub const AionContext = struct {
    allocator: std.mem.Allocator = std.heap.page_allocator,
    ctx: api.Context = undefined,

    last_err_buf: [512]u8 = .{0} ** 512,
    last_err_len: usize = 0,

    fn init(thread_count: usize) !AionContext {
        var out: AionContext = .{};
        out.ctx = try api.Context.initCpu(out.allocator, .{ .thread_count = thread_count });
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
    };
}

fn dtypeFromC(dt: AionDType) ?types.DType {
    return switch (dt) {
        .AION_DTYPE_F32 => .f32,
        .AION_DTYPE_F16 => .f16,
        .AION_DTYPE_I8 => .i8,
        .AION_DTYPE_Q4_0 => .q4_0,
        .AION_DTYPE_Q8_0 => .q8_0,
    };
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
    // Rank-0 tensors (scalars) are valid; they have an empty shape.
    if (rank == 0) return &[_]usize{};
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

    ctx_ptr.* = AionContext.init(thread_count) catch |e| {
        const st = mapError(e);
        // No context to store the error yet.
        return st;
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

        // v1: initialization from host values is supported only for f32.
        if (dt != .f32) return .AION_UNSUPPORTED;

        const raw_addr: usize = @intFromPtr(values_ptr.?);
        if ((raw_addr & (@alignOf(f32) - 1)) != 0) return .AION_INVALID_ARGUMENT;

        const f32_ptr: [*]const f32 = @ptrCast(@alignCast(values_ptr.?));
        const values: []const f32 = f32_ptr[0..values_len];

        break :blk ctx.ctx.fromF32(shape, values) catch |e| {
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

    switch (dt) {
        .f32 => {
            const raw_addr: usize = @intFromPtr(out_values.?);
            if ((raw_addr & (@alignOf(f32) - 1)) != 0) return .AION_INVALID_ARGUMENT;
            const f32_ptr: [*]f32 = @ptrCast(@alignCast(out_values.?));
            const dst: []f32 = f32_ptr[0..out_len];
            t.tensor.readF32(dst) catch |e| {
                ctx.setLastError("tensor_read", e);
                return mapError(e);
            };
            return .AION_OK;
        },
        else => return .AION_UNSUPPORTED,
    }
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

    switch (dt) {
        .f32 => {
            const raw_addr: usize = @intFromPtr(values_ptr.?);
            if ((raw_addr & (@alignOf(f32) - 1)) != 0) return .AION_INVALID_ARGUMENT;
            const f32_ptr: [*]const f32 = @ptrCast(@alignCast(values_ptr.?));
            const src: []const f32 = f32_ptr[0..values_len];
            t.tensor.writeF32(src) catch |e| {
                ctx.setLastError("tensor_write", e);
                return mapError(e);
            };
            return .AION_OK;
        },
        else => return .AION_UNSUPPORTED,
    }
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

    switch (dt) {
        .f32 => {
            const raw_addr: usize = @intFromPtr(out_value.?);
            if ((raw_addr & (@alignOf(f32) - 1)) != 0) return .AION_INVALID_ARGUMENT;

            const v: f32 = t.tensor.readScalar(f32) catch |e| {
                ctx.setLastError("tensor_read_scalar", e);
                return mapError(e);
            };
            const out_f32: *f32 = @ptrCast(@alignCast(out_value.?));
            out_f32.* = v;
            return .AION_OK;
        },
        else => return .AION_UNSUPPORTED,
    }
}

pub export fn aion_loaded_model_load_path(
    ctx_opt: ?*AionContext,
    path: ?[*:0]const u8,
    out_model: ?*?*AionLoadedModel,
) callconv(.c) AionStatus {
    const ctx: *AionContext = ctx_opt orelse return .AION_INVALID_ARGUMENT;
    ctx.clearLastError();

    if (out_model == null) return .AION_INVALID_ARGUMENT;
    out_model.?.* = null;
    if (path == null) return .AION_INVALID_ARGUMENT;

    const p: []const u8 = std.mem.span(path.?);
    const lm: api.LoadedModel = ctx.ctx.loadModelPath(p, .{}) catch |e| {
        ctx.setLastError("load_model_path", e);
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

pub export fn aion_loaded_model_load_path_absolute(
    ctx_opt: ?*AionContext,
    path: ?[*:0]const u8,
    out_model: ?*?*AionLoadedModel,
) callconv(.c) AionStatus {
    const ctx: *AionContext = ctx_opt orelse return .AION_INVALID_ARGUMENT;
    ctx.clearLastError();

    if (out_model == null) return .AION_INVALID_ARGUMENT;
    out_model.?.* = null;
    if (path == null) return .AION_INVALID_ARGUMENT;

    const p: []const u8 = std.mem.span(path.?);
    const lm: api.LoadedModel = ctx.ctx.loadModelPathAbsolute(p, .{}) catch |e| {
        ctx.setLastError("load_model_path_absolute", e);
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

comptime {
    // Ensure these names stay in sync with the package errors we map.
    _ = pkg_types.PackageError;
}
