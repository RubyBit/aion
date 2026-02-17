const std = @import("std");

const backend_mod = @import("../backend/backend.zig");
const cpu_backend_mod = @import("../backend/cpu/cpu_backend.zig");
const types = @import("../backend/types.zig");
const manager_mod = @import("../storage/manager.zig");
const plan_mod = @import("../graph/plan.zig");
const graph_mod = @import("../graph/graph.zig");
const program_mod = @import("../graph/program.zig");

const api_builder = @import("builder.zig");
const api_model = @import("model.zig");
const api_tensor = @import("tensor.zig");
const api_tiling = @import("tiling.zig");
const api_errors = @import("errors.zig");

pub const DType = types.DType;
pub const TilePolicy = plan_mod.TilePolicy;

pub const Context = struct {
    allocator: std.mem.Allocator,

    /// v0: CPU backend only (other backends can be added later).
    cpu: cpu_backend_mod.CpuBackend,

    store: manager_mod.StorageManager,
    policy: plan_mod.TilePolicy,

    const Self = @This();

    pub const Options = struct {
        /// Total threads including the calling thread.
        thread_count: usize = 1,

        /// Optional tiling policy override.
        ///
        /// If null, a default policy is used and users don't need to think about tiling.
        /// Power users can provide this to tune performance.
        tile_policy_override: ?plan_mod.TilePolicy = null,
    };

    pub fn initCpu(allocator: std.mem.Allocator, opts: Options) api_errors.InitError!Self {
        var cpu: cpu_backend_mod.CpuBackend = if (opts.thread_count <= 1)
            cpu_backend_mod.CpuBackend.init(allocator)
        else
            try cpu_backend_mod.CpuBackend.initWithOptions(allocator, .{ .thread_count = opts.thread_count });
        errdefer cpu.deinit();

        var sm: manager_mod.StorageManager = manager_mod.StorageManager.init(allocator);
        errdefer sm.deinit();

        const out: Self = .{
            .allocator = allocator,
            .cpu = cpu,
            .store = sm,
            .policy = opts.tile_policy_override orelse plan_mod.TilePolicy{},
        };
        return out;
    }

    pub fn deinit(self: *Self) void {
        self.store.deinit();
        self.cpu.deinit();
        self.* = undefined;
    }

    pub fn storage(self: *Self) *manager_mod.StorageManager {
        return &self.store;
    }

    pub fn tilePolicy(self: *const Self) plan_mod.TilePolicy {
        return self.policy;
    }

    /// Return a backend handle bound to this context's CPU backend.
    ///
    /// Important: this is computed on-demand to avoid stale internal pointers
    /// if a `Context` value is moved.
    pub fn backend(self: *Self) backend_mod.Backend {
        return self.cpu.backend();
    }

    pub fn builder(self: *const Self) api_builder.Builder {
        return api_builder.Builder.init(self.allocator);
    }

    /// Create a new owned tensor with a default tile shape.
    pub fn tensor(self: *Self, dtype: DType, shape: []const usize) api_errors.ApiError!api_tensor.Tensor {
        if (shape.len == 0 or shape.len > api_tiling.MAX_RANK) return api_errors.ApiError.InvalidArgument;

        var tile_mem: [api_tiling.MAX_RANK]usize = undefined;
        const tile_slice: []usize = tile_mem[0..shape.len];
        try api_tiling.fillDefaultTileShape(self.policy, dtype, shape, tile_slice);

        const tid: manager_mod.TensorId = try self.store.createTiledTensor(dtype, shape, tile_slice, .{ .tile_alignment = self.policy.tile_alignment });
        const t = try self.store.getConst(tid);
        return .{ .store = &self.store, .id = tid, .dtype = t.dtype, .shape = t.shape };
    }

    /// Create a new tensor with an explicit tile shape.
    pub fn tensorTiled(self: *Self, dtype: DType, shape: []const usize, tile_shape: []const usize) api_errors.ApiError!api_tensor.Tensor {
        const tid: manager_mod.TensorId = try self.store.createTiledTensor(dtype, shape, tile_shape, .{ .tile_alignment = self.policy.tile_alignment });
        const t = try self.store.getConst(tid);
        return .{ .store = &self.store, .id = tid, .dtype = t.dtype, .shape = t.shape };
    }

    /// Convenience: allocate and initialize from packed scalar bytes.
    pub fn fromPackedScalar(self: *Self, dtype: DType, shape: []const usize, packed_bytes: []const u8) api_errors.ApiError!api_tensor.Tensor {
        var t: api_tensor.Tensor = try self.tensor(dtype, shape);
        try t.writePackedScalar(packed_bytes);
        return t;
    }

    /// Convenience: allocate and initialize from packed quant bytes.
    pub fn fromPackedQuant(self: *Self, dtype: DType, shape: []const usize, packed_bytes: []const u8) api_errors.ApiError!api_tensor.Tensor {
        // For quant tensors, avoid retile surprises: pick a tile shape that respects quant block alignment.
        if (!dtype.info().is_quantized) return api_errors.ApiError.InvalidArgument;

        if (shape.len == 2) {
            const k: usize = shape[0];
            const n: usize = shape[1];
            const tn: [2]usize = api_tiling.chooseQuantMatMulBTiles(self.policy, k, n, dtype);
            var t: api_tensor.Tensor = try self.tensorTiled(dtype, shape, tn[0..2]);
            try t.writePackedQuant(packed_bytes);
            return t;
        }

        // Fallback for non-2D quant tensors.
        var t0: api_tensor.Tensor = try self.tensor(dtype, shape);
        try t0.writePackedQuant(packed_bytes);
        return t0;
    }

    /// Convenience: allocate and initialize an f32 tensor from typed values.
    pub fn fromF32(self: *Self, shape: []const usize, values: []const f32) api_errors.ApiError!api_tensor.Tensor {
        return self.from(shape, values);
    }

    /// Convenience: allocate and initialize an f16 tensor from typed values.
    pub fn fromF16(self: *Self, shape: []const usize, values: []const f16) api_errors.ApiError!api_tensor.Tensor {
        return self.from(shape, values);
    }

    /// Convenience: allocate and initialize a scalar tensor (shape `{1}`).
    pub fn scalar(self: *Self, comptime T: type, value: T) api_errors.ApiError!api_tensor.Tensor {
        const shape: [1]usize = .{1};
        var tmp: [1]T = .{value};
        return self.from(shape[0..1], tmp[0..1]);
    }

    fn elemTypeOfValues(comptime V: type) ?type {
        // []T / []const T
        if (api_tensor.Tensor.sliceElemType(V)) |Elem| return Elem;

        // *[N]T
        const pinfo = switch (@typeInfo(V)) {
            .pointer => |p| p,
            else => return null,
        };
        if (pinfo.size != .one) return null;
        return switch (@typeInfo(pinfo.child)) {
            .array => |a| a.child,
            else => null,
        };
    }

    fn valuesLen(values: anytype) ?usize {
        const V: type = @TypeOf(values);
        return switch (@typeInfo(V)) {
            .pointer => |p| switch (p.size) {
                .slice => values.len,
                .one => switch (@typeInfo(p.child)) {
                    .array => |a| @as(usize, @intCast(a.len)),
                    else => null,
                },
                else => null,
            },
            .array => |a| @as(usize, @intCast(a.len)),
            else => null,
        };
    }

    /// Convenience: allocate a 1D tensor whose length is inferred from `values`.
    pub fn vector(self: *Self, values: anytype) api_errors.ApiError!api_tensor.Tensor {
        const n_opt: ?usize = valuesLen(values);
        if (n_opt == null) return api_errors.ApiError.InvalidArgument;
        const n: usize = n_opt.?;
        const shape: [1]usize = .{n};
        return self.from(shape[0..1], values);
    }

    /// Convenience: allocate a 2D tensor and fill it from a flat value buffer.
    pub fn matrix(self: *Self, rows: usize, cols: usize, values: anytype) api_errors.ApiError!api_tensor.Tensor {
        const shape: [2]usize = .{ rows, cols };
        return self.from(shape[0..2], values);
    }

    /// Convenience: allocate and initialize a tensor from typed values.
    ///
    /// The tensor dtype is inferred from the element type of `values`.
    pub fn from(self: *Self, shape: []const usize, values: anytype) api_errors.ApiError!api_tensor.Tensor {
        const ElemOpt: ?type = elemTypeOfValues(@TypeOf(values));
        if (ElemOpt == null) return api_errors.ApiError.InvalidArgument;
        const Elem: type = ElemOpt.?;
        const dt_opt: ?DType = api_tensor.Tensor.dtypeOf(Elem);
        if (dt_opt == null) return api_errors.ApiError.InvalidArgument;
        const dt: DType = dt_opt.?;

        var t: api_tensor.Tensor = try self.tensor(dt, shape);
        try t.write(values);
        return t;
    }

    /// Convenience: allocate and initialize a tensor from a (possibly nested) Zig array.
    ///
    /// Example:
    /// - `ctx.fromArray([2][3]f32{ ... })` infers shape `{2,3}` and dtype `f32`.
    ///
    /// This uses packed scalar I/O (no intermediate flattening allocations).
    pub fn fromArray(self: *Self, arr: anytype) api_errors.ApiError!api_tensor.Tensor {
        const ArrT: type = @TypeOf(arr);

        const base_arr_t: type = switch (@typeInfo(ArrT)) {
            .array => ArrT,
            .pointer => |p| switch (p.size) {
                .one => p.child,
                else => return api_errors.ApiError.InvalidArgument,
            },
            else => return api_errors.ApiError.InvalidArgument,
        };

        switch (@typeInfo(base_arr_t)) {
            .array => {},
            else => return api_errors.ApiError.InvalidArgument,
        }

        const ShapeInfo = comptime blk: {
            var dims: [api_tiling.MAX_RANK]usize = .{0} ** api_tiling.MAX_RANK;
            var rank: usize = 0;
            var cur: type = base_arr_t;
            while (true) {
                switch (@typeInfo(cur)) {
                    .array => |a| {
                        if (rank >= api_tiling.MAX_RANK) break;
                        dims[rank] = @as(usize, @intCast(a.len));
                        rank += 1;
                        cur = a.child;
                    },
                    else => break,
                }
            }
            const dt_opt: ?DType = api_tensor.Tensor.dtypeOf(cur);
            const dims_out: [api_tiling.MAX_RANK]usize = dims;
            break :blk .{ .rank = rank, .dims = dims_out, .dt_opt = dt_opt };
        };

        if (ShapeInfo.rank == 0 or ShapeInfo.rank > api_tiling.MAX_RANK) return api_errors.ApiError.InvalidArgument;
        if (ShapeInfo.dt_opt == null) return api_errors.ApiError.InvalidArgument;
        const dt: DType = ShapeInfo.dt_opt.?;

        var shape_mem: [api_tiling.MAX_RANK]usize = undefined;
        var i: usize = 0;
        while (i < ShapeInfo.rank) : (i += 1) {
            shape_mem[i] = ShapeInfo.dims[i];
        }
        const shape: []const usize = shape_mem[0..ShapeInfo.rank];

        const packed_bytes: []const u8 = switch (@typeInfo(ArrT)) {
            .array => std.mem.asBytes(&arr),
            .pointer => std.mem.asBytes(arr),
            else => unreachable,
        };

        return self.fromPackedScalar(dt, shape, packed_bytes);
    }

    /// Compile a builder into a runnable model.
    pub fn compile(self: *Self, b: *api_builder.Builder, outputs: []const api_builder.TensorRef) api_errors.ApiError!api_model.Model {
        if (outputs.len == 0) return api_errors.ApiError.InvalidArgument;

        const g: *graph_mod.Graph = b.innerGraph();

        // Set graph outputs.
        const tmp: []graph_mod.ValueId = try self.allocator.alloc(graph_mod.ValueId, outputs.len);
        defer self.allocator.free(tmp);
        for (outputs, 0..) |o, i| tmp[i] = o.value;
        try g.setOutputs(tmp);

        var prog = try program_mod.compileGraph(self.allocator, g, &self.store, self.policy);
        errdefer prog.deinit();

        return .{ .allocator = self.allocator, .ctx = self, .program = prog };
    }
};
