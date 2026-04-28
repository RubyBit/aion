// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
const std = @import("std");

const executable_mod = @import("../runtime/executable.zig");
const manager_mod = @import("../storage/manager.zig");

const api_tensor = @import("tensor.zig");
const api_errors = @import("errors.zig");
const api_context = @import("context.zig");

pub const ExecuteError = api_errors.ExecuteError;

pub const Model = struct {
    allocator: std.mem.Allocator,

    ctx: *api_context.Context,

    program: executable_mod.ExecutableProgram,

    const Self = @This();

    pub fn deinit(self: *Self) void {
        self.program.deinit();
        self.* = undefined;
    }

    pub fn run(self: *Self) ExecuteError!void {
        const be = self.ctx.backend();
        return be.executeProgram(&self.program, self.ctx.store.tensorStore());
    }

    pub fn outputCount(self: *const Self) usize {
        return self.program.outputs.len;
    }

    pub fn outputTensor(self: *Self, index: usize) api_tensor.Tensor {
        const tid: manager_mod.TensorId = self.program.outputs[index];
        const t = self.ctx.store.getConst(tid) catch unreachable;
        return .{ .store = &self.ctx.store, .id = tid, .dtype = t.dtype, .shape = t.shape };
    }

    /// Execute the model and return output `index` as a Tensor handle.
    ///
    /// This keeps the API "tensor-first"; callers can then read scalar/slices via `Tensor` helpers.
    pub fn runOutputTensor(self: *Self, index: usize) ExecuteError!api_tensor.Tensor {
        try self.run();
        return self.outputTensor(index);
    }

    /// Execute the model and return output `index` as a newly allocated f32 slice.
    /// Caller owns the returned memory.
    pub fn runOutputF32Alloc(self: *Self, allocator: std.mem.Allocator, index: usize) ExecuteError![]f32 {
        try self.run();
        const t: api_tensor.Tensor = self.outputTensor(index);
        return t.readF32Alloc(allocator);
    }

    /// Execute the model and return output `index` as a newly allocated f16 slice.
    /// Caller owns the returned memory.
    pub fn runOutputF16Alloc(self: *Self, allocator: std.mem.Allocator, index: usize) ExecuteError![]f16 {
        try self.run();
        const t: api_tensor.Tensor = self.outputTensor(index);
        return t.readF16Alloc(allocator);
    }

    /// Execute the model and return scalar output `index` as f32.
    /// Requires output tensor to be dtype f32 and contain exactly 1 element.
    pub fn runScalarF32(self: *Self, index: usize) ExecuteError!f32 {
        try self.run();
        const t: api_tensor.Tensor = self.outputTensor(index);
        const n: usize = try t.elemCount();
        if (n != 1) return api_errors.ExecuteError.InvalidArgument;

        var tmp: [1]f32 = .{0.0};
        try t.readF32(tmp[0..1]);
        return tmp[0];
    }
};
