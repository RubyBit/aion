// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Shader-module + compute-pipeline cache for the GPU backend.
//!
//! This is the GPU analogue of the CPU backend's kernel registry: a `KernelDesc`
//! names a WGSL source blob (embedded at comptime), and `Pipelines` lazily builds
//! and caches the GPU objects derived from it — one `WGPUShaderModule` per kernel,
//! and a `{pipeline, bind-group-layout}` pair per entry point. Entry-point names
//! are unique across all kernels, so they key the pipeline cache directly.
//!
//! Pipelines are created with an auto-derived layout; we cache
//! `getBindGroupLayout(0)` alongside each pipeline so the layout is reflected
//! exactly once (at first use) rather than on every dispatch — the per-dispatch
//! reflection the v0 backend did.

const std = @import("std");
const wgpu = @import("wgpu.zig");

const c = wgpu.c;

/// A WGSL kernel: a name (cache key for its module) and its embedded source.
pub const KernelDesc = struct {
    name: []const u8,
    wgsl: []const u8,
};

/// A built pipeline plus its (cached) bind-group layout.
pub const Built = struct {
    pipeline: c.WGPUComputePipeline,
    bgl: c.WGPUBindGroupLayout,
};

pub const Pipelines = struct {
    gpu: *wgpu.Gpu,
    modules: std.StringHashMap(c.WGPUShaderModule), // key: kernel name
    entries: std.StringHashMap(Built), // key: entry-point name (globally unique)

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, gpu: *wgpu.Gpu) Self {
        return .{
            .gpu = gpu,
            .modules = std.StringHashMap(c.WGPUShaderModule).init(allocator),
            .entries = std.StringHashMap(Built).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        var eit = self.entries.valueIterator();
        while (eit.next()) |b| {
            c.wgpuBindGroupLayoutRelease(b.bgl);
            c.wgpuComputePipelineRelease(b.pipeline);
        }
        self.entries.deinit();
        var mit = self.modules.valueIterator();
        while (mit.next()) |m| c.wgpuShaderModuleRelease(m.*);
        self.modules.deinit();
        self.* = undefined;
    }

    fn moduleFor(self: *Self, kernel: KernelDesc) error{ExecutionFailed}!c.WGPUShaderModule {
        if (self.modules.get(kernel.name)) |m| return m;
        var w: c.WGPUShaderSourceWGSL = std.mem.zeroes(c.WGPUShaderSourceWGSL);
        w.chain.sType = c.WGPUSType_ShaderSourceWGSL;
        w.code = wgpu.strv(kernel.wgsl);
        var smd: c.WGPUShaderModuleDescriptor = std.mem.zeroes(c.WGPUShaderModuleDescriptor);
        smd.nextInChain = &w.chain;
        const m = c.wgpuDeviceCreateShaderModule(self.gpu.device, &smd) orelse return error.ExecutionFailed;
        self.modules.put(kernel.name, m) catch return error.ExecutionFailed;
        return m;
    }

    /// Get-or-build the pipeline + bind-group layout for `entry` in `kernel`.
    pub fn get(self: *Self, kernel: KernelDesc, entry: [:0]const u8) error{ExecutionFailed}!Built {
        if (self.entries.get(entry)) |b| return b;
        const module = try self.moduleFor(kernel);

        var cpd: c.WGPUComputePipelineDescriptor = std.mem.zeroes(c.WGPUComputePipelineDescriptor);
        cpd.compute.module = module;
        cpd.compute.entryPoint = wgpu.strv(entry);
        const pipeline = c.wgpuDeviceCreateComputePipeline(self.gpu.device, &cpd) orelse return error.ExecutionFailed;

        // Reflect the auto-derived layout once and cache it.
        const bgl = c.wgpuComputePipelineGetBindGroupLayout(pipeline, 0) orelse {
            c.wgpuComputePipelineRelease(pipeline);
            return error.ExecutionFailed;
        };
        const built: Built = .{ .pipeline = pipeline, .bgl = bgl };
        self.entries.put(entry, built) catch {
            c.wgpuBindGroupLayoutRelease(bgl);
            c.wgpuComputePipelineRelease(pipeline);
            return error.ExecutionFailed;
        };
        return built;
    }
};
