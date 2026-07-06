// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
//! Shader-module + compute-pipeline cache for the GPU backend.
//!
//! This is the GPU analogue of the CPU backend's kernel registry: a `KernelDesc`
//! names a WGSL source blob (embedded at comptime), and `Pipelines` lazily builds
//! and caches the GPU objects derived from it — one `WGPUShaderModule` per kernel,
//! and a `{pipeline, bind-group-layout}` pair per (kernel, entry point). Entry
//! names may repeat across kernels (elementwise.wgsl and elementwise_i32.wgsl
//! both export `add`), so the pipeline cache keys on the composite.
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
    allocator: std.mem.Allocator,
    modules: std.StringHashMap(c.WGPUShaderModule), // key: kernel name
    entries: std.StringHashMap(Built), // key: "<kernel>\x00<entry>" (owned)

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator, gpu: *wgpu.Gpu) Self {
        return .{
            .gpu = gpu,
            .allocator = allocator,
            .modules = std.StringHashMap(c.WGPUShaderModule).init(allocator),
            .entries = std.StringHashMap(Built).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        var eit = self.entries.iterator();
        while (eit.next()) |e| {
            c.wgpuBindGroupLayoutRelease(e.value_ptr.bgl);
            c.wgpuComputePipelineRelease(e.value_ptr.pipeline);
            self.allocator.free(e.key_ptr.*);
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
        var key_buf: [128]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "{s}\x00{s}", .{ kernel.name, entry }) catch return error.ExecutionFailed;
        if (self.entries.get(key)) |b| return b;
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
        const owned_key = self.allocator.dupe(u8, key) catch {
            c.wgpuBindGroupLayoutRelease(bgl);
            c.wgpuComputePipelineRelease(pipeline);
            return error.ExecutionFailed;
        };
        self.entries.put(owned_key, built) catch {
            self.allocator.free(owned_key);
            c.wgpuBindGroupLayoutRelease(bgl);
            c.wgpuComputePipelineRelease(pipeline);
            return error.ExecutionFailed;
        };
        return built;
    }
};
