// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later

const std = @import("std");

// Although this function looks imperative, it does not perform the build
// directly and instead it mutates the build graph (`b`) that will be then
// executed by an external runner. The functions in `std.Build` implement a DSL
// for defining build steps and express dependencies between them, allowing the
// build runner to parallelize the build automatically (and the cache system to
// know when a step doesn't need to be re-run).
pub fn build(b: *std.Build) void {
    // Standard target options allow the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // target.query.cpu_features_add.addFeature(@intFromEnum(std.Target.x86.Feature.avxvnni));
    // // add also avxvnniint8
    // target.query.cpu_features_add.addFeature(@intFromEnum(std.Target.x86.Feature.avxvnniint8));
    // // also add avxvnniint16
    // target.query.cpu_features_add.addFeature(@intFromEnum(std.Target.x86.Feature.avxvnniint16));

    // // Print default features
    // {
    //     // Re-resolve target to see updated features in result if query was modified
    //     const resolved_target = std.Build.resolveTargetQuery(b, target.query);
    //     const cpu = resolved_target.result.cpu;
    //     std.debug.print("Target: {s}-{s}-{s}\n", .{ @tagName(cpu.arch), @tagName(resolved_target.result.os.tag), cpu.model.name });
    //     std.debug.print("Features: ", .{});
    //     var first = true;

    //     switch (cpu.arch) {
    //         .x86_64 => {
    //             for (std.Target.x86.all_features, 0..) |feature, i| {
    //                 if (cpu.features.isEnabled(@intCast(i))) {
    //                     if (!first) std.debug.print(", ", .{});
    //                     std.debug.print("{s}", .{feature.name});
    //                     first = false;
    //                 }
    //             }
    //         },
    //         else => std.debug.print("(unsupported architecture for feature listing)", .{}),
    //     }
    //     std.debug.print("\n", .{});
    // }

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});
    // It's also possible to define more custom flags to toggle optional features
    // of this build script using `b.option()`. All defined flags (including
    // target and optimize options) will be listed when running `zig build --help`
    // in this directory.

    // Portable multi-ISA CPU kernels: when enabled (the default), the public
    // `aion` module and shipped library are compiled at the x86_64_v3 floor
    // (AVX2 + FMA) and carry per-ISA kernel objects (v3/v3_vnni/v4). Runtime
    // CPUID dispatch selects among them, and any Zig artifact that imports the
    // module links the tier objects transitively.
    // Pass `-Dmultiversion=false` for a plain native library (fast dev iteration).
    const multiversion: bool = b.option(
        bool,
        "multiversion",
        "Build portable multi-ISA CPU kernel objects with runtime dispatch (x86_64)",
    ) orelse true;

    const want_x86_multiversion: bool = multiversion and target.result.cpu.arch == .x86_64;
    // aarch64: NEON f32 is fixed 4-wide, so the non-registry kernels don't vary by
    // width — multiversioning here is purely the int8 path: baseline (f32 accumulate),
    // FEAT_DotProd (`sdot`), and FEAT_I8MM (`smmla` matrix-multiply).
    const want_arm_multiversion: bool = multiversion and target.result.cpu.arch == .aarch64;
    const want_multiversion: bool = want_x86_multiversion or want_arm_multiversion;

    // Portable-distribution floor for the main module. Only the registry kernels
    // (matmul/conv/quant) are dispatched to the tier objects; every non-registry
    // kernel (elementwise, activations, softmax, norms, RoPE, LSTM, the
    // simd.lanesF32 family) runs at the main module's ISA. The x86 multiversion
    // main module is therefore ALWAYS floored to x86_64_v3 (AVX2 + FMA): the
    // shipped artifact is a reproducible AVX2-class binary regardless of the build
    // machine's CPU, so a wheel built on an AVX-512 box never SIGILLs elsewhere.
    //
    // v3 (Haswell 2013 / AMD 2015+) is the modern portability floor. We do NOT
    // fall back to v2 (SSE-only, no FMA): v2 de-vectorizes the FMA-heavy
    // activation/norm kernels and regresses badly (~7x slower silero VAD), and a
    // genuine v2-only CPU is pre-2013 and out of support. The registry kernels
    // still dispatch up to v3_vnni / v4 (AVX-512/VNNI) via the tiers.
    //
    // For a plain native build (no floor, no dispatch — fastest local iteration),
    // pass `-Dmultiversion=false`.
    const main_target: std.Build.ResolvedTarget = if (want_x86_multiversion) blk: {
        var q = target.query;
        q.cpu_model = .{ .explicit = &std.Target.x86.cpu.x86_64_v3 };
        q.cpu_features_add = std.Target.Cpu.Feature.Set.empty;
        q.cpu_features_sub = std.Target.Cpu.Feature.Set.empty;
        break :blk b.resolveTargetQuery(q);
    } else target;

    const linkage = b.option(
        std.builtin.LinkMode,
        "linkage",
        "Library linkage: .static (default) or .dynamic",
    ) orelse .static;

    // When statically linking Aion into a shared object (e.g. Python extension
    // modules on Linux), the static library objects must be position-independent.
    //
    // Enable with: `zig build install -Dpic=true`.
    const pic: bool = b.option(
        bool,
        "pic",
        "Build the library with position-independent code (useful for FFI)",
    ) orelse false;

    // Feature-gate the WebGPU GPU backend (Rust `#[cfg(feature)]` equivalent).
    // Default ON so a plain `zig build` (incl. zls's module-resolution runner)
    // compiles the in-tree gpu code; `-Dgpu=false` excludes it entirely (no wgpu
    // fetch/link). Drives `build_options.enable_gpu`, read by root.zig.
    const want_gpu = b.option(bool, "gpu", "Build the WebGPU (wgpu-native) backend (default true; -Dgpu=false to skip)") orelse true;
    // Resolve the prebuilt wgpu package up front. Null when disabled, on an
    // unsupported target, or on the lazy-fetch's first pass (build re-runs once
    // fetched). `enable_gpu` follows this so the in-tree gpu code is only enabled
    // once its `wgpu` import is actually wired - never half-configured.
    const wgpu_dep: ?WgpuDep = if (want_gpu) wgpuDependency(b, target) else null;
    const enable_gpu = wgpu_dep != null;

    // Expose a public Zig module for consumers (and our examples). With
    // multiversioning enabled this is the same portable-dispatch module shape as
    // the installed library: v3 floor on x86_64 plus transitive tier objects.
    // In a dependent package's build.zig:
    //   const aion_dep = b.dependency("aion", .{});
    //   exe.root_module.addImport("aion", aion_dep.module("aion"));
    const aion_mod = b.addModule("aion", .{
        .root_source_file = b.path("src/root.zig"),
        .target = main_target,
        .optimize = optimize,
        .imports = &.{},
    });
    const aion_build_options = b.addOptions();
    aion_build_options.addOption(bool, "multiversion", want_multiversion);
    aion_build_options.addOption(bool, "enable_gpu", enable_gpu);
    aion_mod.addImport("build_options", aion_build_options.createModule());

    // When the GPU feature is on, the `aion` module carries both the translate-c'd
    // `wgpu` bindings and the native wgpu link inputs. Importers inherit those
    // link inputs transitively; `-Dgpu=false` keeps consumers CPU-only.
    if (wgpu_dep) |wd| {
        const translate = b.addTranslateC(.{
            .root_source_file = wd.dep.path("include/webgpu/wgpu.h"),
            .target = main_target,
            .optimize = optimize,
        });
        translate.addIncludePath(wd.dep.path("include/webgpu"));
        aion_mod.addImport("wgpu", translate.createModule());
        wd.link(aion_mod);
    }

    // Portable multiversioning: compile `kernels_export.zig` once per ISA tier.
    // Each object is built at its own CPU model/feature set, so its `@Vector`
    // kernels lower to that ISA (e.g. the v4 object emits real AVX-512). The
    // objects share no global symbols except their uniquely-named accessor, so the
    // per-tier kernel copies don't collide. Built once and attached to the public
    // module plus the shipped C/FFI library.
    var tier_objs: [3]*std.Build.Step.Compile = undefined;
    var n_tiers: usize = 0;

    // Builds one tier object from `tier_kernels_root.zig` at `tier_target`, with
    // the given lane width and quant encoding, and records it in `tier_objs`.
    const TierBuild = struct {
        fn add(
            bld: *std.Build,
            objs: *[3]*std.Build.Step.Compile,
            count: *usize,
            opt: std.builtin.OptimizeMode,
            want_pic: bool,
            name: []const u8,
            lanes: u32,
            quant_enc: u8,
            tier_target: std.Build.ResolvedTarget,
        ) void {
            const tier_opts = bld.addOptions();
            tier_opts.addOption(u32, "lanes", lanes);
            tier_opts.addOption([]const u8, "tier_name", name);
            tier_opts.addOption(u8, "quant_enc", quant_enc);

            const tier_mod = bld.createModule(.{
                .root_source_file = bld.path("src/tier_kernels_root.zig"),
                .target = tier_target,
                .optimize = opt,
            });
            tier_mod.addOptions("tier_options", tier_opts);

            const tier_obj = bld.addObject(.{
                .name = bld.fmt("aion_kernels_{s}", .{name}),
                .root_module = tier_mod,
            });
            if (want_pic) tier_obj.root_module.pic = true;

            objs[count.*] = tier_obj;
            count.* += 1;
        }
    };

    if (want_x86_multiversion) {
        // quant_enc: 0 = f32-accumulate, 1 = AVX-VNNI (VEX vpdpbusd), 2 = AVX-512-VNNI (EVEX).
        const X86Tier = struct { name: []const u8, lanes: u32, model: *const std.Target.Cpu.Model, add: []const std.Target.x86.Feature, quant_enc: u8 };
        const tiers = [_]X86Tier{
            .{ .name = "v3", .lanes = 8, .model = &std.Target.x86.cpu.x86_64_v3, .add = &.{}, .quant_enc = 0 },
            .{ .name = "v3_vnni", .lanes = 8, .model = &std.Target.x86.cpu.x86_64_v3, .add = &.{.avxvnni}, .quant_enc = 1 },
            .{ .name = "v4", .lanes = 16, .model = &std.Target.x86.cpu.x86_64_v4, .add = &.{.avx512vnni}, .quant_enc = 2 },
        };
        for (tiers) |tier| {
            var q = target.query;
            q.cpu_model = .{ .explicit = tier.model };
            q.cpu_features_add = std.Target.Cpu.Feature.Set.empty;
            q.cpu_features_sub = std.Target.Cpu.Feature.Set.empty;
            for (tier.add) |f| q.cpu_features_add.addFeature(@intFromEnum(f));
            TierBuild.add(b, &tier_objs, &n_tiers, optimize, pic, tier.name, tier.lanes, tier.quant_enc, b.resolveTargetQuery(q));
        }
    } else if (want_arm_multiversion) {
        // NEON is always 4-wide for f32; the only ISA differentiator is the int8
        // path. quant_enc: 0 = f32-accumulate, 3 = FEAT_DotProd (sdot, grouped-by-4
        // dot), 4 = FEAT_I8MM (smmla, int8 2×2 matrix-multiply — ~2x sdot on prefill).
        const ArmTier = struct { name: []const u8, add: []const std.Target.aarch64.Feature, quant_enc: u8 };
        const tiers = [_]ArmTier{
            .{ .name = "arm_baseline", .add = &.{}, .quant_enc = 0 },
            .{ .name = "arm_dotprod", .add = &.{.dotprod}, .quant_enc = 3 },
            .{ .name = "arm_i8mm", .add = &.{ .dotprod, .i8mm }, .quant_enc = 4 },
        };
        for (tiers) |tier| {
            var q = target.query;
            q.cpu_model = .baseline;
            q.cpu_features_add = std.Target.Cpu.Feature.Set.empty;
            q.cpu_features_sub = std.Target.Cpu.Feature.Set.empty;
            for (tier.add) |f| q.cpu_features_add.addFeature(@intFromEnum(f));
            TierBuild.add(b, &tier_objs, &n_tiers, optimize, pic, tier.name, 4, tier.quant_enc, b.resolveTargetQuery(q));
        }
    }

    for (tier_objs[0..n_tiers]) |o| aion_mod.addObject(o);

    // Shipped library: main module floored to x86_64_v3, dispatch on, tier
    // objects linked. This is the artifact installed to `zig-out/lib`.
    const lib_mod = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = main_target,
        .optimize = optimize,
    });
    {
        const bo = b.addOptions();
        bo.addOption(bool, "multiversion", want_multiversion);
        // The shipped C-ABI library stays CPU-only (no wgpu); the GPU backend is
        // for Zig consumers via the `aion` module.
        bo.addOption(bool, "enable_gpu", false);
        lib_mod.addOptions("build_options", bo);
    }

    const lib = b.addLibrary(.{
        .name = "aion",
        .root_module = lib_mod,
        .linkage = linkage,
    });

    // When a C/FFI consumer links against the static library on Windows using
    // MSVC's linker (e.g. Python extensions via setuptools), the final link
    // step will not automatically pull in Zig/Clang builtins (compiler-rt).
    // Bundle compiler-rt into the archive so consumers don't need to know
    // about extra runtime libraries.
    if (linkage == .static and target.result.os.tag == .windows) {
        lib.bundle_compiler_rt = true;
    }

    if (pic) {
        lib_mod.pic = true;
    }

    for (tier_objs[0..n_tiers]) |o| lib_mod.addObject(o);

    b.installArtifact(lib);

    // Install public C header for FFI consumers.
    const install_header = b.addInstallFile(b.path("include/aion.h"), "include/aion.h");
    b.getInstallStep().dependOn(&install_header.step);

    // Unit tests.
    // On this Zig snapshot, running the test artifact through Build's special
    // `--listen=-` test runner mode can stall on Windows. The direct `zig test`
    // path is the closest stable fallback and still honors the selected
    // target/optimize settings.
    const skip_thread_pool_tests: bool = b.option(
        bool,
        "skip-thread-pool-tests",
        "Skip thread pool tests (avoids occasional Windows stalls)",
    ) orelse false;

    const test_step = b.step("test", "Run tests");
    const test_build_options = b.addOptions();
    test_build_options.addOption(bool, "multiversion", false);
    test_build_options.addOption(bool, "enable_gpu", false);

    const run_lib_tests = b.addSystemCommand(&.{
        b.graph.zig_exe,
        "test",
        "--cache-dir",
        ".zig-cache",
        optimizeArg(optimize),
    });
    run_lib_tests.has_side_effects = true;
    if (!target.query.isNativeTriple()) {
        run_lib_tests.addArgs(&.{
            "-target",
            target.query.zigTriple(b.allocator) catch @panic("OOM"),
        });
    }
    if (b.args) |args| run_lib_tests.addArgs(args);
    addRawTestModules(b, run_lib_tests, test_build_options);
    if (skip_thread_pool_tests) run_lib_tests.setEnvironmentVariable("AION_SKIP_THREAD_POOL_TESTS", "1");
    test_step.dependOn(&run_lib_tests.step);

    const test_fast_step = b.step("test-fast", "Run tests (skip thread pool suite)");
    const run_fast_tests = b.addSystemCommand(&.{
        b.graph.zig_exe,
        "test",
        "--cache-dir",
        ".zig-cache",
        optimizeArg(optimize),
    });
    run_fast_tests.has_side_effects = true;
    if (!target.query.isNativeTriple()) {
        run_fast_tests.addArgs(&.{
            "-target",
            target.query.zigTriple(b.allocator) catch @panic("OOM"),
        });
    }
    if (b.args) |args| run_fast_tests.addArgs(args);
    addRawTestModules(b, run_fast_tests, test_build_options);
    run_fast_tests.setEnvironmentVariable("AION_SKIP_THREAD_POOL_TESTS", "1");
    test_fast_step.dependOn(&run_fast_tests.step);

    // ---------------------------------------------------------------------
    // Benchmarks.
    // Run with: zig build bench -Doptimize=ReleaseFast -- [bench args]
    // ---------------------------------------------------------------------
    // Shared benchmark core (op list + unified reporter), imported by both the
    // CPU `bench` and the GPU `gpu-bench` so their ops and output stay in parity.
    const bench_kernels_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_kernels.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "aion", .module = aion_mod },
        },
    });

    const bench_mod = b.createModule(.{
        .root_source_file = b.path("src/bench.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "aion", .module = aion_mod },
            .{ .name = "bench_kernels", .module = bench_kernels_mod },
        },
    });

    const bench_exe = b.addExecutable(.{
        .name = "aion-bench",
        .root_module = bench_mod,
    });
    const run_bench = b.addRunArtifact(bench_exe);
    if (wgpu_dep) |wd| wd.prepareRun(b, run_bench);
    if (b.args) |args| run_bench.addArgs(args);

    const bench_step = b.step("bench", "Run microbenchmarks");
    bench_step.dependOn(&run_bench.step);

    // ---------------------------------------------------------------------
    // GPU backend test (the in-tree backend already compiled into `aion_mod`):
    //   zig build gpu-test     — build and run the CPU-vs-GPU correctness test
    //   zig build gpu-check    — type-check only (used by zls build-on-save)
    // The test also runs as part of `zig build test` / `test-fast` (it is its own
    // artifact because it links wgpu-native; `-Dgpu=false` opts out entirely).
    // ---------------------------------------------------------------------
    if (wgpu_dep) |wd| {
        const gpu_check = b.step("gpu-check", "Type-check the GPU backend (for zls)");
        const gpu_run = addGpuTest(b, target, optimize, aion_mod, wd, gpu_check);

        const gpu_test_step = b.step("gpu-test", "Build and run the GPU backend test");
        gpu_test_step.dependOn(&gpu_run.step);

        // API-level device-selection test (Context.gpus / compileOn(.gpu) / .to()).
        // Must live in its own artifact against `aion_mod` (enable_gpu=true); the raw
        // `test`/`test-fast` runner compiles `test_api.zig` with enable_gpu=false, so
        // the GPU path would be comptime-pruned there.
        const api_gpu_run = addApiGpuTest(b, target, optimize, aion_mod, wd, gpu_check);
        gpu_test_step.dependOn(&api_gpu_run.step);

        // Fold the GPU tests into the default test suites.
        test_step.dependOn(&gpu_run.step);
        test_fast_step.dependOn(&gpu_run.step);
        test_step.dependOn(&api_gpu_run.step);
        test_fast_step.dependOn(&api_gpu_run.step);

        // CPU-vs-GPU benchmark (separate artifact for GPU-specific workloads).
        //   zig build gpu-bench -Doptimize=ReleaseFast -- --m 1024 --n 1024 --k 1024
        const gpu_bench_run = addGpuBench(b, target, optimize, aion_mod, bench_kernels_mod, wd, gpu_check);
        if (b.args) |args| gpu_bench_run.addArgs(args);
        const gpu_bench_step = b.step("gpu-bench", "Build and run the CPU-vs-GPU benchmark");
        gpu_bench_step.dependOn(&gpu_bench_run.step);
    }

    // ---------------------------------------------------------------------
    // Examples.
    // Run with:
    //   zig build examples -- [example args]
    //   zig build bench-examples -- [example args]
    // ---------------------------------------------------------------------
    const examples_step = b.step("examples", "Run all Zig examples under examples/");
    const bench_examples_step = b.step("bench-examples", "Run all examples in benchmark mode");

    var io_backend: std.Io.Threaded = .init_single_threaded;
    const io = io_backend.io();

    const examples_dir_opt: ?std.Io.Dir = std.Io.Dir.cwd().openDir(io, "examples", .{ .iterate = true }) catch |e| switch (e) {
        error.FileNotFound => null,
        else => @panic("failed to open examples directory"),
    };

    if (examples_dir_opt) |examples_dir| {
        var dir = examples_dir;
        defer dir.close(io);

        var it = dir.iterate();
        while (true) {
            const entry_opt = it.next(io) catch @panic("failed iterating examples directory");
            const entry = entry_opt orelse break;
            if (entry.kind != .file) continue;
            if (!std.mem.endsWith(u8, entry.name, ".zig")) continue;

            const stem: []const u8 = std.fs.path.stem(entry.name);
            const rel_path: []const u8 = b.fmt("examples/{s}", .{entry.name});

            const ex_mod = b.createModule(.{
                .root_source_file = b.path(rel_path),
                .target = target,
                .optimize = optimize,
                .imports = &.{
                    .{ .name = "aion", .module = aion_mod },
                },
            });

            const ex_exe = b.addExecutable(.{
                .name = b.fmt("aion-example-{s}", .{stem}),
                .root_module = ex_mod,
            });
            const run_ex = b.addRunArtifact(ex_exe);
            if (wgpu_dep) |wd| wd.prepareRun(b, run_ex);
            if (b.args) |args| run_ex.addArgs(args);
            examples_step.dependOn(&run_ex.step);

            const run_ex_bench = b.addRunArtifact(ex_exe);
            if (wgpu_dep) |wd| wd.prepareRun(b, run_ex_bench);
            run_ex_bench.addArgs(&.{ "--bench-iters", "10" });
            if (b.args) |args| run_ex_bench.addArgs(args);
            bench_examples_step.dependOn(&run_ex_bench.step);
        }
    }

    // Just like flags, top level steps are also listed in the `--help` menu.
    //
    // The Zig build system is entirely implemented in userland, which means
    // that it cannot hook into private compiler APIs. All compilation work
    // orchestrated by the build system will result in other Zig compiler
    // subcommands being invoked with the right flags defined. You can observe
    // these invocations when one fails (or you pass a flag to increase
    // verbosity) to validate assumptions and diagnose problems.
    //
    // Lastly, the Zig build system is relatively simple and self-contained,
    // and reading its source code will allow you to master it.
}

const WgpuDep = struct {
    dep: *std.Build.Dependency,
    os: std.Target.Os.Tag,

    fn link(self: WgpuDep, mod: *std.Build.Module) void {
        mod.addLibraryPath(self.dep.path("lib"));
        switch (self.os) {
            .windows => mod.addObjectFile(self.dep.path("lib/wgpu_native.dll.lib")),
            else => {
                mod.linkSystemLibrary("wgpu_native", .{ .use_pkg_config = .no });
                mod.addRPath(self.dep.path("lib"));
            },
        }
    }

    fn prepareRun(self: WgpuDep, b: *std.Build, run: *std.Build.Step.Run) void {
        if (self.os == .windows) {
            run.addPathDir(self.dep.path("lib").getPath(b));
        }
    }
};

/// Select the prebuilt wgpu-native package matching the build target. Lazy:
/// returns null (triggering a fetch + build re-run) the first time, and null
/// for unsupported target triples (with a helpful message).
fn wgpuDependency(b: *std.Build, target: std.Build.ResolvedTarget) ?WgpuDep {
    const t = target.result;
    const name: ?[]const u8 = switch (t.os.tag) {
        .windows => if (t.cpu.arch == .x86_64) "wgpu_windows_x86_64" else null,
        .linux => if (t.cpu.arch == .x86_64) "wgpu_linux_x86_64" else null,
        .macos => switch (t.cpu.arch) {
            .aarch64 => "wgpu_macos_aarch64",
            .x86_64 => "wgpu_macos_x86_64",
            else => null,
        },
        else => null,
    };
    const dep_name = name orelse {
        std.debug.print("gpu: no prebuilt wgpu-native for {s}-{s}\n", .{ @tagName(t.cpu.arch), @tagName(t.os.tag) });
        return null;
    };
    const dep = b.lazyDependency(dep_name, .{}) orelse return null;
    return .{ .dep = dep, .os = t.os.tag };
}

/// Keep the raw `zig test` runner workaround, but provide the generated options
/// module that `cpu_backend.zig` imports directly.
fn addRawTestModules(b: *std.Build, run: *std.Build.Step.Run, options: *std.Build.Step.Options) void {
    run.addArgs(&.{ "--dep", "build_options" });
    run.addPrefixedFileArg("-Mroot=", b.path("src/tests.zig"));
    run.addPrefixedFileArg("-Mbuild_options=", options.getOutput());
}

/// The CPU-vs-GPU correctness test artifact. Its module imports the `aion` module
/// (whose in-tree gpu backend already has the `wgpu` bindings and native link
/// inputs); the test references `aion.gpu`, so its compilation pulls the gpu code.
/// `check_step` gets a compile-only dependency for zls build-on-save; the returned
/// run step is wired into `gpu-test` and the default test suites.
fn addGpuTest(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    aion_mod: *std.Build.Module,
    wd: WgpuDep,
    check_step: *std.Build.Step,
) *std.Build.Step.Run {
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/aion/backend/gpu/test_gpu_backend.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_mod.addImport("aion", aion_mod);

    const gpu_test = b.addTest(.{ .name = "aion-gpu-test", .root_module = test_mod });
    const run = b.addRunArtifact(gpu_test);
    wd.prepareRun(b, run);

    check_step.dependOn(&gpu_test.step); // compile-only, for zls build-on-save
    return run;
}

/// The API-level GPU device-selection test artifact. Imports `aion` (enable_gpu=true)
/// and drives the public `Context` device API (`.gpus`, `compileOn(.gpu)`, `.to()`).
/// Skips at runtime when no adapter is present.
fn addApiGpuTest(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    aion_mod: *std.Build.Module,
    wd: WgpuDep,
    check_step: *std.Build.Step,
) *std.Build.Step.Run {
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/aion/api/test_api_gpu.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_mod.addImport("aion", aion_mod);

    const api_gpu_test = b.addTest(.{ .name = "aion-api-gpu-test", .root_module = test_mod });
    const run = b.addRunArtifact(api_gpu_test);
    wd.prepareRun(b, run);

    check_step.dependOn(&api_gpu_test.step); // compile-only, for zls build-on-save
    return run;
}

/// The CPU-vs-GPU benchmark exe (`zig build gpu-bench`). Like the test, it imports
/// `aion` and gets wgpu transitively from that module; kept separate from the main
/// `bench` exe because it runs GPU-specific workloads.
fn addGpuBench(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    aion_mod: *std.Build.Module,
    bench_kernels_mod: *std.Build.Module,
    wd: WgpuDep,
    check_step: *std.Build.Step,
) *std.Build.Step.Run {
    const bench_mod = b.createModule(.{
        .root_source_file = b.path("src/aion/backend/gpu/bench_gpu.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    bench_mod.addImport("aion", aion_mod);
    bench_mod.addImport("bench_kernels", bench_kernels_mod);

    const bench = b.addExecutable(.{ .name = "aion-gpu-bench", .root_module = bench_mod });
    const run = b.addRunArtifact(bench);
    wd.prepareRun(b, run);

    check_step.dependOn(&bench.step); // compile-only, for zls build-on-save
    return run;
}

fn optimizeArg(optimize: std.builtin.OptimizeMode) []const u8 {
    return switch (optimize) {
        .Debug => "-ODebug",
        .ReleaseSafe => "-OReleaseSafe",
        .ReleaseFast => "-OReleaseFast",
        .ReleaseSmall => "-OReleaseSmall",
    };
}
