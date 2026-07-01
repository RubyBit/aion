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

    // Portable multi-ISA CPU kernels: when enabled (the default), the shipped
    // library's main module is compiled at the x86_64_v3 floor (AVX2 + FMA) and
    // per-ISA kernel objects (v3/v3_vnni/v4) are linked in, with runtime CPUID
    // dispatch selecting among them. The benchmark exe also links the tier objects
    // and dispatches, so it measures exactly the kernels the shipped library runs.
    //
    // The exported `aion` module used by examples (and external source consumers)
    // stays at the native target with dispatch off, so it is unaffected.
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
    // Resolve the prebuilt wgpu archive up front. Null when disabled, on an
    // unsupported target, or on the lazy-fetch's first pass (build re-runs once
    // fetched). `enable_gpu` follows this so the in-tree gpu code is only enabled
    // once its `wgpu` import is actually wired — never half-configured.
    const wgpu_dep: ?WgpuDep = if (want_gpu) wgpuDependency(b, target) else null;
    const enable_gpu = wgpu_dep != null;

    // Expose a public Zig module for consumers (and our examples). It stays at the
    // user's target with dispatch off — multiversioning targets the shipped library
    // artifact, not source consumers who compile for their own CPU.
    // In a dependent package's build.zig:
    //   const aion_dep = b.dependency("aion", .{});
    //   exe.root_module.addImport("aion", aion_dep.module("aion"));
    const aion_mod = b.addModule("aion", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{},
    });
    // One shared `build_options` module instance for the native world (exported
    // module + bench). Sharing the same instance (rather than addOptions on each)
    // avoids two Options modules wrapping the same generated file when their values
    // coincide — which Zig rejects ("file exists in two modules"). Examples/external
    // consumers never use `aion_mod`'s root as their compilation root, so this value
    // only actually drives dispatch for the bench exe (which links the tier objects).
    const native_build_options = b.addOptions();
    native_build_options.addOption(bool, "multiversion", want_multiversion);
    native_build_options.addOption(bool, "enable_gpu", enable_gpu);
    const native_build_options_mod = native_build_options.createModule();
    aion_mod.addImport("build_options", native_build_options_mod);

    // When the GPU feature is on, give the `aion` module the translate-c'd `wgpu`
    // bindings so its (in-tree, feature-gated) gpu backend compiles. The wgpu
    // import lives on the module; the library *link* is added per-artifact (only
    // the gpu test references `aion.gpu`, so bench/examples/lib never pull wgpu).
    if (wgpu_dep) |wd| {
        const translate = b.addTranslateC(.{
            .root_source_file = wd.dep.path("include/webgpu/wgpu.h"),
            .target = target,
            .optimize = optimize,
        });
        translate.addIncludePath(wd.dep.path("include/webgpu"));
        aion_mod.addImport("wgpu", translate.createModule());
    }

    // Portable multiversioning: compile `kernels_export.zig` once per ISA tier.
    // Each object is built at its own CPU model/feature set, so its `@Vector`
    // kernels lower to that ISA (e.g. the v4 object emits real AVX-512). The
    // objects share no global symbols except their uniquely-named accessor, so the
    // per-tier kernel copies don't collide. Built once and shared by the shipped
    // library and the benchmark exe.
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
    const run_lib_tests = b.addSystemCommand(&.{
        b.graph.zig_exe,
        "test",
        "src/tests.zig",
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
    if (skip_thread_pool_tests) run_lib_tests.setEnvironmentVariable("AION_SKIP_THREAD_POOL_TESTS", "1");
    test_step.dependOn(&run_lib_tests.step);

    const test_fast_step = b.step("test-fast", "Run tests (skip thread pool suite)");
    const run_fast_tests = b.addSystemCommand(&.{
        b.graph.zig_exe,
        "test",
        "src/tests.zig",
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
    run_fast_tests.setEnvironmentVariable("AION_SKIP_THREAD_POOL_TESTS", "1");
    test_fast_step.dependOn(&run_fast_tests.step);

    // ---------------------------------------------------------------------
    // Benchmarks.
    // Run with: zig build bench -Doptimize=ReleaseFast -- [bench args]
    // ---------------------------------------------------------------------
    const bench_mod = b.createModule(.{
        .root_source_file = b.path("src/bench.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{
            .{ .name = "aion", .module = aion_mod },
        },
    });
    // `src/bench.zig` sets `aion_multiversion` from this, so the CPU backend
    // (which reads `@import("root")`) dispatches to the linked tier objects —
    // i.e. the benchmark measures the same kernels the shipped library runs.
    // Reuses the shared instance to avoid the duplicate-options-file conflict.
    bench_mod.addImport("build_options", native_build_options_mod);

    const bench_exe = b.addExecutable(.{
        .name = "aion-bench",
        .root_module = bench_mod,
    });
    for (tier_objs[0..n_tiers]) |o| bench_exe.root_module.addObject(o);

    const run_bench = b.addRunArtifact(bench_exe);
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

        // Fold the GPU test into the default test suites.
        test_step.dependOn(&gpu_run.step);
        test_fast_step.dependOn(&gpu_run.step);

        // CPU-vs-GPU benchmark (separate artifact; keeps `bench` wgpu-free).
        //   zig build gpu-bench -Doptimize=ReleaseFast -- --m 1024 --n 1024 --k 1024
        const gpu_bench_run = addGpuBench(b, target, optimize, aion_mod, wd, gpu_check);
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
            if (b.args) |args| run_ex.addArgs(args);
            examples_step.dependOn(&run_ex.step);

            const run_ex_bench = b.addRunArtifact(ex_exe);
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
};

/// Dependency name for the prebuilt wgpu-native archive matching `t`, or null
/// for an unsupported target triple. Written with independent `return`s (rather
/// than an `if (cond) "lit" else null` expression) so each value coerces to
/// `?[]const u8` on its own — avoids a pointer-vs-optional peer-type resolution
/// that crashes some zls versions during inlay-hint analysis.
fn wgpuDepName(t: std.Target) ?[]const u8 {
    switch (t.os.tag) {
        .windows => if (t.cpu.arch == .x86_64) return "wgpu_windows_x86_64",
        .linux => if (t.cpu.arch == .x86_64) return "wgpu_linux_x86_64",
        .macos => switch (t.cpu.arch) {
            .aarch64 => return "wgpu_macos_aarch64",
            .x86_64 => return "wgpu_macos_x86_64",
            else => {},
        },
        else => {},
    }
    return null;
}

/// Select the prebuilt wgpu-native archive matching the build target. Lazy:
/// returns null (triggering a fetch + build re-run) the first time, and null
/// for unsupported target triples (with a helpful message).
fn wgpuDependency(b: *std.Build, target: std.Build.ResolvedTarget) ?WgpuDep {
    const t = target.result;
    const n = wgpuDepName(t) orelse {
        std.debug.print("gpu: no prebuilt wgpu-native for {s}-{s}\n", .{ @tagName(t.cpu.arch), @tagName(t.os.tag) });
        return null;
    };
    const dep = b.lazyDependency(n, .{}) orelse return null;
    return .{ .dep = dep, .os = t.os.tag };
}

/// Link the prebuilt wgpu-native library into a GPU artifact's module and make its
/// run step able to find the shared lib. Shared by the gpu test + gpu bench (both
/// reference `aion.gpu`, so their compilation pulls the gpu code and must link wgpu).
fn linkWgpu(b: *std.Build, mod: *std.Build.Module, run: *std.Build.Step.Run, wd: WgpuDep) void {
    mod.addLibraryPath(wd.dep.path("lib"));
    switch (wd.os) {
        .windows => {
            mod.addObjectFile(wd.dep.path("lib/wgpu_native.dll.lib"));
            run.addPathDir(wd.dep.path("lib").getPath(b));
        },
        else => {
            mod.linkSystemLibrary("wgpu_native", .{});
            mod.addRPath(wd.dep.path("lib"));
        },
    }
}

/// The CPU-vs-GPU correctness test artifact. Its module imports the `aion` module
/// (whose in-tree gpu backend already has the `wgpu` bindings); the test references
/// `aion.gpu`, so its compilation pulls the gpu code and must link the wgpu library.
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
    linkWgpu(b, test_mod, run, wd);

    check_step.dependOn(&gpu_test.step); // compile-only, for zls build-on-save
    return run;
}

/// The CPU-vs-GPU benchmark exe (`zig build gpu-bench`). Like the test, it imports
/// `aion` and links wgpu; kept separate from the main `bench` exe so a plain
/// `zig build bench` never pulls wgpu.
fn addGpuBench(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    aion_mod: *std.Build.Module,
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

    const bench = b.addExecutable(.{ .name = "aion-gpu-bench", .root_module = bench_mod });
    const run = b.addRunArtifact(bench);
    linkWgpu(b, bench_mod, run, wd);

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
