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

    // Expose a public Zig module for consumers.
    // In a dependent package's build.zig:
    //   const aion_dep = b.dependency("aion", .{});
    //   exe.root_module.addImport("aion", aion_dep.module("aion"));
    const aion_mod = b.addModule("aion", .{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
        .imports = &.{},
    });

    // Build an installable library artifact (useful for C/FFI consumers, and for
    // `zig build install` producing `zig-out/lib/*`).
    const linkage = b.option(
        std.builtin.LinkMode,
        "linkage",
        "Library linkage: .static (default) or .dynamic",
    ) orelse .static;

    const lib = b.addLibrary(.{
        .name = "aion",
        .root_module = aion_mod,
        .linkage = linkage,
    });

    b.installArtifact(lib);

    // Unit tests.
    // On this Zig snapshot, running the test artifact through Build's special
    // `--listen=-` test runner mode can stall on Windows. The direct `zig test`
    // path is the closest stable fallback and still honors the selected
    // target/optimize settings.
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
    test_step.dependOn(&run_lib_tests.step);

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

    const bench_exe = b.addExecutable(.{
        .name = "aion-bench",
        .root_module = bench_mod,
    });

    const run_bench = b.addRunArtifact(bench_exe);
    if (b.args) |args| run_bench.addArgs(args);

    const bench_step = b.step("bench", "Run microbenchmarks");
    bench_step.dependOn(&run_bench.step);

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

fn optimizeArg(optimize: std.builtin.OptimizeMode) []const u8 {
    return switch (optimize) {
        .Debug => "-ODebug",
        .ReleaseSafe => "-OReleaseSafe",
        .ReleaseFast => "-OReleaseFast",
        .ReleaseSmall => "-OReleaseSmall",
    };
}
