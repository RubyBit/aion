const std = @import("std");
const builtin = @import("builtin");
const cpuid_root = @import("cpuid.zig");

pub fn detect() cpuid_root.CpuInfo {
    var info: cpuid_root.CpuInfo = .{ .arch = .x86_64 };

    const max_leaf: u32 = cpuid(0, 0).eax;

    // Topology (best-effort).
    if (max_leaf >= 1) {
        const l1 = cpuid(1, 0);
        info.logical_processors = @as(usize, @intCast((l1.ebx >> 16) & 0xFF));
    }

    // Prefer modern topology leaves (0x1F / 0x0B):
    // - level type 1 => SMT logical processors per core
    // - level type 2 => logical processors per package
    // physical cores per package ~= level2 / level1
    const topo_leaf: u32 = if (max_leaf >= 0x1F)
        0x1F
    else if (max_leaf >= 0x0B)
        0x0B
    else
        0;

    if (topo_leaf != 0) {
        var smt_logical: usize = 0;
        var pkg_logical: usize = 0;
        var subleaf: u32 = 0;
        while (subleaf < 8) : (subleaf += 1) {
            const r = cpuid(topo_leaf, subleaf);
            const level_type: u32 = (r.ecx >> 8) & 0xFF;
            const logical_at_level: usize = @as(usize, @intCast(r.ebx & 0xFFFF));
            if (logical_at_level == 0) break;

            switch (level_type) {
                1 => smt_logical = logical_at_level,
                2 => pkg_logical = logical_at_level,
                else => {},
            }
        }

        if (pkg_logical != 0) {
            info.logical_processors = pkg_logical;
            if (smt_logical != 0) {
                info.physical_cores = @max(@as(usize, 1), pkg_logical / smt_logical);
            }
        }
    }

    // Fallback topology from deterministic cache params when modern leaves are
    // unavailable or incomplete.
    if (info.physical_cores == 0 and max_leaf >= 4) {
        const l4_0 = cpuid(4, 0);
        // Deterministic cache leaf encodes "number of cores - 1" in EAX[31:26].
        if ((l4_0.eax & 0x1F) != 0) {
            info.physical_cores = @as(usize, @intCast(((l4_0.eax >> 26) & 0x3F) + 1));
        }
    }
    if (info.logical_processors != 0 and info.physical_cores == 0) {
        info.physical_cores = info.logical_processors;
    }

    if (max_leaf >= 7) {
        const l7_0 = cpuid(7, 0);
        // AVX2: EBX bit 5
        info.features.avx2 = ((l7_0.ebx >> 5) & 1) != 0;
        // AVX-VNNI: ECX bit 4
        info.features.avx_vnni = ((l7_0.ecx >> 4) & 1) != 0;
        // AVX512-VNNI: ECX bit 11
        info.features.avx512_vnni = ((l7_0.ecx >> 11) & 1) != 0;
        // AMX-INT8: EDX bit 25
        info.features.amx_int8 = ((l7_0.edx >> 25) & 1) != 0;
    }

    info.caches = detectCaches();
    return info;
}

const CpuidRegs = struct { eax: u32, ebx: u32, ecx: u32, edx: u32 };

fn cpuid(leaf: u32, subleaf: u32) CpuidRegs {
    var eax: u32 = leaf;
    var ebx: u32 = 0;
    var ecx: u32 = subleaf;
    var edx: u32 = 0;

    if (builtin.cpu.arch == .x86_64) {
        asm volatile (
            \\cpuid
            : [eax] "+{eax}" (eax),
              [ebx] "=&{rbx}" (ebx),
              [ecx] "+{ecx}" (ecx),
              [edx] "={edx}" (edx),
            :
            : .{ .cc = true, .memory = true });
    } else {
        asm volatile (
            \\cpuid
            : [eax] "+{eax}" (eax),
              [ebx] "={ebx}" (ebx),
              [ecx] "+{ecx}" (ecx),
              [edx] "={edx}" (edx),
            :
            : .{ .cc = true, .memory = true });
    }

    return .{ .eax = eax, .ebx = ebx, .ecx = ecx, .edx = edx };
}

fn detectCaches() cpuid_root.Caches {
    const max_leaf: u32 = cpuid(0, 0).eax;
    if (max_leaf < 4) return .{};

    var caches: cpuid_root.Caches = .{};

    var i: u32 = 0;
    while (true) : (i += 1) {
        const r = cpuid(4, i);
        const cache_type: u32 = r.eax & 0x1F;
        if (cache_type == 0) break;

        const level: u32 = (r.eax >> 5) & 0x7;
        const line_size: u32 = (r.ebx & 0xFFF) + 1;
        const partitions: u32 = ((r.ebx >> 12) & 0x3FF) + 1;
        const ways: u32 = ((r.ebx >> 22) & 0x3FF) + 1;
        const sets: u32 = r.ecx + 1;
        const size_bytes_u64: u64 = @as(u64, line_size) * @as(u64, partitions) * @as(u64, ways) * @as(u64, sets);
        const size_bytes: usize = @intCast(@min(size_bytes_u64, @as(u64, std.math.maxInt(usize))));

        switch (level) {
            1 => {
                if (cache_type == 1 or cache_type == 3) caches.l1d_bytes = @max(caches.l1d_bytes, size_bytes);
            },
            2 => caches.l2_bytes = @max(caches.l2_bytes, size_bytes),
            3 => caches.l3_bytes = @max(caches.l3_bytes, size_bytes),
            else => {},
        }
    }

    return caches;
}
