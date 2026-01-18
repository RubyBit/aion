const std = @import("std");
const builtin = @import("builtin");

pub const CpuFeatures = struct {
    avx2: bool = false,
    avx_vnni: bool = false,
    avx512_vnni: bool = false,
    amx_int8: bool = false,
};

pub const Caches = struct {
    l1d_bytes: usize = 0,
    l2_bytes: usize = 0,
    l3_bytes: usize = 0,
};

pub const CpuInfo = struct {
    features: CpuFeatures = .{},
    caches: Caches = .{},
};

pub fn detect() CpuInfo {
    if (builtin.cpu.arch != .x86_64 and builtin.cpu.arch != .x86) {
        return CpuInfo{};
    }

    var info: CpuInfo = CpuInfo{};

    const max_leaf: u32 = cpuid(0, 0).eax;
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
    // For x86_64 under the SysV and Win64 ABIs, RBX is a callee-saved register.
    // Use an early-clobber output for RBX to make the register allocator happy
    // and avoid the "couldn't allocate output register" failures.
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
            : .{ .cc = true, .memory = true }
        );
    } else {
        asm volatile (
            \\cpuid
            : [eax] "+{eax}" (eax),
              [ebx] "={ebx}" (ebx),
              [ecx] "+{ecx}" (ecx),
              [edx] "={edx}" (edx),
            :
            : .{ .cc = true, .memory = true }
        );
    }

    return .{ .eax = eax, .ebx = ebx, .ecx = ecx, .edx = edx };
}

fn detectCaches() Caches {
    const max_leaf: u32 = cpuid(0, 0).eax;
    if (max_leaf < 4) return Caches{};

    var caches: Caches = Caches{};

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
                // Type 1 = data cache, 3 = unified.
                if (cache_type == 1 or cache_type == 3) caches.l1d_bytes = @max(caches.l1d_bytes, size_bytes);
            },
            2 => caches.l2_bytes = @max(caches.l2_bytes, size_bytes),
            3 => caches.l3_bytes = @max(caches.l3_bytes, size_bytes),
            else => {},
        }
    }

    return caches;
}
