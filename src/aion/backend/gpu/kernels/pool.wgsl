// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
enable f16;
struct Params {
    src_shape: vec4<u32>, src_origin: vec4<u32>, src_stride: vec4<u32>,
    dst_shape: vec4<u32>, dst_origin: vec4<u32>, dst_stride: vec4<u32>,
    window: vec4<u32>, geometry: vec4<u32>, control: vec4<u32>,
};
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(0) var<storage, read> xh: array<f16>;
@group(0) @binding(1) var<storage, read_write> yh: array<f16>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(64)
fn pool_f32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    for (var i = gid.x; i < p.control.x; i += ng.x * 64u) {
        var rem = i;
        var local: vec4<u32>;
        for (var d = 3; d >= 0; d--) { local[d] = rem % p.dst_shape[d]; rem /= p.dst_shape[d]; }
        let coord = local + p.dst_origin;
        let offset = local.x*p.dst_stride.x + local.y*p.dst_stride.y + local.z*p.dst_stride.z + local.w*p.dst_stride.w;
        var best = bitcast<f32>(0xff800000u);
        if (p.control.y == 0u) { best = f32(y[offset]); }
        if (coord.x >= p.src_origin.x && coord.x < p.src_origin.x + p.src_shape.x && coord.w >= p.src_origin.w && coord.w < p.src_origin.w + p.src_shape.w) {
            for (var kh = 0u; kh < p.window.x; kh++) {
                let ih = i32(coord.y * p.window.z + kh * p.geometry.x) - i32(p.geometry.z);
                if (ih < i32(p.src_origin.y) || ih >= i32(p.src_origin.y + p.src_shape.y)) { continue; }
                for (var kw = 0u; kw < p.window.y; kw++) {
                    let iw = i32(coord.z * p.window.w + kw * p.geometry.y) - i32(p.geometry.w);
                    if (iw < i32(p.src_origin.z) || iw >= i32(p.src_origin.z + p.src_shape.z)) { continue; }
                    let c = vec4<u32>(coord.x, u32(ih), u32(iw), coord.w) - p.src_origin;
                    let index = c.x*p.src_stride.x + c.y*p.src_stride.y + c.z*p.src_stride.z + c.w*p.src_stride.w;
                    let v = f32(x[index]);
                    if ((bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u || v > best) { best = v; }
                }
            }
        }
        y[offset] = f32(best);
    }
}

@compute @workgroup_size(64)
fn pool_f16(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    for (var i = gid.x; i < p.control.x; i += ng.x * 64u) {
        var rem = i;
        var local: vec4<u32>;
        for (var d = 3; d >= 0; d--) { local[d] = rem % p.dst_shape[d]; rem /= p.dst_shape[d]; }
        let coord = local + p.dst_origin;
        let offset = local.x*p.dst_stride.x + local.y*p.dst_stride.y + local.z*p.dst_stride.z + local.w*p.dst_stride.w;
        var best = bitcast<f32>(0xff800000u);
        if (p.control.y == 0u) { best = f32(yh[offset]); }
        if (coord.x >= p.src_origin.x && coord.x < p.src_origin.x + p.src_shape.x && coord.w >= p.src_origin.w && coord.w < p.src_origin.w + p.src_shape.w) {
            for (var kh = 0u; kh < p.window.x; kh++) {
                let ih = i32(coord.y * p.window.z + kh * p.geometry.x) - i32(p.geometry.z);
                if (ih < i32(p.src_origin.y) || ih >= i32(p.src_origin.y + p.src_shape.y)) { continue; }
                for (var kw = 0u; kw < p.window.y; kw++) {
                    let iw = i32(coord.z * p.window.w + kw * p.geometry.y) - i32(p.geometry.w);
                    if (iw < i32(p.src_origin.z) || iw >= i32(p.src_origin.z + p.src_shape.z)) { continue; }
                    let c = vec4<u32>(coord.x, u32(ih), u32(iw), coord.w) - p.src_origin;
                    let index = c.x*p.src_stride.x + c.y*p.src_stride.y + c.z*p.src_stride.z + c.w*p.src_stride.w;
                    let v = f32(xh[index]);
                    if ((bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u || v > best) { best = v; }
                }
            }
        }
        yh[offset] = f16(best);
    }
}
