// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// Zero-pad (or crop) packed NHWC: destination (n, h, w, c) is source
// (n, h - top, w - left, c), or zero where that leaves the source. Channels
// move four at a time, so both shapes are (N, H, W, C / 4); `at` is
// (top, left, destination vec4 count).
struct Params { src_shape: vec4<u32>, dst_shape: vec4<u32>, at: vec4<u32> };
@group(0) @binding(0) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> y: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> p: Params;

@compute @workgroup_size(64)
fn pad_f32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    for (var i = gid.x; i < p.at.z; i += ng.x * 64u) {
        let c = i % p.dst_shape.w;
        let w = (i / p.dst_shape.w) % p.dst_shape.z;
        let h = (i / (p.dst_shape.w * p.dst_shape.z)) % p.dst_shape.y;
        let n = i / (p.dst_shape.w * p.dst_shape.z * p.dst_shape.y);
        let sh = i32(h) - i32(p.at.x);
        let sw = i32(w) - i32(p.at.y);
        var v = vec4<f32>(0.0);
        if (sh >= 0 && sh < i32(p.src_shape.y) && sw >= 0 && sw < i32(p.src_shape.z)) {
            v = x[((n * p.src_shape.y + u32(sh)) * p.src_shape.z + u32(sw)) * p.src_shape.w + c];
        }
        y[i] = v;
    }
}
