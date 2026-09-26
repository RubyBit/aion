// SPDX-License-Identifier: EPL-2.0 OR GPL-2.0-or-later
//
// MaxPool2D over packed NHWC: one work item per output element. `window` is
// (kernel_h, kernel_w, stride_h, stride_w), `geometry` (dilation_h, dilation_w,
// pad_top, pad_left), `control.x` the output element count. A NaN in the window
// wins, as on the CPU.
enable f16;
struct Params {
    src_shape: vec4<u32>, dst_shape: vec4<u32>,
    window: vec4<u32>, geometry: vec4<u32>, control: vec4<u32>,
};
@group(0) @binding(0) var<storage, read> x: array<f32>;
@group(0) @binding(1) var<storage, read_write> y: array<f32>;
@group(0) @binding(0) var<storage, read> xh: array<f16>;
@group(0) @binding(1) var<storage, read_write> yh: array<f16>;
@group(0) @binding(2) var<uniform> p: Params;

/// Output element `i`'s (n, h, w, c) coordinate.
fn outCoord(i: u32) -> vec4<u32> {
    var rem = i;
    var coord: vec4<u32>;
    for (var d = 3; d >= 0; d--) { coord[d] = rem % p.dst_shape[d]; rem /= p.dst_shape[d]; }
    return coord;
}

/// Flat index of input (n, ih, iw, c), or -1 where the window leaves the image.
fn srcIndex(coord: vec4<u32>, kh: u32, kw: u32) -> i32 {
    let ih = i32(coord.y * p.window.z + kh * p.geometry.x) - i32(p.geometry.z);
    let iw = i32(coord.z * p.window.w + kw * p.geometry.y) - i32(p.geometry.w);
    if (ih < 0 || ih >= i32(p.src_shape.y) || iw < 0 || iw >= i32(p.src_shape.z)) { return -1; }
    return i32(((coord.x * p.src_shape.y + u32(ih)) * p.src_shape.z + u32(iw)) * p.src_shape.w + coord.w);
}

fn better(v: f32, best: f32) -> bool {
    return (bitcast<u32>(v) & 0x7fffffffu) > 0x7f800000u || v > best;
}

@compute @workgroup_size(64)
fn pool_f32(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    for (var i = gid.x; i < p.control.x; i += ng.x * 64u) {
        let coord = outCoord(i);
        var best = bitcast<f32>(0xff800000u);
        for (var kh = 0u; kh < p.window.x; kh++) {
            for (var kw = 0u; kw < p.window.y; kw++) {
                let s = srcIndex(coord, kh, kw);
                if (s < 0) { continue; }
                let v = x[u32(s)];
                if (better(v, best)) { best = v; }
            }
        }
        y[i] = best;
    }
}

@compute @workgroup_size(64)
fn pool_f16(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) ng: vec3<u32>) {
    for (var i = gid.x; i < p.control.x; i += ng.x * 64u) {
        let coord = outCoord(i);
        var best = bitcast<f32>(0xff800000u);
        for (var kh = 0u; kh < p.window.x; kh++) {
            for (var kw = 0u; kw < p.window.y; kw++) {
                let s = srcIndex(coord, kh, kw);
                if (s < 0) { continue; }
                let v = f32(xh[u32(s)]);
                if (better(v, best)) { best = v; }
            }
        }
        yh[i] = f16(best);
    }
}
