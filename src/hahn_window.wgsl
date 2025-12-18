// Hahn windowing function

struct Params {
    n: u32,
    batch_stride: u32,
};

struct Cx {
    data: array<vec2<f32>>,
};
struct W {
    w: array<f32>,
};

@group(0) @binding(0) var<storage, read_write> X: Cx; // In-place windowing
@group(0) @binding(1) var<storage, read> WIN: W;
@group(0) @binding(2) var<uniform> U: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let b = gid.y;
    if i >= U.n {
        return;
    }

    let idx = b * U.batch_stride + i;
    let s = WIN.w[i];
    let v = X.data[idx];
    X.data[idx] = vec2<f32>(v.x * s, v.y * s);
}
