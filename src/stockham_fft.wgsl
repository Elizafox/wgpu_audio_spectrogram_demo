// Stockham algorithm for FFT

const PI: f32 = 3.14159265358979323846;

struct Params {
    n: u32,             // FFT length (power of two)
    m: u32,             // Stage size (2, 4, 8, ..., n)
    mh: u32,            // m / 2
    direction: i32,     // -1 forward, +1 inverse
    batch_stride: u32,  // Elements between batch starts
};

struct Cx {
    data: array<vec2<f32>>,
};

// Precomputed forward twiddles: TW[k] = exp(-i * 2 * pi * k / N)
struct Tw {
    w: array<vec2<f32>>,
};

@group(0) @binding(0) var<storage, read> X_in:  Cx;
@group(0) @binding(1) var<storage, read_write> X_out: Cx;
@group(0) @binding(2) var<uniform> U: Params;
@group(0) @binding(3) var<storage, read> TW: Tw;

fn cx_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // gid.x: butterflies across N / 2
    // gid.y: batch index (frame)
    let t = gid.x;
    if t >= U.n / 2u {
        return;
    }

    let batch = gid.y;
    let base = batch * U.batch_stride;

    let k = t % U.mh;
    let g = t / U.mh;

    // Input indices (contiguous pairs this stage)
    let i0 = base + g * U.m + k;
    let i1 = i0 + U.mh;

    // Twiddle lookup: stride picks every (N / m) entry
    let stride = U.n / U.m;
    var W = TW.w[(k * stride) % U.n];
    if U.direction == 1 { // inverse: conj(W)
        W.y = -W.y;
    }

    let a = X_in.data[i0];
    let b = X_in.data[i1];
    let tval = cx_mul(W, b);

    // Stockham shuffle (autosort)
    let o0 = base + g * U.m + (k * 2u);
    let o1 = o0 + 1u;

    X_out.data[o0] = a + tval;
    X_out.data[o1] = a - tval;
}
