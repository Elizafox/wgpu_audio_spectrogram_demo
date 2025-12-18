// Writes one spectrogram column for a single FFT frame into a storage texture.

struct Params {
    n: u32,       // FFT size
    bins: u32,    // N / 2 + 1
    col: u32,     // Target column in the texture
    width: u32,   // Texture width (history columns)
    height: u32,  // Texture height
};

@group(0) @binding(0) var<storage, read> X: array<vec2<f32>>;
@group(0) @binding(1) var<uniform> U: Params;
@group(0) @binding(2) var spec_tex: texture_storage_2d<rgba8unorm, write>;

fn lerp3(a: vec3<f32>, b: vec3<f32>, t: f32) -> vec3<f32> {
    return a + (b - a) * t;
}

// Log mapping helper
fn log_sample_bin(y: u32, bins: u32, out_h: u32, k_min: f32, k_max: f32) -> f32 {
    // y = 0 top = highest freq; flip so y grows downward
    let t = 1.0 - (f32(y) / f32(max(out_h, 1u) - 1u));
    let lo = log(k_min);
    let hi = log(k_max);
    return exp(lo + t * (hi - lo)); // Fractional source bin index in [k_min..k_max]
}

// Jet colourisation of spectrogram
fn jet(t0: f32) -> vec3<f32> {
    const PAL = array<vec3<f32>, 5>(
        vec3<f32>(0.0, 0.0, 0.5),
        vec3<f32>(0.0, 0.5, 1.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(1.0, 1.0, 0.0),
        vec3<f32>(1.0, 0.0, 0.0)
    );
    let t = pow(clamp(t0, 0.0, 1.0), 0.8);
    let s = 4.0 * t;
    let i = u32(clamp(floor(s), 0.0, 3.0));
    let f = fract(s);
    return lerp3(PAL[i], PAL[i + 1u], f);
}

@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    const FLOOR: f32 = -100.0;
    const CEIL: f32 = -25.0;

    let y = gid.x;  // One thread per output row
    if y >= U.height {
        return;
    }

    // Pick log range; skip DC so k_min = 1
    let k_min = 1.0;
    let k_max = f32(max(U.bins, 2u) - 1u);

    // Fractional source bin
    var kf = log_sample_bin(y, U.bins, U.height, k_min, k_max);
    kf = clamp(kf, k_min, k_max);
    let k0 = u32(floor(kf));
    let k1 = min(k0 + 1u, U.bins - 1u);
    let w = kf - f32(k0);

    // Linear interpolate complex bins
    let c0 = X[k0];
    let c1 = X[k1];
    let c = vec2<f32>(mix(c0.x, c1.x, w), mix(c0.y, c1.y, w));

    // dB mapping
    let ref_mag = 0.25 * f32(U.n);  // Hann coherent gain
    let mag = length(c);
    let db = 20.0 * log(max(mag / ref_mag, 1e-20)) / log(10.0);
    var norm = clamp((db - FLOOR) / (CEIL - FLOOR), 0.0, 1.0);
    let g = pow(norm, 0.6);

    let rgb = max(jet(g) - vec3<f32>(0.02), vec3<f32>(0.0));
    textureStore(spec_tex, vec2<i32>(i32(U.col), i32(y)), vec4<f32>(rgb, 1.0));
}
