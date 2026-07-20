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

// Bias applied to the row->frequency mapping so the (mostly featureless,
// heavily-averaged) top octaves get fewer pixel rows and the rest of the
// spectrum gets more. 1.0 = pure log-uniform; higher = more compression
// at the top.
const FREQ_GAMMA: f32 = 0.5;

// Log mapping helper: continuous version so we can evaluate row boundaries
// at yf = y - 0.5 and yf = y + 0.5, not just integer row centers.
fn log_sample_bin(yf: f32, out_h: f32, k_min: f32, k_max: f32) -> f32 {
    // y = 0 top = highest freq; flip so y grows downward
    let t = 1.0 - (yf / max(out_h - 1.0, 1.0));
    let t_biased = pow(clamp(t, 0.0, 1.0), FREQ_GAMMA);
    let lo = log(k_min);
    let hi = log(k_max);
    return exp(lo + t_biased * (hi - lo)); // Fractional source bin index in [k_min..k_max]
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
    let out_h = f32(U.height);

    // Each output row covers a range of source bins under the log mapping
    // (a wide range near the top/high-frequency end, at N=1024 the top octave
    // alone spans roughly half of all bins). Average power over that whole
    // range instead of interpolating just the two nearest bins, otherwise
    // most of the spectral energy there is discarded and the picture is
    // dominated by per-bin periodogram noise (visible as speckle/grain,
    // worst near the top).
    let k_hi = clamp(log_sample_bin(f32(y) - 0.5, out_h, k_min, k_max), k_min, k_max);
    let k_lo = clamp(log_sample_bin(f32(y) + 0.5, out_h, k_min, k_max), k_min, k_max);

    // Gaussian-weighted average over each row's neighborhood in bin space
    // (sigma tied to the row's own span, so bottom rows get a touch of blur
    // and top rows - which already span many bins - get proportionally more).
    // A soft Gaussian roll-off avoids both the per-bin periodogram grain and
    // the hard-edged-box staircase, without touching the time axis at all.
    let center = 0.5 * (k_lo + k_hi);
    let sigma = max((k_hi - k_lo) * 0.5, 0.6);
    let b0 = u32(clamp(floor(center - 3.0 * sigma), k_min, k_max));
    let b1 = u32(clamp(ceil(center + 3.0 * sigma), k_min, k_max));

    var power_sum = 0.0;
    var weight_sum = 0.0;
    for (var b = b0; b <= b1; b = b + 1u) {
        let d = (f32(b) + 0.5 - center) / sigma;
        let weight = exp(-0.5 * d * d);
        let c = X[b];
        power_sum += dot(c, c) * weight;
        weight_sum += weight;
    }
    let mag = sqrt(power_sum / max(weight_sum, 1e-6));

    // dB mapping
    let ref_mag = 0.5 * f32(U.n);  // Hann window coherent gain
    let db = 20.0 * log(max(mag / ref_mag, 1e-20)) / log(10.0);
    var norm = clamp((db - FLOOR) / (CEIL - FLOOR), 0.0, 1.0);
    let g = pow(norm, 0.6);

    let rgb = max(jet(g) - vec3<f32>(0.02), vec3<f32>(0.0));
    textureStore(spec_tex, vec2<i32>(i32(U.col), i32(y)), vec4<f32>(rgb, 1.0));
}
