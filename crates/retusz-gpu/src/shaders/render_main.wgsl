// Retusz GPU Compute Shader (WGSL):
// 33^3 Tetrahedral 3D LUT + Linear-light spatial gain (Dodge & Burn v2 + Plastyka 3D + Clarity)
// + Vignette + Highlight/Shadow clipping zebra overlay.
// Author: Jack4B.

struct Uniforms {
    width: u32,
    height: u32,
    lut_n: u32,
    show_clip: u32,
    vignette: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> src_rgba: array<u32>;
@group(0) @binding(2) var<storage, read> lut_rgb: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> gain_map: array<f32>;
@group(0) @binding(4) var<storage, read_write> dst_rgba: array<u32>;

fn srgb_to_lin(c: f32) -> f32 {
    let x = clamp(c, 0.0, 1.0);
    if (x <= 0.04045) {
        return x / 12.92;
    }
    return pow((x + 0.055) / 1.055, 2.4);
}

fn lin_to_srgb(c: f32) -> f32 {
    let x = clamp(c, 0.0, 1.0);
    if (x <= 0.0031308) {
        return 12.92 * x;
    }
    return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

fn lut_at(ir: u32, ig: u32, ib: u32) -> vec3<f32> {
    let n = u.lut_n;
    return lut_rgb[(ir * n + ig) * n + ib].xyz;
}

fn sample_tetra(rgb01: vec3<f32>) -> vec3<f32> {
    let nm1 = f32(u.lut_n - 1u);
    let x = clamp(rgb01, vec3<f32>(0.0), vec3<f32>(1.0)) * nm1;
    let max_i = u.lut_n - 2u;
    let r0 = min(u32(floor(x.x)), max_i);
    let g0 = min(u32(floor(x.y)), max_i);
    let b0 = min(u32(floor(x.z)), max_i);
    let fr = x.x - f32(r0);
    let fg = x.y - f32(g0);
    let fb = x.z - f32(b0);

    let v000 = lut_at(r0, g0, b0);
    let v111 = lut_at(r0 + 1u, g0 + 1u, b0 + 1u);

    if (fr >= fg) {
        if (fg >= fb) {
            let va = lut_at(r0 + 1u, g0, b0);
            let vb = lut_at(r0 + 1u, g0 + 1u, b0);
            return (1.0 - fr) * v000 + (fr - fg) * va + (fg - fb) * vb + fb * v111;
        } else if (fr >= fb) {
            let va = lut_at(r0 + 1u, g0, b0);
            let vb = lut_at(r0 + 1u, g0, b0 + 1u);
            return (1.0 - fr) * v000 + (fr - fb) * va + (fb - fg) * vb + fg * v111;
        } else {
            let va = lut_at(r0, g0, b0 + 1u);
            let vb = lut_at(r0 + 1u, g0, b0 + 1u);
            return (1.0 - fb) * v000 + (fb - fr) * va + (fr - fg) * vb + fg * v111;
        }
    } else {
        if (fb > fg) {
            let va = lut_at(r0, g0, b0 + 1u);
            let vb = lut_at(r0, g0 + 1u, b0 + 1u);
            return (1.0 - fb) * v000 + (fb - fg) * va + (fg - fr) * vb + fr * v111;
        } else if (fb > fr) {
            let va = lut_at(r0, g0 + 1u, b0);
            let vb = lut_at(r0, g0 + 1u, b0 + 1u);
            return (1.0 - fg) * v000 + (fg - fb) * va + (fb - fr) * vb + fr * v111;
        } else {
            let va = lut_at(r0, g0 + 1u, b0);
            let vb = lut_at(r0 + 1u, g0 + 1u, b0);
            return (1.0 - fg) * v000 + (fg - fr) * va + (fr - fb) * vb + fb * v111;
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= u.width || y >= u.height) {
        return;
    }
    let idx = y * u.width + x;
    let packed = src_rgba[idx];
    let r8 = f32(packed & 0xFFu) / 255.0;
    let g8 = f32((packed >> 8u) & 0xFFu) / 255.0;
    let b8 = f32((packed >> 16u) & 0xFFu) / 255.0;

    var c = sample_tetra(vec3<f32>(r8, g8, b8));
    var g = 1.0 + gain_map[idx];

    if (abs(u.vignette) > 1e-4) {
        let nx = (f32(x) / f32(max(u.width, 1u))) - 0.5;
        let ny = (f32(y) / f32(max(u.height, 1u))) - 0.5;
        let r2 = 2.2 * (nx * nx + ny * ny);
        g = g * clamp(1.0 + 0.55 * u.vignette * r2 * r2, 0.25, 1.75);
    }

    if (abs(g - 1.0) > 1e-4) {
        let lr = srgb_to_lin(c.x) * g;
        let lg = srgb_to_lin(c.y) * g;
        let lb = srgb_to_lin(c.z) * g;
        c = vec3<f32>(lin_to_srgb(lr), lin_to_srgb(lg), lin_to_srgb(lb));
    }

    var ro = u32(clamp(c.x, 0.0, 1.0) * 255.0 + 0.5);
    var go = u32(clamp(c.y, 0.0, 1.0) * 255.0 + 0.5);
    var bo = u32(clamp(c.z, 0.0, 1.0) * 255.0 + 0.5);

    if (u.show_clip != 0u) {
        if (ro >= 254u || go >= 254u || bo >= 254u) {
            ro = 245u; go = 55u; bo = 55u;
        } else if (ro <= 1u && go <= 1u && bo <= 1u) {
            ro = 55u; go = 135u; bo = 250u;
        }
    }

    dst_rgba[idx] = ro | (go << 8u) | (bo << 16u) | (255u << 24u);
}
