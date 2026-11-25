//! CIE / ICC Bradford Chromatic Adaptation Transform (Lam 1985) and one-click white balance eyedropper.
//! Author: Jack4B.

use crate::color::{lin_to_srgb, srgb_to_lin};

pub const TEMP_STRENGTH: f32 = 0.35;
pub const TINT_STRENGTH: f32 = 0.25;

// sRGB (D65) -> CIE XYZ (IEC 61966-2-1)
const SRGB_TO_XYZ: [[f64; 3]; 3] = [
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
];

// CIE XYZ -> Bradford LMS cone response matrix (Lam 1985)
const XYZ_TO_BFD: [[f64; 3]; 3] = [
    [ 0.8951000,  0.2664000, -0.1614000],
    [-0.7502000,  1.7135000,  0.0367000],
    [ 0.0389000, -0.0685000,  1.0296000],
];

fn mat_mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0f64; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = a[r][0] * b[0][c] + a[r][1] * b[1][c] + a[r][2] * b[2][c];
        }
    }
    out
}

fn mat_inv(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let i00 = m[1][1] * m[2][2] - m[1][2] * m[2][1];
    let i01 = m[0][2] * m[2][1] - m[0][1] * m[2][2];
    let i02 = m[0][1] * m[1][2] - m[0][2] * m[1][1];
    let i10 = m[1][2] * m[2][0] - m[1][0] * m[2][2];
    let i11 = m[0][0] * m[2][2] - m[0][2] * m[2][0];
    let i12 = m[0][2] * m[1][0] - m[0][0] * m[1][2];
    let i20 = m[1][0] * m[2][1] - m[1][1] * m[2][0];
    let i21 = m[0][1] * m[2][0] - m[0][0] * m[2][1];
    let i22 = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let det = m[0][0] * i00 + m[0][1] * i10 + m[0][2] * i20;
    let inv_d = 1.0 / det;
    [
        [i00 * inv_d, i01 * inv_d, i02 * inv_d],
        [i10 * inv_d, i11 * inv_d, i12 * inv_d],
        [i20 * inv_d, i21 * inv_d, i22 * inv_d],
    ]
}

#[inline]
fn mul_mv(m: [[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

#[inline]
pub fn gains(temp: f32, tint: f32) -> [f32; 3] {
    let r = (TEMP_STRENGTH * temp).exp();
    let g = (-TINT_STRENGTH * tint).exp();
    let b = (-TEMP_STRENGTH * temp).exp();
    let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    [r / lum, g / lum, b / lum]
}

/// 3x3 linear-sRGB Bradford chromatic adaptation matrix for `(temp, tint)` in `-1..1`.
/// Forms an exact Abelian group homomorphism in cone space (up to luminance normalisation).
pub fn bradford_matrix(temp: f32, tint: f32) -> [[f32; 3]; 3] {
    let srgb_to_bfd = mat_mul(XYZ_TO_BFD, SRGB_TO_XYZ);
    let bfd_to_srgb = mat_inv(srgb_to_bfd);
    let s = [
        (TEMP_STRENGTH as f64 * temp as f64).exp(),
        (-TINT_STRENGTH as f64 * tint as f64).exp(),
        (-TEMP_STRENGTH as f64 * temp as f64).exp(),
    ];
    let diag = [
        [s[0], 0.0, 0.0],
        [0.0, s[1], 0.0],
        [0.0, 0.0, s[2]],
    ];
    let m = mat_mul(bfd_to_srgb, mat_mul(diag, srgb_to_bfd));
    let w = mul_mv(m, [1.0, 1.0, 1.0]);
    let lum = (0.2126 * w[0] + 0.7152 * w[1] + 0.0722 * w[2]).max(1e-9);
    let mut out = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = (m[r][c] / lum) as f32;
        }
    }
    out
}

/// Apply Bradford CAT white balance to sRGB [0..1].
pub fn apply_bradford01(rgb01: [f32; 3], temp: f32, tint: f32) -> [f32; 3] {
    if temp.abs() < 1e-6 && tint.abs() < 1e-6 {
        return rgb01;
    }
    let lin = srgb_to_lin(rgb01);
    let m = bradford_matrix(temp, tint);
    let x = [
        (m[0][0] * lin[0] + m[0][1] * lin[1] + m[0][2] * lin[2]).clamp(0.0, 1.0),
        (m[1][0] * lin[0] + m[1][1] * lin[1] + m[1][2] * lin[2]).clamp(0.0, 1.0),
        (m[2][0] * lin[0] + m[2][1] * lin[1] + m[2][2] * lin[2]).clamp(0.0, 1.0),
    ];
    lin_to_srgb(x)
}

/// Exact inverse of Bradford CAT for the white-balance eyedropper:
/// returns `(dtemp, dtint)` in `-1..1` that neutralises the mean of an RGB8 sample patch.
pub fn solve_eyedropper(patch_rgb8: &[[u8; 3]]) -> (f32, f32) {
    if patch_rgb8.is_empty() {
        return (0.0, 0.0);
    }
    let srgb_to_bfd = mat_mul(XYZ_TO_BFD, SRGB_TO_XYZ);
    let white_bfd = mul_mv(srgb_to_bfd, [1.0, 1.0, 1.0]);

    let mut mean_lin = [0.0f64; 3];
    for px in patch_rgb8 {
        let lin = srgb_to_lin([
            px[0] as f32 / 255.0,
            px[1] as f32 / 255.0,
            px[2] as f32 / 255.0,
        ]);
        mean_lin[0] += lin[0] as f64;
        mean_lin[1] += lin[1] as f64;
        mean_lin[2] += lin[2] as f64;
    }
    let n = patch_rgb8.len() as f64;
    mean_lin[0] /= n;
    mean_lin[1] /= n;
    mean_lin[2] /= n;

    let raw_lms = mul_mv(srgb_to_bfd, mean_lin);
    let l = (raw_lms[0] / white_bfd[0]).max(1e-6);
    let m = (raw_lms[1] / white_bfd[1]).max(1e-6);
    let s = (raw_lms[2] / white_bfd[2]).max(1e-6);

    let temp = ((s / l).ln() / (2.0 * TEMP_STRENGTH as f64)).clamp(-1.0, 1.0) as f32;
    let tint = (-((l * s).sqrt() / m).ln() / (TINT_STRENGTH as f64)).clamp(-1.0, 1.0) as f32;
    (temp, tint)
}
