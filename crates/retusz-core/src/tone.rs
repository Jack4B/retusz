//! Photographic tone, colour, and local contrast engine in linear light & OKLab (Björn Ottosson 2020/2021).
//! Author: Jack4B.
//!
//! - Exposure (`-2.0..+2.0 EV`): linear-light `2^EV` with chromaticity-preserving filmic highlight shoulder above 0.82.
//! - Contrast (`-1.0..+1.0`): monotonic S-curve in OKLab `L` around 18% mid-grey (`L = 0.58`), applied as a
//!   linear luminance ratio so skin chroma never shifts.
//! - Highlights, Shadows, Whites, Blacks: independent tonal zones in OKLab `L`.
//! - Vibrance & Saturation in OKLab with constant-hue, constant-lightness gamut clipping (`0 deg` hue drift).
//! - Clarity (`-1.0..+1.0`): halo-free guided-filter mid-tone local contrast, skin-protected.

use crate::color::{guided_filter, rgb01_to_lab};

/// Linear sRGB -> OKLab `[L, a, b]` (Ottosson 2020).
#[inline]
pub fn lin_to_oklab(lin: [f32; 3]) -> [f32; 3] {
    let r = lin[0].max(0.0);
    let g = lin[1].max(0.0);
    let b = lin[2].max(0.0);
    let l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b).cbrt();
    let m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b).cbrt();
    let s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b).cbrt();
    [
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    ]
}

/// OKLab `[L, a, b]` -> raw Linear sRGB (unclamped).
#[inline]
pub fn oklab_to_lin_raw(lab: [f32; 3]) -> [f32; 3] {
    let (l_, a, b) = (lab[0], lab[1], lab[2]);
    let l = l_ + 0.3963377774 * a + 0.2158037573 * b;
    let m = l_ - 0.1055613458 * a - 0.0638541728 * b;
    let s = l_ - 0.0894841775 * a - 1.2914855480 * b;
    let (l3, m3, s3) = (l * l * l, m * m * m, s * s * s);
    [
         4.0767416621 * l3 - 3.3077115913 * m3 + 0.2309699292 * s3,
        -1.2684380046 * l3 + 2.6097574011 * m3 - 0.3413193965 * s3,
        -0.0041960863 * l3 - 0.7034186147 * m3 + 1.7076147010 * s3,
    ]
}

/// OKLab -> Linear sRGB [0..1] with constant-hue, constant-lightness gamut clipping (Ottosson 2021).
/// Scales only OKLab chroma `(a, b)` via 9-step bisection when out of `[0, 1]^3`, achieving `0 deg` hue drift.
#[inline]
pub fn oklab_to_lin_gamut(mut lab: [f32; 3]) -> [f32; 3] {
    lab[0] = lab[0].clamp(0.0, 1.0);
    let raw = oklab_to_lin_raw(lab);
    if raw[0] >= 0.0 && raw[0] <= 1.0 && raw[1] >= 0.0 && raw[1] <= 1.0 && raw[2] >= 0.0 && raw[2] <= 1.0 {
        return raw;
    }
    let mut lo = 0.0f32;
    let mut hi = 1.0f32;
    for _ in 0..9 {
        let mid = 0.5 * (lo + hi);
        let c = oklab_to_lin_raw([lab[0], lab[1] * mid, lab[2] * mid]);
        if c[0] >= -1e-5 && c[0] <= 1.00001 && c[1] >= -1e-5 && c[1] <= 1.00001 && c[2] >= -1e-5 && c[2] <= 1.00001 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let out = oklab_to_lin_raw([lab[0], lab[1] * lo, lab[2] * lo]);
    [out[0].clamp(0.0, 1.0), out[1].clamp(0.0, 1.0), out[2].clamp(0.0, 1.0)]
}

/// Exposure in EV (`-2.0..+2.0`) in linear light with chromaticity-preserving filmic highlight shoulder.
#[inline]
pub fn apply_exposure_lin(lin: [f32; 3], ev: f32) -> [f32; 3] {
    if ev.abs() < 1e-6 {
        return lin;
    }
    let g = 2.0_f32.powf(ev);
    let mut x = [lin[0] * g, lin[1] * g, lin[2] * g];
    if ev > 0.0 {
        let knee = 0.82f32;
        let span = 1.0 - knee;
        let peak = x[0].max(x[1]).max(x[2]);
        if peak > knee {
            let excess = (peak - knee) / span;
            let comp = knee + span * (excess / (1.0 + excess + 0.25 * excess * excess));
            let scale = comp / peak.max(1e-8);
            x[0] *= scale;
            x[1] *= scale;
            x[2] *= scale;
        }
    }
    [x[0].clamp(0.0, 1.0), x[1].clamp(0.0, 1.0), x[2].clamp(0.0, 1.0)]
}

/// Perceptual tonal curve in OKLab lightness `L`, applied as a linear-light luminance ratio.
#[inline]
pub fn apply_tone_lin(
    lin: [f32; 3],
    contrast: f32,
    highlights: f32,
    shadows: f32,
    whites: f32,
    blacks: f32,
) -> [f32; 3] {
    if contrast.abs() < 1e-6
        && highlights.abs() < 1e-6
        && shadows.abs() < 1e-6
        && whites.abs() < 1e-6
        && blacks.abs() < 1e-6
    {
        return lin;
    }
    let y0 = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2];
    let l0 = y0.max(0.0).cbrt();
    let mut l = l0;

    if contrast.abs() > 1e-6 {
        let p = 0.58f32; // 18% linear mid-grey in OKLab L
        let mut w = 4.0 * l * (1.0 - l);
        w = w.clamp(0.0, 1.0).powf(0.85);
        l = (l + 0.38 * contrast * (l - p) * w).clamp(0.0, 1.0);
    }
    if shadows.abs() > 1e-6 {
        let ws = (1.0 - l).powf(1.8) * (4.0 * l).clamp(0.0, 1.0);
        l += 0.16 * shadows * ws;
    }
    if highlights.abs() > 1e-6 {
        let wh = l.powf(1.8) * (4.0 * (1.0 - l)).clamp(0.0, 1.0);
        l += 0.16 * highlights * wh;
    }
    if blacks.abs() > 1e-6 {
        let wb = ((0.55 - l) / 0.55).clamp(0.0, 1.0).powi(2);
        l += 0.15 * blacks * wb;
    }
    if whites.abs() > 1e-6 {
        let ww = ((l - 0.45) / 0.55).clamp(0.0, 1.0).powi(2);
        let d = 0.14 * whites * ww;
        if d > 0.0 {
            let room = (1.0 - l).max(1e-6);
            l += room * (1.0 - (-d / room).exp());
        } else {
            l += d;
        }
    }
    l = l.clamp(0.0, 1.0);
    let y1 = l * l * l;
    let ratio = y1 / (y0 + 1e-6);
    let out = [lin[0] * ratio, lin[1] * ratio, lin[2] * ratio];
    let peak = out[0].max(out[1]).max(out[2]);
    if peak > 1.0 {
        let lab = lin_to_oklab([out[0].max(0.0), out[1].max(0.0), out[2].max(0.0)]);
        return oklab_to_lin_gamut(lab);
    }
    [out[0].clamp(0.0, 1.0), out[1].clamp(0.0, 1.0), out[2].clamp(0.0, 1.0)]
}

/// Saturation and skin-sparing vibrance in OKLab with constant-hue gamut clipping.
#[inline]
pub fn apply_color_oklab(lin: [f32; 3], saturation: f32, vibrance: f32) -> [f32; 3] {
    if saturation.abs() < 1e-6 && vibrance.abs() < 1e-6 {
        return lin;
    }
    let mut lab = lin_to_oklab(lin);
    let (l, mut a, mut b) = (lab[0], lab[1], lab[2]);
    let c = a.hypot(b);
    let cn = (c / 0.32).clamp(0.0, 1.0);

    let h = b.atan2(a);
    let skin_angle = ((h - 0.72).sin() * 0.5).atan2((h - 0.72).cos()); // smooth wrap
    let skin_h = (-0.5 * (skin_angle / 0.48).powi(2)).exp();
    let skin_l = (-0.5 * ((l - 0.70) / 0.22).powi(2)).exp();
    let skin_prot = 0.65 * skin_h * skin_l;

    let mut g = 1.0 + saturation;
    if vibrance.abs() > 1e-6 {
        let w_vib = if vibrance >= 0.0 {
            (1.0 - cn) * (1.0 - skin_prot)
        } else {
            1.0
        };
        g *= 1.0 + vibrance * w_vib;
    }
    g = g.max(0.0);
    a *= g;
    b *= g;
    lab[1] = a;
    lab[2] = b;
    oklab_to_lin_gamut(lab)
}

/// OKLCh skin-locus cast reduction (`skin_cast` in `0.0..1.0`) and skin hue control (`skin_hue` in `-1.0..1.0`).
#[inline]
pub fn apply_skin_oklch(lin: [f32; 3], skin_cast: f32, skin_hue: f32) -> [f32; 3] {
    if skin_cast.abs() < 1e-6 && skin_hue.abs() < 1e-6 {
        return lin;
    }
    let ok = lin_to_oklab(lin);
    let (l, a, b) = (ok[0], ok[1], ok[2]);
    let c = a.hypot(b);
    let h = b.atan2(a);
    let h_target = 50.0f32.to_radians();
    let dh = (h - h_target).sin().atan2((h - h_target).cos());
    let w_locus = (-(dh / 1.15).powi(4)).exp() * (c / 0.015).clamp(0.0, 1.0);
    let h_new = h - skin_cast * 0.88 * w_locus * dh + skin_hue * 0.16 * w_locus;
    let c_excess = (c - 0.11).max(0.0);
    let c_new = c - skin_cast.max(0.0) * 0.35 * w_locus * c_excess;
    oklab_to_lin_gamut([l, c_new * h_new.cos(), c_new * h_new.sin()])
}

/// Dedicated OKLCh Skin Fill Light (`skin_fill` in `-1.0..1.0`).
#[inline]
pub fn apply_skin_fill_oklch(lin: [f32; 3], skin_fill: f32) -> [f32; 3] {
    let sf = skin_fill.clamp(-1.0, 1.0);
    if sf.abs() < 1e-5 {
        return lin;
    }
    let ok = lin_to_oklab(lin);
    let l = ok[0].clamp(0.0, 1.0);
    let c = ok[1].hypot(ok[2]);
    let h = ok[2].atan2(ok[1]);
    let dh = (h - 48.0f32.to_radians()).sin().atan2((h - 48.0f32.to_radians()).cos());
    let w_hue = (-(dh / 1.05).powi(4)).exp();
    let w_dark = ((l - 0.20) / 0.11).clamp(0.0, 1.0);
    let w_loc = w_dark * (0.35 + 0.65 * w_hue) * (c / 0.015).clamp(0.0, 1.0);
    let (l_new, c_new) = if sf > 0.0 {
        let t_hi = ((0.91 - l) / 0.32).clamp(0.0, 1.0);
        let w_hi = t_hi * t_hi * (3.0 - 2.0 * t_hi);
        let d = 0.175 * sf * w_loc * w_hi;
        let room = (0.96 - l).max(1e-4);
        let ln = (l + d.min(room * (1.0 - (-d / room).exp()))).clamp(0.0, 0.985);
        let kc = (ln / l.max(0.20)).max(1.0).powf(0.50).clamp(1.0, 1.20);
        let hi_tame = 1.0 - 0.35 * ((c - 0.095) / 0.055).clamp(0.0, 1.0);
        (ln, c * (1.0 + (kc - 1.0) * hi_tame))
    } else {
        let w_neg = w_loc * ((l - 0.28) / 0.42).clamp(0.0, 1.0).powf(1.10) * (1.0 - l.powi(12));
        let ln = (l + 0.18 * sf * w_neg).clamp(0.0, 1.0);
        let c_rec = 1.0 + 0.08 * (-sf) * w_neg * ((0.09 - c) / 0.06).clamp(0.0, 1.0);
        (ln, c * c_rec)
    };
    oklab_to_lin_gamut([l_new, c_new * h.cos(), c_new * h.sin()])
}

/// Halo-free mid-tone local contrast (Clarity, `-1.0..+1.0`) via Guided Filter on CIELAB `L*`.
/// Returns a relative luminance gain map `g` (`w * h`) for linear light: `L_lin_out = L_lin * (1 + g)`.
pub fn clarity_gain(rgb8: &[u8], w: usize, h: usize, skin_mask: Option<&[f32]>, clarity: f32) -> Vec<f32> {
    let n = w * h;
    if clarity.abs() < 1e-3 || n == 0 {
        return vec![0.0f32; n];
    }
    let mut l_star = vec![0.0f32; n];
    for i in 0..n {
        let rgb = [
            rgb8[i * 3] as f32 / 255.0,
            rgb8[i * 3 + 1] as f32 / 255.0,
            rgb8[i * 3 + 2] as f32 / 255.0,
        ];
        l_star[i] = rgb01_to_lab(rgb)[0];
    }
    let r = (w.max(h) / 48).max(2);
    let base = guided_filter(&l_star, &l_star, w, h, r, 64.0);
    let mut gain = vec![0.0f32; n];
    for i in 0..n {
        let l = l_star[i];
        let d = l - base[i];
        let d_soft = 10.0 * (d / 10.0).tanh();
        let mid = (4.0 * (l / 100.0) * (1.0 - l / 100.0)).clamp(0.0, 1.0);
        let sk = skin_mask.map(|m| m[i].clamp(0.0, 1.0)).unwrap_or(0.0);
        let sw = if clarity > 0.0 { 1.0 - 0.72 * sk } else { 1.0 - 0.25 * sk };
        let dl = 0.55 * clarity * d_soft * mid * sw;
        gain[i] = (3.0 * dl / l.max(25.0)).clamp(-0.35, 0.35);
    }
    gain
}
