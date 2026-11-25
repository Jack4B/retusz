//! Full studio rendering pipeline: builds a 33^3 3D LUT from photographic tone/colour/WB/reference
//! parameters, applies Dodge & Burn v2, Plastyka 3D, Clarity, Vignette, and non-destructive crop/straighten.
//! Works both standalone (no reference image required) and with optional reference colour matching (MKL/Reinhard).
//! Author: Jack4B.

use crate::color::{lab_to_rgb01, lin_to_srgb, rgb01_to_lab, srgb_to_lin, Lut3D};
use crate::dnb::{dodge_burn_v2, skin_mask, FaceBox};
use crate::geometry::clamp_crop;
use crate::params::Params;
use crate::relight::{relight_gain, subject_mask};
use crate::tone::{
    apply_color_oklab, apply_exposure_lin, apply_skin_oklch, apply_tone_lin, clarity_gain,
};
use crate::wb::apply_bradford01;
use rayon::prelude::*;

/// Reference colour statistics (mean & std of CIELAB L*, a*, b*) for optional reference matching.
#[derive(Debug, Clone, Copy)]
pub struct ReferenceStats {
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl ReferenceStats {
    pub fn from_rgb8(rgb8: &[u8]) -> Self {
        let n = (rgb8.len() / 3).max(1);
        let mut sum = [0.0f64; 3];
        let mut sum2 = [0.0f64; 3];
        for px in rgb8.chunks_exact(3) {
            let lab = rgb01_to_lab([
                px[0] as f32 / 255.0,
                px[1] as f32 / 255.0,
                px[2] as f32 / 255.0,
            ]);
            for c in 0..3 {
                let v = lab[c] as f64;
                sum[c] += v;
                sum2[c] += v * v;
            }
        }
        let inv = 1.0 / n as f64;
        let mean = [
            (sum[0] * inv) as f32,
            (sum[1] * inv) as f32,
            (sum[2] * inv) as f32,
        ];
        let std = [
            ((sum2[0] * inv - (sum[0] * inv).powi(2)).max(1.0).sqrt()) as f32,
            ((sum2[1] * inv - (sum[1] * inv).powi(2)).max(1.0).sqrt()) as f32,
            ((sum2[2] * inv - (sum[2] * inv).powi(2)).max(1.0).sqrt()) as f32,
        ];
        Self { mean, std }
    }
}

/// Build the 33^3 3D LUT encoding white balance (Bradford CAT), optional reference match,
/// exposure EV, OKLab contrast/highlights/shadows/whites/blacks, OKLCh skin cast/hue, and OKLab vibrance/saturation.
pub fn build_studio_lut(
    params: &Params,
    src_stats: Option<ReferenceStats>,
    ref_stats: Option<ReferenceStats>,
) -> Lut3D {
    let mut lut = Lut3D::identity(33);
    let has_ref = src_stats.is_some() && ref_stats.is_some() && params.strength > 1e-4;

    lut.data.par_iter_mut().for_each(|rgb| {
        let s0 = *rgb;
        // 1. Optional reference colour matching (bonus collapsible section)
        let mut s = if has_ref {
            let ss = src_stats.unwrap();
            let rs = ref_stats.unwrap();
            let lab = rgb01_to_lab(s0);
            let mut matched = [0.0f32; 3];
            for c in 0..3 {
                let gain = (rs.std[c] / ss.std[c].max(1.0)).clamp(0.65, 1.45);
                matched[c] = (lab[c] - ss.mean[c]) * gain + rs.mean[c];
            }
            let m_rgb = lab_to_rgb01(matched);
            let k = params.strength.clamp(0.0, 1.0);
            [
                s0[0] + k * (m_rgb[0] - s0[0]),
                s0[1] + k * (m_rgb[1] - s0[1]),
                s0[2] + k * (m_rgb[2] - s0[2]),
            ]
        } else {
            s0
        };

        // 2. CIE/ICC Bradford Chromatic Adaptation white balance
        s = apply_bradford01(s, params.temp, params.tint);

        // 3. Linear-light Exposure (EV) + OKLab tone curve + OKLCh skin locus + OKLab vibrance/saturation
        let mut lin = srgb_to_lin(s);
        lin = apply_exposure_lin(lin, params.exposure);
        lin = apply_tone_lin(
            lin,
            params.contrast,
            params.highlights,
            params.shadows,
            params.whites,
            params.blacks,
        );
        if params.skin_cast.abs() > 1e-5 || params.skin_hue.abs() > 1e-5 {
            lin = apply_skin_oklch(lin, params.skin_cast, params.skin_hue);
        }
        let vib = params.vibrance + 0.25 * params.wide;
        let sat = params.saturation + 0.12 * params.wide;
        lin = apply_color_oklab(lin, sat, vib);
        *rgb = lin_to_srgb(lin);
    });

    if has_ref {
        lut.smooth();
    }
    lut
}

/// Render a photo (`w * h * 3` RGB8) through the full studio pipeline.
pub fn render_photo_cpu(
    rgb8: &[u8],
    w: usize,
    h: usize,
    params: &Params,
    faces: &[FaceBox],
    depth: Option<&[f32]>,
    ref_stats: Option<ReferenceStats>,
) -> (Vec<u8>, usize, usize) {
    if w == 0 || h == 0 || rgb8.len() != w * h * 3 {
        return (rgb8.to_vec(), w, h);
    }
    let n = w * h;
    let skin = if !faces.is_empty() {
        skin_mask(rgb8, w, h, faces)
    } else {
        vec![0.0f32; n]
    };

    // 1. Dodge & Burn v2 on skin before 3D LUT
    let base_rgb = if params.db > 1e-3 && !faces.is_empty() {
        dodge_burn_v2(rgb8, w, h, &skin, faces, params.db)
    } else {
        rgb8.to_vec()
    };

    // 2. Plastyka 3D + Clarity spatial luminance gain map
    let mut gain = clarity_gain(&base_rgb, w, h, Some(&skin), params.clarity);
    if params.plastic > 1e-3 {
        if let Some(d_map) = depth {
            if d_map.len() == n {
                let faces_norm: Vec<FaceBox> = faces
                    .iter()
                    .map(|f| [f[0] / w as f32, f[1] / h as f32, f[2] / w as f32, f[3] / h as f32])
                    .collect();
                let fg = subject_mask(d_map, &skin, &faces_norm, &base_rgb, w, h);
                let mut l_star = vec![0.0f32; n];
                let mut rgb01 = vec![0.0f32; n * 3];
                for i in 0..n {
                    let c = [
                        base_rgb[i * 3] as f32 / 255.0,
                        base_rgb[i * 3 + 1] as f32 / 255.0,
                        base_rgb[i * 3 + 2] as f32 / 255.0,
                    ];
                    rgb01[i * 3] = c[0];
                    rgb01[i * 3 + 1] = c[1];
                    rgb01[i * 3 + 2] = c[2];
                    l_star[i] = rgb01_to_lab(c)[0];
                }
                let angle = params.light_angle.unwrap_or(120.0);
                let p_gain = relight_gain(d_map, &fg, &l_star, Some(&rgb01), w, h, params.plastic, angle);
                for i in 0..n {
                    gain[i] += p_gain[i];
                }
            }
        }
    }

    // 3. Build & apply 33^3 tetrahedral LUT + spatial gain + vignette in parallel
    let src_stats = ref_stats.map(|_| ReferenceStats::from_rgb8(&base_rgb));
    let lut = build_studio_lut(params, src_stats, ref_stats);
    let vig = params.vignette;

    let mut out = vec![0u8; n * 3];
    out.par_chunks_mut(w * 3)
        .zip(base_rgb.par_chunks(w * 3))
        .enumerate()
        .for_each(|(y, (dst_row, src_row))| {
            let ny = (y as f32 / h.max(1) as f32) - 0.5;
            for x in 0..w {
                let nx = (x as f32 / w.max(1) as f32) - 0.5;
                let idx = y * w + x;
                let s = [
                    src_row[x * 3] as f32 / 255.0,
                    src_row[x * 3 + 1] as f32 / 255.0,
                    src_row[x * 3 + 2] as f32 / 255.0,
                ];
                let mut c = lut.sample_tetra(s);
                let mut g = 1.0 + gain[idx];
                if vig.abs() > 1e-4 {
                    let r2 = 2.2 * (nx * nx + ny * ny);
                    g *= (1.0 + 0.55 * vig * r2 * r2).clamp(0.25, 1.75);
                }
                if (g - 1.0).abs() > 1e-4 {
                    let lin = srgb_to_lin(c);
                    c = lin_to_srgb([
                        (lin[0] * g).clamp(0.0, 1.0),
                        (lin[1] * g).clamp(0.0, 1.0),
                        (lin[2] * g).clamp(0.0, 1.0),
                    ]);
                }
                dst_row[x * 3] = (c[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                dst_row[x * 3 + 1] = (c[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                dst_row[x * 3 + 2] = (c[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        });

    // 4. Optional non-destructive crop
    if let Some(raw_c) = params.crop {
        let c = clamp_crop(raw_c, w as f32, h as f32, params.angle, None);
        let x0 = ((c[0] * w as f32).round() as usize).min(w - 1);
        let y0 = ((c[1] * h as f32).round() as usize).min(h - 1);
        let x1 = ((c[2] * w as f32).round() as usize).clamp(x0 + 1, w);
        let y1 = ((c[3] * h as f32).round() as usize).clamp(y0 + 1, h);
        if x0 > 0 || y0 > 0 || x1 < w || y1 < h {
            let cw = x1 - x0;
            let ch = y1 - y0;
            let mut cropped = vec![0u8; cw * ch * 3];
            for yy in 0..ch {
                let src_off = ((y0 + yy) * w + x0) * 3;
                let dst_off = yy * cw * 3;
                cropped[dst_off..dst_off + cw * 3].copy_from_slice(&out[src_off..src_off + cw * 3]);
            }
            return (cropped, cw, ch);
        }
    }
    (out, w, h)
}
