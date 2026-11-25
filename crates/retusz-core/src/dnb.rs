//! Face bounding boxes, adaptive cheek/forehead core skin calibration, andDodge & Burn v2
//! (weighted guided filter on L* at face=256px scale + redescending Huber/Tukey weight + 3-channel
//! RGB guided edge snapping + linear-light luminance application).
//! Author: Jack4B.

use crate::color::{box_filter, guided_refine_rgb, rgb01_to_lab};
use rayon::prelude::*;

/// Face bounding box `[x, y, w, h]` in pixels.
pub type FaceBox = [f32; 4];

/// Compute a clean skin mask (`w * h` in `0.0..1.0`) calibrated on the inner cheek/forehead core of each face.
/// Rejects dark brown hair (adaptive luminance floor) and skin-coloured background patches, then snaps to
/// 3-channel RGB edges via `guided_refine_rgb`.
pub fn skin_mask(rgb8: &[u8], w: usize, h: usize, faces: &[FaceBox]) -> Vec<f32> {
    let n = w * h;
    if faces.is_empty() || n == 0 {
        return vec![0.0f32; n];
    }

    let mut y_ch = vec![0.0f32; n];
    let mut cr_ch = vec![0.0f32; n];
    let mut cb_ch = vec![0.0f32; n];
    let mut rgb01 = vec![0.0f32; n * 3];

    for i in 0..n {
        let r = rgb8[i * 3] as f32;
        let g = rgb8[i * 3 + 1] as f32;
        let b = rgb8[i * 3 + 2] as f32;
        rgb01[i * 3] = r / 255.0;
        rgb01[i * 3 + 1] = g / 255.0;
        rgb01[i * 3 + 2] = b / 255.0;
        let yv = 0.299 * r + 0.587 * g + 0.114 * b;
        y_ch[i] = yv;
        cr_ch[i] = 128.0 + 0.713 * (r - yv);
        cb_ch[i] = 128.0 + 0.564 * (b - yv);
    }

    let mut mask = vec![0.0f32; n];

    for &[fx, fy, fw, fh] in faces {
        if fw < 8.0 || fh < 8.0 {
            continue;
        }
        let cx = fx + 0.5 * fw;
        let cy = fy + 0.52 * fh;
        let rx_core = (0.38 * fw).max(4.0);
        let ry_core = (0.44 * fh).max(4.0);

        // Gather broad YCrCb skin candidates inside the inner face core
        let x0 = ((cx - rx_core).floor() as isize).max(0) as usize;
        let x1 = ((cx + rx_core).ceil() as usize).min(w);
        let y0 = ((cy - ry_core).floor() as isize).max(0) as usize;
        let y1 = ((cy + ry_core).ceil() as usize).min(h);

        let mut core_ys = Vec::new();
        let mut core_pts = Vec::new();
        for yy in y0..y1 {
            for xx in x0..x1 {
                let dx = (xx as f32 - cx) / rx_core;
                let dy = (yy as f32 - cy) / ry_core;
                if dx * dx + dy * dy <= 1.0 {
                    let idx = yy * w + xx;
                    let (yv, cr, cb) = (y_ch[idx], cr_ch[idx], cb_ch[idx]);
                    if (130.0..=182.0).contains(&cr) && (75.0..=135.0).contains(&cb) && yv >= 16.0 {
                        core_ys.push(yv);
                        core_pts.push(idx);
                    }
                }
            }
        }
        if core_ys.len() < 12 {
            continue;
        }
        core_ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let y_p25 = core_ys[core_ys.len() / 4];
        let y_med = core_ys[(core_ys.len() * 5) / 8];
        let mut devs: Vec<f32> = core_ys[core_ys.len() / 4..]
            .iter()
            .map(|&v| (v - y_med).abs())
            .collect();
        devs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let y_mad = devs[devs.len() / 2].max(6.0);
        let y_lo = (y_med - (3.2 * y_mad).max(42.0)).max(18.0);
        let y_hi = (y_med + (3.8 * y_mad).max(70.0)).min(253.0);

        let mut cr_sum = 0.0f32;
        let mut cb_sum = 0.0f32;
        let mut cnt = 0.0f32;
        for &idx in &core_pts {
            if y_ch[idx] >= y_p25 {
                cr_sum += cr_ch[idx];
                cb_sum += cb_ch[idx];
                cnt += 1.0;
            }
        }
        let cr_mu = cr_sum / cnt.max(1.0);
        let cb_mu = cb_sum / cnt.max(1.0);
        let mut v_cr = 16.0f32;
        let mut v_cb = 16.0f32;
        for &idx in &core_pts {
            if y_ch[idx] >= y_p25 {
                v_cr += (cr_ch[idx] - cr_mu).powi(2);
                v_cb += (cb_ch[idx] - cb_mu).powi(2);
            }
        }
        v_cr = (v_cr / cnt.max(1.0)).max(18.0);
        v_cb = (v_cb / cnt.max(1.0)).max(18.0);

        // Evaluate inside face + neck ellipse
        let rx_full = (0.72 * fw).max(6.0);
        let ry_full = (0.95 * fh).max(6.0);
        let cy_full = fy + 0.62 * fh;
        let fx0 = ((cx - rx_full).floor() as isize).max(0) as usize;
        let fx1 = ((cx + rx_full).ceil() as usize).min(w);
        let fy0 = ((cy_full - ry_full).floor() as isize).max(0) as usize;
        let fy1 = ((cy_full + ry_full).ceil() as usize).min(h);

        for yy in fy0..fy1 {
            for xx in fx0..fx1 {
                let dx = (xx as f32 - cx) / rx_full;
                let dy = (yy as f32 - cy_full) / ry_full;
                let r2 = dx * dx + dy * dy;
                if r2 > 1.0 {
                    continue;
                }
                let idx = yy * w + xx;
                let (yv, cr, cb) = (y_ch[idx], cr_ch[idx], cb_ch[idx]);
                if yv < y_lo || yv > y_hi {
                    continue;
                }
                if !(130.0..=182.0).contains(&cr) || !(75.0..=135.0).contains(&cb) {
                    continue;
                }
                let md2 = (cr - cr_mu).powi(2) / v_cr + (cb - cb_mu).powi(2) / v_cb;
                if md2 <= 12.5 {
                    let edge_w = ((1.0 - r2) * 3.0).clamp(0.0, 1.0);
                    let col_w = (1.0 - md2 / 14.0).clamp(0.0, 1.0);
                    mask[idx] = mask[idx].max(edge_w * col_w);
                }
            }
        }
    }

    let r_ref = (w.max(h) / 90).max(2);
    guided_refine_rgb(&rgb01, &mask, w, h, r_ref, 1e-3)
}

/// Weighted Guided Filter (He et al. 2010 with skin weights `w_s` in both box passes)
/// for Dodge & Burn v2 base layer estimation.
pub fn weighted_guided(l_star: &[f32], w_s: &[f32], w: usize, h: usize, r: usize, eps: f32) -> Vec<f32> {
    let n = w * h;
    let ws: Vec<f32> = w_s.iter().map(|&v| v.max(1e-4)).collect();
    let mw = box_filter(&ws, w, h, r);
    let wmean = |x: &[f32]| -> Vec<f32> {
        let wx: Vec<f32> = x.iter().zip(ws.iter()).map(|(&a, &b)| a * b).collect();
        let mwx = box_filter(&wx, w, h, r);
        mwx.iter().zip(mw.iter()).map(|(&num, &den)| num / den.max(1e-6)).collect()
    };
    let ml = wmean(l_star);
    let l2: Vec<f32> = l_star.iter().map(|&v| v * v).collect();
    let ml2 = wmean(&l2);
    let mut a = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    for i in 0..n {
        let var = (ml2[i] - ml[i] * ml[i]).max(0.0);
        let ai = var / (var + eps);
        a[i] = ai;
        b[i] = (1.0 - ai) * ml[i];
    }
    let ma = wmean(&a);
    let mb = wmean(&b);
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = ma[i] * l_star[i] + mb[i];
    }
    out
}

/// Dodge & Burn v2: smooths mid-frequency skin blotches while preserving pores (high frequency)
/// and natural facial structure (nose shadow, glasses, eyes, lips) via redescending weights
/// and linear-light luminance scaling (no orange saturation shift when darkening).
pub fn dodge_burn_v2(
    rgb8: &[u8],
    w: usize,
    h: usize,
    skin: &[f32],
    faces: &[FaceBox],
    strength: f32,
) -> Vec<u8> {
    if strength <= 1e-3 || faces.is_empty() || w == 0 || h == 0 {
        return rgb8.to_vec();
    }
    let n = w * h;
    let mut l_star = vec![0.0f32; n];
    for i in 0..n {
        let rgb = [
            rgb8[i * 3] as f32 / 255.0,
            rgb8[i * 3 + 1] as f32 / 255.0,
            rgb8[i * 3 + 2] as f32 / 255.0,
        ];
        l_star[i] = rgb01_to_lab(rgb)[0];
    }

    let fw = faces.iter().map(|f| f[2]).fold(80.0f32, f32::max);
    let r_fine = (fw / 85.0).round().max(1.0) as usize;
    let r_base = (fw / 13.0).round().max(3.0) as usize;
    let eps = 48.0f32;

    // Fine pore preservation layer + two-pass redescending weighted guided base
    let l_smooth = box_filter(&l_star, w, h, r_fine);
    let base1 = weighted_guided(&l_smooth, skin, w, h, r_base, eps);
    let mut w2 = vec![0.0f32; n];
    for i in 0..n {
        let r = (l_smooth[i] - base1[i]) / 7.5;
        let tukey = if r.abs() < 1.0 { (1.0 - r * r).powi(2) } else { 0.03 };
        w2[i] = skin[i] * tukey;
    }
    let base2 = weighted_guided(&l_smooth, &w2, w, h, r_base, eps);

    let mut out = vec![0u8; rgb8.len()];
    out.par_chunks_mut(3)
        .zip(rgb8.par_chunks(3))
        .enumerate()
        .for_each(|(i, (dst, src))| {
            let sk = w2[i];
            if sk <= 1e-3 {
                dst.copy_from_slice(src);
                return;
            }
            let l0 = l_star[i];
            let blotch = base2[i] - l_smooth[i];
            let dl = 10.0 * (0.72 * strength * sk * blotch / 10.0).tanh();
            let g = (3.0 * dl / l0.max(25.0)).clamp(-0.38, 0.38);
            // Apply in linear light so chromaticity / saturation stays invariant
            for c in 0..3 {
                let s = src[c] as f32 / 255.0;
                let lin = crate::color::srgb_to_lin_scalar(s) * (1.0 + g);
                let sr = crate::color::lin_to_srgb_scalar(lin.clamp(0.0, 1.0));
                dst[c] = (sr * 255.0 + 0.5) as u8;
            }
        });
    out
}
