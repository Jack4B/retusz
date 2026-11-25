//! Plastyka 3D (3D portrait relighting & sculpting):
//! - Subject mask segmentation anchored to face depth, Otsu valley threshold, colour-coherent foreground
//!   check, and 3-channel RGB Guided Filter edge snapping (never darkens or sculpts the background).
//! - Surface normals from person-only depth + zero-centred directional Lambertian shading (preserves
//!   mean subject exposure while sculpting 3D relief) + halo-free guided micro-contrast.
//! Author: Jack4B.

use crate::color::{box_filter, guided_filter, guided_refine_rgb};
use crate::dnb::FaceBox;

/// Compute Otsu valley midpoint threshold on a depth map `depth` (`0..1`).
pub fn otsu_valley_threshold(depth: &[f32]) -> f32 {
    if depth.is_empty() {
        return 0.5;
    }
    let mut hist = [0u32; 256];
    for &d in depth {
        let bin = (d.clamp(0.0, 1.0) * 255.0 + 0.5) as usize;
        hist[bin.min(255)] += 1;
    }
    let total = depth.len() as f64;
    let mut sum_all = 0.0f64;
    for (i, &c) in hist.iter().enumerate() {
        sum_all += (i as f64) * (c as f64);
    }
    let mut w_b = 0.0f64;
    let mut sum_b = 0.0f64;
    let mut max_var = -1.0f64;
    let mut best_t = 128usize;

    for (t, &c) in hist.iter().enumerate() {
        w_b += c as f64;
        if w_b <= 0.0 { continue; }
        let w_f = total - w_b;
        if w_f <= 0.0 { break; }
        sum_b += (t as f64) * (c as f64);
        let m_b = sum_b / w_b;
        let m_f = (sum_all - sum_b) / w_f;
        let var_between = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if var_between > max_var {
            max_var = var_between;
            best_t = t;
        }
    }
    // Midpoint across flat valley
    let mut hi_t = best_t;
    while hi_t + 1 < 255 && hist[hi_t + 1] == 0 {
        hi_t += 1;
    }
    0.5 * (best_t + hi_t) as f32 / 255.0
}

/// Segment the person (`w * h` in `0..1`) from a depth map (`0..1`), skin mask, faces, and RGB8 image.
/// Rejects background lobes that the depth model reads as near by colour-coherence against the
/// sure background vs sure foreground, then snaps the boundary with a 3-channel RGB Guided Filter.
pub fn subject_mask(
    depth: &[f32],
    skin: &[f32],
    faces_norm: &[FaceBox],
    rgb8: &[u8],
    w: usize,
    h: usize,
) -> Vec<f32> {
    let n = w * h;
    if n == 0 || depth.len() != n {
        return vec![0.0f32; n];
    }
    let t_otsu = otsu_valley_threshold(depth);

    // Anchor depth threshold to face median depth if faces are present
    let mut face_depths = Vec::new();
    for &[fx, fy, fw, fh] in faces_norm {
        let x0 = ((fx + 0.2 * fw) * w as f32).clamp(0.0, (w - 1) as f32) as usize;
        let x1 = ((fx + 0.8 * fw) * w as f32).clamp((x0 + 1) as f32, w as f32) as usize;
        let y0 = ((fy + 0.2 * fh) * h as f32).clamp(0.0, (h - 1) as f32) as usize;
        let y1 = ((fy + 0.8 * fh) * h as f32).clamp((y0 + 1) as f32, h as f32) as usize;
        for yy in y0..y1 {
            for xx in x0..x1 {
                face_depths.push(depth[yy * w + xx]);
            }
        }
    }
    let t_depth = if !face_depths.is_empty() {
        face_depths.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let d_face = face_depths[face_depths.len() / 2];
        t_otsu.max(d_face - 0.30).min(d_face - 0.06)
    } else {
        t_otsu
    };

    // Colour-coherence check against sure background (d <= t_otsu) vs sure foreground (d >= t_depth + 0.14 or skin > 0.5)
    let mut bg_rgb = [0.0f64; 3];
    let mut bg_cnt = 0.0f64;
    let mut fg_rgb = [0.0f64; 3];
    let mut fg_cnt = 0.0f64;
    for i in 0..n {
        let d = depth[i];
        let sk = skin.get(i).copied().unwrap_or(0.0);
        let r = rgb8[i * 3] as f64 / 255.0;
        let g = rgb8[i * 3 + 1] as f64 / 255.0;
        let b = rgb8[i * 3 + 2] as f64 / 255.0;
        if d <= t_otsu + 0.02 && sk < 0.1 {
            bg_rgb[0] += r; bg_rgb[1] += g; bg_rgb[2] += b;
            bg_cnt += 1.0;
        } else if d >= (t_depth + 0.14).min(0.75) || sk > 0.55 {
            fg_rgb[0] += r; fg_rgb[1] += g; fg_rgb[2] += b;
            fg_cnt += 1.0;
        }
    }
    if bg_cnt > 20.0 {
        for c in &mut bg_rgb { *c /= bg_cnt; }
    }
    if fg_cnt > 20.0 {
        for c in &mut fg_rgb { *c /= fg_cnt; }
    }

    let mut raw_fg = vec![0.0f32; n];
    let mut rgb01 = vec![0.0f32; n * 3];
    for i in 0..n {
        let r = rgb8[i * 3] as f32 / 255.0;
        let g = rgb8[i * 3 + 1] as f32 / 255.0;
        let b = rgb8[i * 3 + 2] as f32 / 255.0;
        rgb01[i * 3] = r;
        rgb01[i * 3 + 1] = g;
        rgb01[i * 3 + 2] = b;

        let d = depth[i];
        let sk = skin.get(i).copied().unwrap_or(0.0);
        let mut w_d = ((d - t_depth) / 0.12).clamp(0.0, 1.0);

        // If pixel is in the ambiguous mid-depth band (not sure core), test colour distance to background vs foreground
        if w_d > 0.0 && d < (t_depth + 0.18) && sk < 0.25 && bg_cnt > 20.0 && fg_cnt > 20.0 {
            let db2 = (r as f64 - bg_rgb[0]).powi(2)
                + (g as f64 - bg_rgb[1]).powi(2)
                + (b as f64 - bg_rgb[2]).powi(2);
            let df2 = (r as f64 - fg_rgb[0]).powi(2)
                + (g as f64 - fg_rgb[1]).powi(2)
                + (b as f64 - fg_rgb[2]).powi(2);
            if db2 * 1.35 < df2 {
                w_d = 0.0;
            }
        }
        raw_fg[i] = w_d.max(sk);
    }

    let r_guide = (w.max(h) / 36).max(3);
    guided_refine_rgb(&rgb01, &raw_fg, w, h, r_guide, 1e-3)
}

/// Compute Plastyka 3D relative luminance gain map (`w * h`) at angle `angle_deg` (`0..360`).
/// Zero outside the subject mask (`bg_far = 0.0`), zero-centred over `fg` so mean subject exposure
/// stays invariant (`separation >= 1.0`), with smooth rim taper (`halo_in < 0.25`).
pub fn relight_gain(
    depth: &[f32],
    fg: &[f32],
    l_star: &[f32],
    rgb01: Option<&[f32]>,
    w: usize,
    h: usize,
    strength: f32,
    angle_deg: f32,
) -> Vec<f32> {
    let n = w * h;
    if strength <= 1e-3 || n == 0 {
        return vec![0.0f32; n];
    }

    // Snap foreground mask to RGB or L* edges
    let r_snap = (w.max(h) / 50).max(2);
    let fg_ref = if let Some(rgb) = rgb01 {
        guided_refine_rgb(rgb, fg, w, h, r_snap, 1e-3)
    } else {
        let l01: Vec<f32> = l_star.iter().map(|&v| v / 100.0).collect();
        guided_filter(&l01, fg, w, h, r_snap, 1e-3)
    };

    // Person-only depth smoothed inside the silhouette
    let r_norm = (w.max(h) / 45).max(2);
    let dfg: Vec<f32> = depth.iter().zip(fg_ref.iter()).map(|(&d, &f)| d * f).collect();
    let d_num = box_filter(&dfg, w, h, r_norm);
    let d_den = box_filter(&fg_ref, w, h, r_norm);
    let mut d_person = vec![0.0f32; n];
    for i in 0..n {
        d_person[i] = d_num[i] / (d_den[i] + 1e-4);
    }

    // Surface normals from person depth gradient + directional light vector
    let rad = angle_deg.to_radians();
    let lx = rad.cos() * 0.68;
    let ly = -rad.sin() * 0.68;
    let lz = (1.0 - lx * lx - ly * ly).max(0.1).sqrt();
    let k_grad = 8.0 * (w.max(h) as f32) / 180.0;

    let mut shade = vec![0.0f32; n];
    for y in 0..h {
        let ym = y.saturating_sub(1);
        let yp = (y + 1).min(h - 1);
        for x in 0..w {
            let xm = x.saturating_sub(1);
            let xp = (x + 1).min(w - 1);
            let idx = y * w + x;
            let f_here = fg_ref[idx];
            if f_here <= 1e-3 {
                continue;
            }
            let gx = 0.5 * (d_person[y * w + xp] - d_person[y * w + xm]) * k_grad * f_here;
            let gy = 0.5 * (d_person[yp * w + x] - d_person[ym * w + x]) * k_grad * f_here;
            let inv_len = 1.0 / (gx * gx + gy * gy + 1.0).sqrt();
            let nx = -gx * inv_len;
            let ny = -gy * inv_len;
            let nz = inv_len;
            shade[idx] = (nx * lx + ny * ly + nz * lz) - lz;
        }
    }

    // Zero-centre shade over the subject so mean subject lightness does not drift
    let mut w_sum = 0.0f64;
    let mut s_sum = 0.0f64;
    for i in 0..n {
        let f = fg_ref[i] as f64;
        w_sum += f;
        s_sum += (shade[i] as f64) * f;
    }
    if w_sum > 10.0 {
        let mean_s = (s_sum / w_sum) as f32;
        for s in &mut shade {
            *s -= mean_s;
        }
    }

    // Halo-free guided local micro-contrast inside the subject
    let r_clar = (w.max(h) / 60).max(2);
    let l_base = guided_filter(l_star, l_star, w, h, r_clar, 36.0);

    let mut gain = vec![0.0f32; n];
    for i in 0..n {
        let f = fg_ref[i];
        if f <= 1e-3 {
            continue;
        }
        // Smooth rim taper (f^2 * (3 - 2f)) eliminates any dark line along the outline
        let taper = f * f * (3.0 - 2.0 * f);
        let l = l_star[i].max(25.0);
        let clar = (l_star[i] - l_base[i]).clamp(-6.0, 6.0) / l;
        let g = strength * taper * (0.42 * shade[i] + 0.16 * clar);
        gain[i] = g.clamp(-0.28, 0.32);
    }
    gain
}
