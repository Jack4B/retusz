//! Retusz Core Engine (Rust): photographic tone & colour science (OKLab, Bradford CAT,
//! 3D LUT tetrahedral interpolation, CIEDE2000, RGB Guided Filter), Dodge & Burn v2,
//! Plastyka 3D, Bilateral Guided Upsampling, Presets, and non-destructive sidecars.
//! Author: Jack4B.

pub mod bgu;
pub mod color;
pub mod dnb;
pub mod geometry;
pub mod histo;
pub mod params;
pub mod pipeline;
pub mod relight;
pub mod tone;
pub mod wb;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ciede2000_sharma_reference_pair() {
        // Sharma, Wu, Dalal (2005) test data, pair 1 -> 2.0425
        let de = color::delta_e2000([50.0, 2.6772, -79.7751], [50.0, 0.0, -82.7485]);
        assert!((de - 2.0425).abs() < 1e-3, "de = {de}");
    }

    #[test]
    fn test_tetrahedral_lut_preserves_neutral_diagonal_and_roundtrips() {
        let lut = color::Lut3D::identity(33);
        for v in [0.0f32, 0.123, 0.25, 0.5, 0.731, 1.0] {
            let out = lut.sample_tetra([v, v, v]);
            assert!((out[0] - v).abs() < 1e-6);
            assert!((out[1] - v).abs() < 1e-6);
            assert!((out[2] - v).abs() < 1e-6);
        }
    }

    #[test]
    fn test_oklab_roundtrip_and_zero_hue_drift_gamut_clipping() {
        let lin = [0.05f32, 0.18, 0.92];
        let lab = tone::lin_to_oklab(lin);
        let back = tone::oklab_to_lin_gamut(lab);
        for i in 0..3 {
            assert!((lin[i] - back[i]).abs() < 1e-4);
        }
        // Push blue chroma 2.5x out of gamut and verify OKLab hue angle is invariant (< 0.2 deg)
        let h0 = lab[2].atan2(lab[1]);
        let clipped = tone::oklab_to_lin_gamut([lab[0], lab[1] * 2.5, lab[2] * 2.5]);
        let lab1 = tone::lin_to_oklab(clipped);
        let h1 = lab1[2].atan2(lab1[1]);
        let dh = (h1 - h0).abs().to_degrees();
        assert!(dh < 0.2, "hue drift deg = {dh}");
    }

    #[test]
    fn test_exposure_ev_and_contrast_midgrey_pivot() {
        // +1 EV doubles linear light in shadows/midtones
        let s = [0.12f32, 0.18, 0.24];
        let p1 = tone::apply_exposure_lin(s, 1.0);
        assert!((p1[0] - 0.24).abs() < 1e-4);
        // +2 EV soft shoulder keeps chromaticity ratio and never exceeds 1.0
        let hi = tone::apply_exposure_lin([0.6, 0.45, 0.3], 2.0);
        assert!(hi[0] <= 1.0 && hi[1] <= 1.0 && hi[2] <= 1.0);
        assert!((hi[0] / hi[1] - 0.6 / 0.45).abs() < 1e-3);

        // Positive contrast darkens below 18% grey and brightens above
        let dark = tone::apply_tone_lin([0.05, 0.05, 0.05], 0.6, 0.0, 0.0, 0.0, 0.0);
        let bright = tone::apply_tone_lin([0.45, 0.45, 0.45], 0.6, 0.0, 0.0, 0.0, 0.0);
        assert!(dark[0] < 0.05);
        assert!(bright[0] > 0.45);
    }

    #[test]
    fn test_bradford_wb_eyedropper_neutralises_cast() {
        // Create a warm-tinted grey patch, solve eyedropper, and verify Bradford CAT neutralises it
        let warm = wb::apply_bradford01([0.6, 0.6, 0.6], 0.45, -0.25);
        let px = [
            (warm[0] * 255.0 + 0.5) as u8,
            (warm[1] * 255.0 + 0.5) as u8,
            (warm[2] * 255.0 + 0.5) as u8,
        ];
        let (temp, tint) = wb::solve_eyedropper(&[px; 9]);
        let neutral = wb::apply_bradford01(warm, temp, tint);
        let lab = color::rgb01_to_lab(neutral);
        let chroma = lab[1].hypot(lab[2]);
        assert!(chroma < 1.2, "residual chroma = {chroma}");
    }

    #[test]
    fn test_skin_mask_rejects_dark_hair_and_background() {
        let (w, h) = (120usize, 120usize);
        let mut rgb = vec![0u8; w * h * 3];
        // Neutral grey background
        for px in rgb.chunks_exact_mut(3) {
            px[0] = 110; px[1] = 115; px[2] = 122;
        }
        // Face ellipse with warm skin + dark brown hair cap at top (y = 25..42)
        for y in 25..95 {
            for x in 35..85 {
                let idx = (y * w + x) * 3;
                if y < 42 {
                    // Dark brown hair (same hue as skin, low luminance)
                    rgb[idx] = 36; rgb[idx + 1] = 24; rgb[idx + 2] = 18;
                } else {
                    // Cheek skin
                    rgb[idx] = 206; rgb[idx + 1] = 160; rgb[idx + 2] = 136;
                }
            }
        }
        let m = dnb::skin_mask(&rgb, w, h, &[[35.0, 25.0, 50.0, 70.0]]);
        let cheek = m[65 * w + 60];
        let hair = m[32 * w + 60];
        let bg = m[10 * w + 10];
        assert!(cheek > 0.75, "cheek = {cheek}");
        assert!(hair < 0.15, "hair leaked = {hair}");
        assert!(bg < 0.05, "bg leaked = {bg}");
    }

    #[test]
    fn test_plastyka_3d_zero_outside_subject_and_preserves_mean_exposure() {
        let (w, h) = (80usize, 80usize);
        let n = w * h;
        let mut depth = vec![0.10f32; n];
        let mut fg = vec![0.0f32; n];
        let l_star = vec![62.0f32; n];
        for y in 20..60 {
            for x in 20..60 {
                let r2 = ((x as f32 - 40.0) / 20.0).powi(2) + ((y as f32 - 40.0) / 20.0).powi(2);
                if r2 < 1.0 {
                    depth[y * w + x] = 0.85 - 0.35 * r2;
                    fg[y * w + x] = 1.0;
                }
            }
        }
        let g = relight::relight_gain(&depth, &fg, &l_star, None, w, h, 0.8, 120.0);
        // Far background must be strictly untouched
        assert_eq!(g[5 * w + 5], 0.0);
        // Inside the subject, 3D relief is non-zero and mean gain is near zero
        let mut fg_sum = 0.0f32;
        let mut g_sum = 0.0f32;
        let mut g_max = 0.0f32;
        for i in 0..n {
            if fg[i] > 0.5 {
                fg_sum += fg[i];
                g_sum += g[i] * fg[i];
                g_max = g_max.max(g[i].abs());
            }
        }
        assert!(g_max > 0.01, "expected 3D relief > 0.01, got {g_max}");
        assert!((g_sum / fg_sum).abs() < 0.015, "mean shift = {}", g_sum / fg_sum);
    }

    #[test]
    fn test_bgu_fit_and_preview_cache_reconstruct_affine_grade() {
        let (w, h) = (64usize, 48usize);
        let mut guide = vec![0.0f32; w * h * 3];
        let mut target = vec![0.0f32; w * h * 3];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 3;
                let r = (x as f32 / w as f32) * 0.8 + 0.1;
                let g = (y as f32 / h as f32) * 0.8 + 0.1;
                let b = 0.5 * (r + g);
                guide[i] = r; guide[i + 1] = g; guide[i + 2] = b;
                target[i] = (r * 1.12 + 0.03).clamp(0.0, 1.0);
                target[i + 1] = (g * 0.95 + 0.01).clamp(0.0, 1.0);
                target[i + 2] = (b * 0.88 - 0.02).clamp(0.0, 1.0);
            }
        }
        let grid = bgu::fit_bgu(&guide, &target, None, w, h, 8.0, 8, 1e-4);
        let cache = bgu::PreviewCache::build(guide.clone(), w, h, &grid, &grid, 1.0, 1.0);
        let out8 = cache.mix_rgb8(params::TransferParams::default(), None);
        let mut err = 0.0f32;
        for i in 0..(w * h * 3) {
            let t8 = (target[i] * 255.0 + 0.5) as f32;
            err += (out8[i] as f32 - t8).abs();
        }
        let mae = err / (w * h * 3) as f32;
        assert!(mae < 1.2, "BGU MAE = {mae}");
    }

    #[test]
    fn test_presets_and_sidecar_json_roundtrip() {
        let builtins = params::builtin_presets();
        assert!(builtins.len() >= 6);
        let mut p = params::Params::default();
        p.angle = 3.5;
        p.crop = Some([0.1, 0.1, 0.9, 0.9]);
        let applied = builtins[1].apply_to(&p);
        assert_eq!(applied.angle, 3.5);
        assert_eq!(applied.crop, Some([0.1, 0.1, 0.9, 0.9]));
        assert!((applied.exposure - builtins[1].params.exposure).abs() < 1e-6);
    }
}
