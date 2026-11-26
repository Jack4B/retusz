//! Biophysical skin optics & Fast Local Laplacian Filter (Fast LLF) in Rust (`retusz-core`).
//!
//! References:
//! - Tsumura et al., "Image-based skin color and texture analysis/synthesis by extracting
//!   hemoglobin and melanin information in the skin", ACM SIGGRAPH 2003.
//! - Mallick et al., "Beyond Lambert: Reconstructing Specular Surfaces Using Color", CVPR 2005.
//! - Yang, Wang, Ahuja, "Real-Time Specular Highlight Removal Using Bilateral Filtering", ECCV 2010.
//! - Aubry, Paris, Hasinoff, Kautz, Durand, "Fast Local Laplacian Filters: Theory and Applications",
//!   ACM Transactions on Graphics 2014.
//!
//! Author: Jack4B.

use crate::color::srgb_to_lin;

/// Decompose an sRGB 0..1 skin pixel into Beer-Lambert optical density Melanin and Hemoglobin components
/// (`-log10(I_lin)` domain, Tsumura et al. SIGGRAPH 2003).
/// Returns `(melanin, hemoglobin)`.
#[inline]
pub fn decompose_chromophores(srgb: [f32; 3]) -> (f32, f32) {
    let lin = srgb_to_lin(srgb);
    let dr = -lin[0].max(1e-4).log10();
    let dg = -lin[1].max(1e-4).log10();
    let db = -lin[2].max(1e-4).log10();
    let hemoglobin = (dg - dr) - 0.35 * (db - dg);
    let melanin = 0.65 * (db - dr) + 0.35 * dr;
    (melanin, hemoglobin)
}

/// Dichromatic SUV De-Shine (Mallick 2005 / Yang 2010): remove white surface glare `m_s * (1, 1, 1)`
/// relative to the local diffuse skin baseline `base_diffuse` (in 0..255 sRGB units) and restore
/// warm subsurface skin chromaticity.
#[inline]
pub fn deshine_pixel_rgb(rgb: [f32; 3], base_diffuse: [f32; 3], amount: f32) -> [f32; 3] {
    let amt = amount.clamp(0.0, 1.0);
    if amt <= 1e-4 {
        return rgb;
    }
    let i_min = rgb[0].min(rgb[1]).min(rgb[2]);
    let base_min = base_diffuse[0].min(base_diffuse[1]).min(base_diffuse[2]);
    let lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
    let base_lum = 0.299 * base_diffuse[0] + 0.587 * base_diffuse[1] + 0.114 * base_diffuse[2];

    let spec_min = (i_min - base_min).max(0.0);
    let spec_lum = (lum - base_lum).max(0.0);
    let spec = spec_min.min(1.15 * spec_lum);
    let glare_w = (spec / 28.0).clamp(0.0, 1.0);

    let mut out = [0.0f32; 3];
    for c in 0..3 {
        let base_c = base_diffuse[c] - base_lum;
        let cur_c = rgb[c] - lum;
        out[c] = (rgb[c] - amt * 0.96 * spec + amt * 0.55 * glare_w * (base_c - cur_c)).clamp(0.0, 255.0);
    }
    out
}

/// Fast Local Laplacian Filter remapping function `r_gamma(L)` (Aubry et al. 2014):
/// amplifies fine and medium details (`|L - gamma| <= sigma_r`) while leaving large step edges
/// (`|L - gamma| >> sigma_r`) untouched (zero halo).
#[inline]
pub fn fast_llf_remap(l: f32, gamma: f32, alpha: f32, sigma_r: f32) -> f32 {
    let d = l - gamma;
    let w = (-(d * d) / (2.0 * sigma_r * sigma_r)).exp();
    (l + alpha * d * w).clamp(0.0, 100.0)
}
