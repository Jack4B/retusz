//! Colour science & image filters: sRGB <-> Linear, CIELAB, CIEDE2000 (Sharma et al. 2005),
//! 3D LUT with tetrahedral interpolation (Sakamoto 1974 / Kasson 1995), MKL colour transfer
//! (Pitié & Kokaram 2007), and 1-channel / 3-channel RGB Guided Filter (He, Sun, Tang 2010; He & Sun 2015).
//! Author: Jack4B.

use rayon::prelude::*;

/// sRGB [0..1] -> linear RGB [0..1].
#[inline]
pub fn srgb_to_lin_scalar(c: f32) -> f32 {
    let x = c.clamp(0.0, 1.0);
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear RGB [0..1] -> sRGB [0..1].
#[inline]
pub fn lin_to_srgb_scalar(c: f32) -> f32 {
    let x = c.clamp(0.0, 1.0);
    if x <= 0.0031308 {
        12.92 * x
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    }
}

#[inline]
pub fn srgb_to_lin(rgb: [f32; 3]) -> [f32; 3] {
    [
        srgb_to_lin_scalar(rgb[0]),
        srgb_to_lin_scalar(rgb[1]),
        srgb_to_lin_scalar(rgb[2]),
    ]
}

#[inline]
pub fn lin_to_srgb(lin: [f32; 3]) -> [f32; 3] {
    [
        lin_to_srgb_scalar(lin[0]),
        lin_to_srgb_scalar(lin[1]),
        lin_to_srgb_scalar(lin[2]),
    ]
}

/// sRGB [0..1] -> CIELAB (D65, L in 0..100, a/b in -128..127).
#[inline]
pub fn rgb01_to_lab(rgb: [f32; 3]) -> [f32; 3] {
    let [r, g, b] = srgb_to_lin(rgb);
    // sRGB D65 -> XYZ, divided by D65 white point (0.95047, 1.00000, 1.08883)
    let x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047;
    let y =  0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883;
    #[inline]
    fn f(t: f32) -> f32 {
        if t > 0.008856452 {
            t.cbrt()
        } else {
            7.787037 * t + 16.0 / 116.0
        }
    }
    let (fx, fy, fz) = (f(x), f(y), f(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

/// CIELAB (D65) -> sRGB [0..1].
#[inline]
pub fn lab_to_rgb01(lab: [f32; 3]) -> [f32; 3] {
    let fy = (lab[0] + 16.0) / 116.0;
    let fx = fy + lab[1] / 500.0;
    let fz = fy - lab[2] / 200.0;
    #[inline]
    fn finv(t: f32) -> f32 {
        let t3 = t * t * t;
        if t3 > 0.008856452 {
            t3
        } else {
            (t - 16.0 / 116.0) / 7.787037
        }
    }
    let x = finv(fx) * 0.95047;
    let y = finv(fy);
    let z = finv(fz) * 1.08883;
    let r =  3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    let b =  0.0556434 * x - 0.2040259 * y + 1.0572252 * z;
    lin_to_srgb([r.clamp(0.0, 1.0), g.clamp(0.0, 1.0), b.clamp(0.0, 1.0)])
}

/// CIEDE2000 colour difference (Sharma, Wu, Dalal 2005).
pub fn delta_e2000(lab1: [f64; 3], lab2: [f64; 3]) -> f64 {
    let (l1, a1, b1) = (lab1[0], lab1[1], lab1[2]);
    let (l2, a2, b2) = (lab2[0], lab2[1], lab2[2]);
    let c1 = a1.hypot(b1);
    let c2 = a2.hypot(b2);
    let cm = 0.5 * (c1 + c2);
    let cm7 = cm.powi(7);
    let g = 0.5 * (1.0 - (cm7 / (cm7 + 25.0_f64.powi(7))).sqrt());
    let a1p = (1.0 + g) * a1;
    let a2p = (1.0 + g) * a2;
    let c1p = a1p.hypot(b1);
    let c2p = a2p.hypot(b2);
    let mut h1p = b1.atan2(a1p).to_degrees();
    if h1p < 0.0 { h1p += 360.0; }
    let mut h2p = b2.atan2(a2p).to_degrees();
    if h2p < 0.0 { h2p += 360.0; }
    let dlp = l2 - l1;
    let dcp = c2p - c1p;
    let dh = if c1p * c2p == 0.0 {
        0.0
    } else {
        let mut d = h2p - h1p;
        if d > 180.0 { d -= 360.0; } else if d < -180.0 { d += 360.0; }
        d
    };
    let dhp = 2.0 * (c1p * c2p).sqrt() * (0.5 * dh).to_radians().sin();
    let lpm = 0.5 * (l1 + l2);
    let cpm = 0.5 * (c1p + c2p);
    let hs = h1p + h2p;
    let hpm = if c1p * c2p == 0.0 {
        hs
    } else if (h1p - h2p).abs() <= 180.0 {
        0.5 * hs
    } else if hs < 360.0 {
        0.5 * (hs + 360.0)
    } else {
        0.5 * (hs - 360.0)
    };
    let t = 1.0
        - 0.17 * (hpm - 30.0).to_radians().cos()
        + 0.24 * (2.0 * hpm).to_radians().cos()
        + 0.32 * (3.0 * hpm + 6.0).to_radians().cos()
        - 0.20 * (4.0 * hpm - 63.0).to_radians().cos();
    let dth = 30.0 * (-((hpm - 275.0) / 25.0).powi(2)).exp();
    let cpm7 = cpm.powi(7);
    let rc = 2.0 * (cpm7 / (cpm7 + 25.0_f64.powi(7))).sqrt();
    let l50 = (lpm - 50.0).powi(2);
    let sl = 1.0 + 0.015 * l50 / (20.0 + l50).sqrt();
    let sc = 1.0 + 0.045 * cpm;
    let sh = 1.0 + 0.015 * cpm * t;
    let rt = -(2.0 * dth).to_radians().sin() * rc;
    ((dlp / sl).powi(2)
        + (dcp / sc).powi(2)
        + (dhp / sh).powi(2)
        + rt * (dcp / sc) * (dhp / sh))
        .sqrt()
}

/// 3D colour lookup table (`n x n x n x 3` float32 in 0..1) with tetrahedral interpolation
/// (Sakamoto 1974 / Kasson et al. 1995): 4 corners per pixel, exact neutral axis R=G=B.
#[derive(Debug, Clone)]
pub struct Lut3D {
    pub n: usize,
    pub data: Vec<[f32; 3]>,
}

impl Lut3D {
    pub fn identity(n: usize) -> Self {
        let mut data = Vec::with_capacity(n * n * n);
        let scale = 1.0 / (n.saturating_sub(1).max(1) as f32);
        for ir in 0..n {
            for ig in 0..n {
                for ib in 0..n {
                    data.push([ir as f32 * scale, ig as f32 * scale, ib as f32 * scale]);
                }
            }
        }
        Self { n, data }
    }

    #[inline]
    pub fn get(&self, ir: usize, ig: usize, ib: usize) -> [f32; 3] {
        self.data[(ir * self.n + ig) * self.n + ib]
    }

    /// Tetrahedral interpolation of sRGB [0..1] through the 3D LUT.
    #[inline]
    pub fn sample_tetra(&self, rgb01: [f32; 3]) -> [f32; 3] {
        let nm1 = (self.n - 1) as f32;
        let xr = rgb01[0].clamp(0.0, 1.0) * nm1;
        let xg = rgb01[1].clamp(0.0, 1.0) * nm1;
        let xb = rgb01[2].clamp(0.0, 1.0) * nm1;
        let max_i = self.n - 2;
        let r0 = (xr.floor() as usize).min(max_i);
        let g0 = (xg.floor() as usize).min(max_i);
        let b0 = (xb.floor() as usize).min(max_i);
        let fr = xr - r0 as f32;
        let fg = xg - g0 as f32;
        let fb = xb - b0 as f32;

        let v000 = self.get(r0, g0, b0);
        let v111 = self.get(r0 + 1, g0 + 1, b0 + 1);
        let (w0, wa, wb, w1, va, vb) = if fr >= fg {
            if fg >= fb {
                (1.0 - fr, fr - fg, fg - fb, fb, self.get(r0 + 1, g0, b0), self.get(r0 + 1, g0 + 1, b0))
            } else if fr >= fb {
                (1.0 - fr, fr - fb, fb - fg, fg, self.get(r0 + 1, g0, b0), self.get(r0 + 1, g0, b0 + 1))
            } else {
                (1.0 - fb, fb - fr, fr - fg, fg, self.get(r0, g0, b0 + 1), self.get(r0 + 1, g0, b0 + 1))
            }
        } else if fb > fg {
            (1.0 - fb, fb - fg, fg - fr, fr, self.get(r0, g0, b0 + 1), self.get(r0, g0 + 1, b0 + 1))
        } else if fb > fr {
            (1.0 - fg, fg - fb, fb - fr, fr, self.get(r0, g0 + 1, b0), self.get(r0, g0 + 1, b0 + 1))
        } else {
            (1.0 - fg, fg - fr, fr - fb, fb, self.get(r0, g0 + 1, b0), self.get(r0 + 1, g0 + 1, b0))
        };
        [
            w0 * v000[0] + wa * va[0] + wb * vb[0] + w1 * v111[0],
            w0 * v000[1] + wa * va[1] + wb * vb[1] + w1 * v111[1],
            w0 * v000[2] + wa * va[2] + wb * vb[2] + w1 * v111[2],
        ]
    }

    /// Apply 3D LUT in parallel across an RGB8 buffer (`w * h * 3`).
    pub fn apply_rgb8(&self, src: &[u8]) -> Vec<u8> {
        let mut out = vec![0u8; src.len()];
        out.par_chunks_mut(3)
            .zip(src.par_chunks(3))
            .for_each(|(dst, s)| {
                let c = self.sample_tetra([
                    s[0] as f32 / 255.0,
                    s[1] as f32 / 255.0,
                    s[2] as f32 / 255.0,
                ]);
                dst[0] = (c[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                dst[1] = (c[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
                dst[2] = (c[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            });
        out
    }

    /// Separable 1-2-1 smoothing across all 3 LUT axes (removes CDF steps while preserving monotonicity).
    pub fn smooth(&mut self) {
        let n = self.n;
        for axis in 0..3 {
            let prev = self.data.clone();
            for ir in 0..n {
                for ig in 0..n {
                    for ib in 0..n {
                        let idx = [ir, ig, ib];
                        if idx[axis] == 0 || idx[axis] == n - 1 {
                            continue;
                        }
                        let mut im = idx;
                        let mut ip = idx;
                        im[axis] -= 1;
                        ip[axis] += 1;
                        let vm = prev[(im[0] * n + im[1]) * n + im[2]];
                        let v0 = prev[(ir * n + ig) * n + ib];
                        let vp = prev[(ip[0] * n + ip[1]) * n + ip[2]];
                        self.data[(ir * n + ig) * n + ib] = [
                            0.25 * vm[0] + 0.5 * v0[0] + 0.25 * vp[0],
                            0.25 * vm[1] + 0.5 * v0[1] + 0.25 * vp[1],
                            0.25 * vm[2] + 0.5 * v0[2] + 0.25 * vp[2],
                        ];
                    }
                }
            }
        }
    }
}

/// O(1) per-pixel 2D box mean filter with reflect-border padding.
pub fn box_filter(src: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 || w == 0 || h == 0 {
        return src.to_vec();
    }
    let mut tmp = vec![0.0f32; w * h];
    let mut out = vec![0.0f32; w * h];
    let inv = 1.0 / (2 * r + 1) as f32;

    // Horizontal pass
    for y in 0..h {
        let row = &src[y * w..(y + 1) * w];
        let dst = &mut tmp[y * w..(y + 1) * w];
        let mut sum = 0.0f32;
        for k in -(r as isize)..=(r as isize) {
            let x = k.unsigned_abs().min(w - 1);
            sum += row[x];
        }
        dst[0] = sum * inv;
        for x in 1..w {
            let add_idx = (x + r).min(2 * w - 1 - (x + r).min(2 * w - 1)).min(w - 1);
            let sub_raw = x as isize - r as isize - 1;
            let sub_idx = sub_raw.unsigned_abs().min(w - 1);
            sum += row[add_idx] - row[sub_idx];
            dst[x] = sum * inv;
        }
    }
    // Vertical pass
    for x in 0..w {
        let mut sum = 0.0f32;
        for k in -(r as isize)..=(r as isize) {
            let y = k.unsigned_abs().min(h - 1);
            sum += tmp[y * w + x];
        }
        out[x] = sum * inv;
        for y in 1..h {
            let add_idx = (y + r).min(2 * h - 1 - (y + r).min(2 * h - 1)).min(h - 1);
            let sub_raw = y as isize - r as isize - 1;
            let sub_idx = sub_raw.unsigned_abs().min(h - 1);
            sum += tmp[add_idx * w + x] - tmp[sub_idx * w + x];
            out[y * w + x] = sum * inv;
        }
    }
    out
}

/// 1-channel Guided Filter (He, Sun, Tang, ECCV 2010): O(N) edge-preserving filter without halos.
pub fn guided_filter(guide: &[f32], src: &[f32], w: usize, h: usize, r: usize, eps: f32) -> Vec<f32> {
    let n = w * h;
    let mean_i = box_filter(guide, w, h, r);
    let mean_p = box_filter(src, w, h, r);
    let mut ip = vec![0.0f32; n];
    let mut ii = vec![0.0f32; n];
    for idx in 0..n {
        ip[idx] = guide[idx] * src[idx];
        ii[idx] = guide[idx] * guide[idx];
    }
    let corr_ip = box_filter(&ip, w, h, r);
    let corr_ii = box_filter(&ii, w, h, r);
    let mut a = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    for idx in 0..n {
        let cov_ip = corr_ip[idx] - mean_i[idx] * mean_p[idx];
        let var_i = (corr_ii[idx] - mean_i[idx] * mean_i[idx]).max(0.0);
        let ai = cov_ip / (var_i + eps);
        a[idx] = ai;
        b[idx] = mean_p[idx] - ai * mean_i[idx];
    }
    let mean_a = box_filter(&a, w, h, r);
    let mean_b = box_filter(&b, w, h, r);
    let mut out = vec![0.0f32; n];
    for idx in 0..n {
        out[idx] = mean_a[idx] * guide[idx] + mean_b[idx];
    }
    out
}

/// 3-channel RGB Guided Filter (He, Sun, Tang 2010, eqs. 14-16, explicit 3x3 covariance inversion).
/// Snaps a soft mask `p` to colour AND luminance edges of `rgb01` (`w * h * 3` in 0..1).
pub fn guided_refine_rgb(rgb01: &[f32], p: &[f32], w: usize, h: usize, r: usize, eps: f32) -> Vec<f32> {
    let n = w * h;
    let mut ir = vec![0.0f32; n];
    let mut ig = vec![0.0f32; n];
    let mut ib = vec![0.0f32; n];
    for i in 0..n {
        ir[i] = rgb01[i * 3];
        ig[i] = rgb01[i * 3 + 1];
        ib[i] = rgb01[i * 3 + 2];
    }
    let mr = box_filter(&ir, w, h, r);
    let mg = box_filter(&ig, w, h, r);
    let mb = box_filter(&ib, w, h, r);
    let mp = box_filter(p, w, h, r);

    let prod = |a: &[f32], b: &[f32]| -> Vec<f32> {
        let mut v = vec![0.0f32; n];
        for i in 0..n { v[i] = a[i] * b[i]; }
        box_filter(&v, w, h, r)
    };
    let crr_m = prod(&ir, &ir);
    let crg_m = prod(&ir, &ig);
    let crb_m = prod(&ir, &ib);
    let cgg_m = prod(&ig, &ig);
    let cgb_m = prod(&ig, &ib);
    let cbb_m = prod(&ib, &ib);
    let cpr_m = prod(&ir, p);
    let cpg_m = prod(&ig, p);
    let cpb_m = prod(&ib, p);

    let mut ar = vec![0.0f32; n];
    let mut ag = vec![0.0f32; n];
    let mut ab = vec![0.0f32; n];
    let mut b0 = vec![0.0f32; n];

    for i in 0..n {
        let srr = crr_m[i] - mr[i] * mr[i] + eps;
        let srg = crg_m[i] - mr[i] * mg[i];
        let srb = crb_m[i] - mr[i] * mb[i];
        let sgg = cgg_m[i] - mg[i] * mg[i] + eps;
        let sgb = cgb_m[i] - mg[i] * mb[i];
        let sbb = cbb_m[i] - mb[i] * mb[i] + eps;

        let cpr = cpr_m[i] - mr[i] * mp[i];
        let cpg = cpg_m[i] - mg[i] * mp[i];
        let cpb = cpb_m[i] - mb[i] * mp[i];

        let i00 = sgg * sbb - sgb * sgb;
        let i01 = srb * sgb - srg * sbb;
        let i02 = srg * sgb - srb * sgg;
        let i11 = srr * sbb - srb * srb;
        let i12 = srb * srg - srr * sgb;
        let i22 = srr * sgg - srg * srg;
        let det = (srr * i00 + srg * i01 + srb * i02).max(1e-12);
        let inv_det = 1.0 / det;

        let a_r = (i00 * cpr + i01 * cpg + i02 * cpb) * inv_det;
        let a_g = (i01 * cpr + i11 * cpg + i12 * cpb) * inv_det;
        let a_b = (i02 * cpr + i12 * cpg + i22 * cpb) * inv_det;
        ar[i] = a_r;
        ag[i] = a_g;
        ab[i] = a_b;
        b0[i] = mp[i] - a_r * mr[i] - a_g * mg[i] - a_b * mb[i];
    }

    let mar = box_filter(&ar, w, h, r);
    let mag = box_filter(&ag, w, h, r);
    let mab = box_filter(&ab, w, h, r);
    let mb0 = box_filter(&b0, w, h, r);

    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = (mar[i] * ir[i] + mag[i] * ig[i] + mab[i] * ib[i] + mb0[i]).clamp(0.0, 1.0);
    }
    out
}
