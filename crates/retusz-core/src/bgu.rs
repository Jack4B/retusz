//! Bilateral Guided Upsampling (Chen, Adams, Wadhwa, Hasinoff, SIGGRAPH Asia 2016)
//! and AI Grading Transfer (`Przenieś grading z AI`):
//! - Trilinear splatting of 4x4 sufficient statistics into a 3D bilateral grid `(gx, gy, bins)`.
//! - Regularised per-cell 4x4 ridge solver towards a global / coarse affine prior.
//! - Multi-threaded trilinear slicing (`rayon`) + fast CIELAB `PreviewCache` (`mix_cached`).
//! - Multi-scale DoG band-pass `changed_map` above per-image Immerkaer noise floor.
//! Author: Jack4B.

use crate::color::{box_filter, lab_to_rgb01, rgb01_to_lab};
use crate::params::TransferParams;
use rayon::prelude::*;

/// 3D Bilateral Grid of 3x4 affine colour transforms `out_c = A[c, 0..3] . rgb + A[c, 3]`.
#[derive(Debug, Clone)]
pub struct BilateralGrid {
    pub gh: usize,
    pub gw: usize,
    pub bins: usize,
    pub cell: f32,
    pub matrices: Vec<[[f32; 4]; 3]>, // length = gh * gw * bins
    pub prior: [[f32; 4]; 3],
}

#[inline]
fn luma01(rgb: [f32; 3]) -> f32 {
    (0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]).clamp(0.0, 1.0)
}

/// Solve symmetric 4x4 linear system `A x = b` via Gaussian elimination with partial pivoting.
fn solve_4x4(mut a: [[f64; 4]; 4], mut b: [f64; 4]) -> [f64; 4] {
    for i in 0..4 {
        let mut piv = i;
        let mut max_v = a[i][i].abs();
        for r in (i + 1)..4 {
            if a[r][i].abs() > max_v {
                max_v = a[r][i].abs();
                piv = r;
            }
        }
        if piv != i {
            a.swap(i, piv);
            b.swap(i, piv);
        }
        let diag = if a[i][i].abs() < 1e-12 { 1e-12 } else { a[i][i] };
        for r in (i + 1)..4 {
            let f = a[r][i] / diag;
            for c in i..4 {
                a[r][c] -= f * a[i][c];
            }
            b[r] -= f * b[i];
        }
    }
    let mut x = [0.0f64; 4];
    for i in (0..4).rev() {
        let mut s = b[i];
        for c in (i + 1)..4 {
            s -= a[i][c] * x[c];
        }
        let diag = if a[i][i].abs() < 1e-12 { 1e-12 } else { a[i][i] };
        x[i] = s / diag;
    }
    x
}

/// Fit a 3D bilateral grid mapping `guide_rgb01` -> `target_rgb01` (`w * h * 3` in 0..1)
/// weighted by `weight` (`w * h` in 0..1).
pub fn fit_bgu(
    guide_rgb01: &[f32],
    target_rgb01: &[f32],
    weight: Option<&[f32]>,
    w: usize,
    h: usize,
    cell: f32,
    bins: usize,
    lam: f64,
) -> BilateralGrid {
    let gh = ((h.saturating_sub(1) as f32 / cell).ceil() as usize) + 1;
    let gw = ((w.saturating_sub(1) as f32 / cell).ceil() as usize) + 1;
    let n_cells = gh * gw * bins;
    let mut stats = vec![[0.0f64; 28]; n_cells];
    let mut tot = [0.0f64; 28];

    for y in 0..h {
        let gy = y as f32 / cell;
        let y0 = (gy.floor() as usize).min(gh.saturating_sub(1));
        let fy = gy - y0 as f32;
        for x in 0..w {
            let idx = y * w + x;
            let wt = weight.map(|m| m[idx]).unwrap_or(1.0) as f64;
            if wt <= 0.0 {
                continue;
            }
            let gx = x as f32 / cell;
            let x0 = (gx.floor() as usize).min(gw.saturating_sub(1));
            let fx = gx - x0 as f32;

            let g = [
                guide_rgb01[idx * 3],
                guide_rgb01[idx * 3 + 1],
                guide_rgb01[idx * 3 + 2],
            ];
            let t = [
                target_rgb01[idx * 3] as f64,
                target_rgb01[idx * 3 + 1] as f64,
                target_rgb01[idx * 3 + 2] as f64,
            ];
            let zz = luma01(g) * (bins - 1) as f32;
            let z0 = (zz.floor() as usize).min(bins.saturating_sub(2));
            let fz = zz - z0 as f32;

            let x_vec = [g[0] as f64, g[1] as f64, g[2] as f64, 1.0];
            let mut feat = [0.0f64; 28];
            for i in 0..4 {
                for j in 0..4 {
                    feat[i * 4 + j] = x_vec[i] * x_vec[j] * wt;
                }
                for c in 0..3 {
                    feat[16 + i * 3 + c] = x_vec[i] * t[c] * wt;
                }
            }
            for k in 0..28 {
                tot[k] += feat[k];
            }

            for dz in 0..2 {
                let wz = if dz == 1 { fz } else { 1.0 - fz };
                let zz_i = (z0 + dz).min(bins - 1);
                for dy in 0..2 {
                    let wy = if dy == 1 { fy } else { 1.0 - fy };
                    let yy_i = (y0 + dy).min(gh - 1);
                    for dx in 0..2 {
                        let wx = if dx == 1 { fx } else { 1.0 - fx };
                        let xx_i = (x0 + dx).min(gw - 1);
                        let ww = (wx * wy * wz) as f64;
                        if ww <= 0.0 {
                            continue;
                        }
                        let cell_s = &mut stats[(yy_i * gw + xx_i) * bins + zz_i];
                        for k in 0..28 {
                            cell_s[k] += feat[k] * ww;
                        }
                    }
                }
            }
        }
    }

    // Global affine prior
    let mut prior = [[1.0f32, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]];
    let n_tot = tot[15];
    if n_tot > 1e-6 {
        let ridge = 1e-6 * n_tot;
        let mut xxg = [[0.0f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                xxg[i][j] = tot[i * 4 + j];
            }
            xxg[i][i] += if i < 3 { ridge } else { 1e-9 };
        }
        for c in 0..3 {
            let mut xyg = [
                tot[16 + c],
                tot[16 + 3 + c],
                tot[16 + 6 + c],
                tot[16 + 9 + c],
            ];
            xyg[c] += ridge;
            let sol = solve_4x4(xxg, xyg);
            for i in 0..4 {
                prior[c][i] = sol[i] as f32;
            }
        }
    }

    // Smooth sufficient statistics along grid axes (1-2-1 kernel)
    for axis in 0..3 {
        let prev = stats.clone();
        for iy in 0..gh {
            for ix in 0..gw {
                for iz in 0..bins {
                    let mut acc = [0.0f64; 28];
                    let mut wsum = 0.0f64;
                    for d in -1isize..=1 {
                        let kw = if d == 0 { 0.5 } else { 0.25 };
                        let (ny, nx, nz) = match axis {
                            0 => ((iy as isize + d).clamp(0, gh as isize - 1) as usize, ix, iz),
                            1 => (iy, (ix as isize + d).clamp(0, gw as isize - 1) as usize, iz),
                            _ => (iy, ix, (iz as isize + d).clamp(0, bins as isize - 1) as usize),
                        };
                        let s = &prev[(ny * gw + nx) * bins + nz];
                        for k in 0..28 {
                            acc[k] += kw * s[k];
                        }
                        wsum += kw;
                    }
                    let dst = &mut stats[(iy * gw + ix) * bins + iz];
                    for k in 0..28 {
                        dst[k] = acc[k] / wsum;
                    }
                }
            }
        }
    }

    // Per-cell ridge solve towards global prior
    let mut matrices = vec![prior; n_cells];
    for (cell_idx, s) in stats.iter().enumerate() {
        let n_cell = s[15].max(0.0);
        let r_slope = lam * n_cell + 1e-4;
        let r_bias = 1e-4;
        let mut xx = [[0.0f64; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                xx[i][j] = s[i * 4 + j];
            }
            xx[i][i] += if i < 3 { r_slope } else { r_bias };
        }
        let mut m_cell = [[0.0f32; 4]; 3];
        for c in 0..3 {
            let p_c = [
                prior[c][0] as f64,
                prior[c][1] as f64,
                prior[c][2] as f64,
                prior[c][3] as f64,
            ];
            let xy = [
                s[16 + c] + r_slope * p_c[0],
                s[16 + 3 + c] + r_slope * p_c[1],
                s[16 + 6 + c] + r_slope * p_c[2],
                s[16 + 9 + c] + r_bias * p_c[3],
            ];
            let sol = solve_4x4(xx, xy);
            for i in 0..4 {
                m_cell[c][i] = sol[i] as f32;
            }
        }
        matrices[cell_idx] = m_cell;
    }

    BilateralGrid { gh, gw, bins, cell, matrices, prior }
}

/// Evaluate a 3D bilateral grid on an image strip `img_rgb01` (`w * h * 3` in 0..1)
/// starting at full-res row `y0`. Multi-threaded via `rayon`.
pub fn slice_bgu(
    img_rgb01: &[f32],
    w: usize,
    h: usize,
    grid: &BilateralGrid,
    sx: f32,
    sy: f32,
    y0: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h * 3];
    let inv_cell = 1.0 / grid.cell;
    let gh = grid.gh;
    let gw = grid.gw;
    let bins = grid.bins;

    out.par_chunks_mut(w * 3)
        .zip(img_rgb01.par_chunks(w * 3))
        .enumerate()
        .for_each(|(y, (dst_row, src_row))| {
            let gy = ((sy * (y + y0) as f32 + 0.5 * sy - 0.5) * inv_cell).clamp(0.0, (gh - 1) as f32);
            let yi = (gy.floor() as usize).min(gh.saturating_sub(2));
            let fy = gy - yi as f32;

            for x in 0..w {
                let gx = ((sx * x as f32 + 0.5 * sx - 0.5) * inv_cell).clamp(0.0, (gw - 1) as f32);
                let xi = (gx.floor() as usize).min(gw.saturating_sub(2));
                let fx = gx - xi as f32;

                let r = src_row[x * 3];
                let g = src_row[x * 3 + 1];
                let b = src_row[x * 3 + 2];
                let zz = luma01([r, g, b]) * (bins - 1) as f32;
                let zi = (zz.floor() as usize).min(bins.saturating_sub(2));
                let fz = zz - zi as f32;

                let mut o0 = 0.0f32;
                let mut o1 = 0.0f32;
                let mut o2 = 0.0f32;

                for dz in 0..2 {
                    let wz = if dz == 1 { fz } else { 1.0 - fz };
                    let zz_i = (zi + dz).min(bins - 1);
                    for dy in 0..2 {
                        let wy = if dy == 1 { fy } else { 1.0 - fy };
                        let yy_i = (yi + dy).min(gh - 1);
                        for dx in 0..2 {
                            let wx = if dx == 1 { fx } else { 1.0 - fx };
                            let xx_i = (xi + dx).min(gw - 1);
                            let ww = wx * wy * wz;
                            let m = &grid.matrices[(yy_i * gw + xx_i) * bins + zz_i];
                            o0 += ww * (m[0][0] * r + m[0][1] * g + m[0][2] * b + m[0][3]);
                            o1 += ww * (m[1][0] * r + m[1][1] * g + m[1][2] * b + m[1][3]);
                            o2 += ww * (m[2][0] * r + m[2][1] * g + m[2][2] * b + m[2][3]);
                        }
                    }
                }
                dst_row[x * 3] = o0.clamp(0.0, 1.0);
                dst_row[x * 3 + 1] = o1.clamp(0.0, 1.0);
                dst_row[x * 3 + 2] = o2.clamp(0.0, 1.0);
            }
        });
    out
}

/// Robust noise standard deviation estimator (Immerkaer 1996 median form) on a luminance image `g` (`0..1`).
pub fn noise_sigma_immerkaer(g: &[f32], w: usize, h: usize) -> f32 {
    if w < 5 || h < 5 {
        return 0.0;
    }
    let mut res = Vec::with_capacity((w - 4) * (h - 4));
    for y in 2..(h - 2) {
        for x in 2..(w - 2) {
            let v = g[(y - 1) * w + (x - 1)] - 2.0 * g[(y - 1) * w + x] + g[(y - 1) * w + (x + 1)]
                - 2.0 * g[y * w + (x - 1)] + 4.0 * g[y * w + x] - 2.0 * g[y * w + (x + 1)]
                + g[(y + 1) * w + (x - 1)] - 2.0 * g[(y + 1) * w + x] + g[(y + 1) * w + (x + 1)];
            res.push(v.abs());
        }
    }
    if res.is_empty() {
        return 0.0;
    }
    res.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    res[res.len() / 2] / 0.6745 / 6.0
}

/// Detect places where the AI version changed content (`w * h` in `0..1`) via multi-scale band-pass
/// normalised cross-correlation above each image's Immerkaer noise floor.
pub fn changed_map(work_rgb01: &[f32], ai_rgb01: &[f32], w: usize, h: usize) -> Vec<f32> {
    let n = w * h;
    if n == 0 {
        return vec![];
    }
    let mut ga = vec![0.0f32; n];
    let mut gb = vec![0.0f32; n];
    for i in 0..n {
        ga[i] = luma01([work_rgb01[i * 3], work_rgb01[i * 3 + 1], work_rgb01[i * 3 + 2]]);
        gb[i] = luma01([ai_rgb01[i * 3], ai_rgb01[i * 3 + 1], ai_rgb01[i * 3 + 2]]);
    }
    let na = noise_sigma_immerkaer(&ga, w, h);
    let nb = noise_sigma_immerkaer(&gb, w, h);

    let mut score = vec![0.0f32; n];
    for &(r_hi, r_lo, win) in &[(1usize, 4usize, 5usize), (3, 10, 12), (6, 20, 24)] {
        let a_hi = box_filter(&ga, w, h, r_hi);
        let a_lo = box_filter(&ga, w, h, r_lo);
        let b_hi = box_filter(&gb, w, h, r_hi);
        let b_lo = box_filter(&gb, w, h, r_lo);

        let mut ba = vec![0.0f32; n];
        let mut bb = vec![0.0f32; n];
        let mut aa = vec![0.0f32; n];
        let mut b2 = vec![0.0f32; n];
        let mut ab = vec![0.0f32; n];
        for i in 0..n {
            ba[i] = a_hi[i] - a_lo[i];
            bb[i] = b_hi[i] - b_lo[i];
            aa[i] = ba[i] * ba[i];
            b2[i] = bb[i] * bb[i];
            ab[i] = ba[i] * bb[i];
        }
        let va = box_filter(&aa, w, h, win);
        let vb = box_filter(&b2, w, h, win);
        let cab = box_filter(&ab, w, h, win);

        let fa = (2.8 * na).max(0.005);
        let fb = (2.8 * nb).max(0.005);
        for i in 0..n {
            let ncc = cab[i] / (va[i] * vb[i] + 1e-12).sqrt();
            let ga_gate = (va[i].sqrt() / fa - 1.0).clamp(0.0, 1.0);
            let gb_gate = (vb[i].sqrt() / fb - 1.0).clamp(0.0, 1.0);
            let gate = ga_gate.max(gb_gate);
            let s = ((0.55 - ncc) / 0.35).clamp(0.0, 1.0) * gate;
            if s > score[i] {
                score[i] = s;
            }
        }
    }
    let sm = box_filter(&score, w, h, 3);
    sm.into_iter().map(|v| ((v - 0.25) / 0.25).clamp(0.0, 1.0)).collect()
}

/// Pre-sliced preview RGB & CIELAB cache for instant (< 10 ms) `Kolor` / `Jasność` slider updates.
#[derive(Debug, Clone)]
pub struct PreviewCache {
    pub w: usize,
    pub h: usize,
    pub orig_rgb01: Vec<f32>,
    pub orig_lab: Vec<[f32; 3]>,
    pub prot_rgb01: Vec<f32>,
    pub prot_lab: Vec<[f32; 3]>,
    pub raw_rgb01: Vec<f32>,
    pub raw_lab: Vec<[f32; 3]>,
}

impl PreviewCache {
    pub fn build(
        orig_rgb01: Vec<f32>,
        w: usize,
        h: usize,
        grid_prot: &BilateralGrid,
        grid_raw: &BilateralGrid,
        sx: f32,
        sy: f32,
    ) -> Self {
        let n = w * h;
        let prot_rgb01 = slice_bgu(&orig_rgb01, w, h, grid_prot, sx, sy, 0);
        let raw_rgb01 = slice_bgu(&orig_rgb01, w, h, grid_raw, sx, sy, 0);
        let to_lab_vec = |buf: &[f32]| -> Vec<[f32; 3]> {
            let mut v = vec![[0.0f32; 3]; n];
            v.par_iter_mut().enumerate().for_each(|(i, dst)| {
                *dst = rgb01_to_lab([buf[i * 3], buf[i * 3 + 1], buf[i * 3 + 2]]);
            });
            v
        };
        let orig_lab = to_lab_vec(&orig_rgb01);
        let prot_lab = to_lab_vec(&prot_rgb01);
        let raw_lab = to_lab_vec(&raw_rgb01);
        Self { w, h, orig_rgb01, orig_lab, prot_rgb01, prot_lab, raw_rgb01, raw_lab }
    }

    pub fn mix_rgb8(&self, params: TransferParams, mask: Option<&[f32]>) -> Vec<u8> {
        let n = self.w * self.h;
        let k = params.kolor.clamp(0.0, 1.0);
        let j = params.jasnosc.clamp(0.0, 1.0);
        if k >= 0.999 && j >= 0.999 && mask.is_none() {
            let src = if params.protect { &self.prot_rgb01 } else { &self.raw_rgb01 };
            return src.iter().map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8).collect();
        }
        let lr = if params.protect { &self.prot_lab } else { &self.raw_lab };
        let mut out = vec![0u8; n * 3];
        out.par_chunks_mut(3).enumerate().for_each(|(i, dst)| {
            let lo = self.orig_lab[i];
            let lt = lr[i];
            let jm = j * mask.and_then(|m| m.get(i).copied()).unwrap_or(1.0).clamp(0.0, 1.0);
            let lab = [
                lo[0] + jm * (lt[0] - lo[0]),
                lo[1] + k * (lt[1] - lo[1]),
                lo[2] + k * (lt[2] - lo[2]),
            ];
            let rgb = lab_to_rgb01(lab);
            dst[0] = (rgb[0] * 255.0 + 0.5) as u8;
            dst[1] = (rgb[1] * 255.0 + 0.5) as u8;
            dst[2] = (rgb[2] * 255.0 + 0.5) as u8;
        });
        out
    }
}
