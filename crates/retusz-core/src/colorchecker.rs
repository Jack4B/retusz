//! 24-patch ColorChecker Classic automatic detection, Root-Polynomial + Bradford colour calibration
//! (Finlayson, Mackiewicz & Hurlbert 2015, IEEE Trans. Image Processing), and CIEDE2000 report.

use crate::color::{
    delta_e2000, lin_to_srgb, lin_to_srgb_scalar, rgb01_to_lab, srgb_to_lin, srgb_to_lin_scalar,
    Lut3D,
};
use crate::tone::{lin_to_oklab, oklab_to_lin_gamut};
use serde::{Deserialize, Serialize};

/// Standard 24-patch ColorChecker Classic sRGB D65 reference values (BabelColor / X-Rite D65).
pub const CC24_SRGB: [[u8; 3]; 24] = [
    [115, 82, 68],
    [194, 150, 130],
    [98, 122, 157],
    [87, 108, 67],
    [133, 128, 177],
    [103, 189, 170],
    [214, 126, 44],
    [80, 91, 166],
    [193, 90, 99],
    [94, 60, 108],
    [157, 188, 64],
    [224, 163, 46],
    [56, 61, 150],
    [70, 148, 73],
    [175, 54, 60],
    [231, 199, 31],
    [187, 86, 149],
    [8, 133, 161],
    [243, 243, 242],
    [200, 200, 200],
    [160, 160, 160],
    [122, 122, 121],
    [85, 85, 85],
    [52, 52, 52],
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartReport {
    pub de00_before: f32,
    pub de00_after: f32,
    pub max_de00_before: f32,
    pub max_de00_after: f32,
    pub skin_de00: f32,
    pub gray_de00: f32,
    pub patch_de00_before: Vec<f32>,
    pub patch_de00_after: Vec<f32>,
}

impl ChartReport {
    pub fn summary_pl(&self) -> String {
        format!(
            "ColorChecker 24: ΔE₀₀ {:.1} → {:.2} (skóra: {:.2} · szarości: {:.2} · max: {:.2})",
            self.de00_before, self.de00_after, self.skin_de00, self.gray_de00, self.max_de00_after
        )
    }
}

#[derive(Debug, Clone)]
pub struct ChartCalibration {
    pub centres: [[f32; 2]; 24],
    pub patch_radius: f32,
    pub measured_srgb: [[f32; 3]; 24],
    pub calibrated_srgb: [[u8; 3]; 24],
    pub target_srgb: [[u8; 3]; 24],
    pub lut: Lut3D,
    pub report: ChartReport,
}

fn root_poly_basis(r: f64, g: f64, b: f64) -> [f64; 7] {
    let r = r.max(0.0);
    let g = g.max(0.0);
    let b = b.max(0.0);
    [
        r,
        g,
        b,
        (r * g).sqrt(),
        (g * b).sqrt(),
        (b * r).sqrt(),
        1.0,
    ]
}

fn solve_7x7(mut a: [[f64; 7]; 7], mut b: [[f64; 3]; 7]) -> [[f64; 3]; 7] {
    for i in 0..7 {
        let mut piv = i;
        let mut max_v = a[i][i].abs();
        for r in (i + 1)..7 {
            if a[r][i].abs() > max_v {
                max_v = a[r][i].abs();
                piv = r;
            }
        }
        a.swap(i, piv);
        b.swap(i, piv);
        let diag = a[i][i];
        if diag.abs() < 1e-12 {
            continue;
        }
        for c in i..7 {
            a[i][c] /= diag;
        }
        for c in 0..3 {
            b[i][c] /= diag;
        }
        for r in 0..7 {
            if r == i {
                continue;
            }
            let f = a[r][i];
            for c in i..7 {
                a[r][c] -= f * a[i][c];
            }
            for c in 0..3 {
                b[r][c] -= f * b[i][c];
            }
        }
    }
    b
}

#[inline]
fn lab_f64(rgb01: [f32; 3]) -> [f64; 3] {
    let l = rgb01_to_lab(rgb01);
    [l[0] as f64, l[1] as f64, l[2] as f64]
}

/// Calibrate 24 measured sRGB patches (0..255) to the D65 ColorChecker reference using weighted
/// degree-2 Root-Polynomial Color Correction (Finlayson, Mackiewicz & Hurlbert 2015) and bake a 33^3 LUT.
pub fn calibrate_from_patches(
    measured_srgb: &[[f32; 3]; 24],
    target_srgb: Option<&[[u8; 3]; 24]>,
    centres: [[f32; 2]; 24],
    patch_radius: f32,
) -> ChartCalibration {
    let tgt = target_srgb.copied().unwrap_or(CC24_SRGB);
    let mut ata = [[0.0f64; 7]; 7];
    let mut aty = [[0.0f64; 3]; 7];
    let lam = 1e-4f64;
    for i in 0..7 {
        ata[i][i] = lam;
    }
    aty[0][0] = lam;
    aty[1][1] = lam;
    aty[2][2] = lam;

    let mut meas_lin = [[0.0f64; 3]; 24];
    for i in 0..24 {
        let w = if i < 2 {
            3.5f64
        } else if i >= 18 {
            3.0f64
        } else {
            1.0f64
        };
        let ml = [
            srgb_to_lin_scalar((measured_srgb[i][0] / 255.0).clamp(0.0, 1.0)) as f64,
            srgb_to_lin_scalar((measured_srgb[i][1] / 255.0).clamp(0.0, 1.0)) as f64,
            srgb_to_lin_scalar((measured_srgb[i][2] / 255.0).clamp(0.0, 1.0)) as f64,
        ];
        meas_lin[i] = ml;
        let tl = [
            srgb_to_lin_scalar(tgt[i][0] as f32 / 255.0) as f64,
            srgb_to_lin_scalar(tgt[i][1] as f32 / 255.0) as f64,
            srgb_to_lin_scalar(tgt[i][2] as f32 / 255.0) as f64,
        ];
        let phi = root_poly_basis(ml[0], ml[1], ml[2]);
        for r in 0..7 {
            for c in 0..7 {
                ata[r][c] += w * phi[r] * phi[c];
            }
            for c in 0..3 {
                aty[r][c] += w * phi[r] * tl[c];
            }
        }
    }

    let m = solve_7x7(ata, aty);

    let mut calibrated_srgb = [[0u8; 3]; 24];
    let mut de_before = Vec::with_capacity(24);
    let mut de_after = Vec::with_capacity(24);

    for i in 0..24 {
        let phi = root_poly_basis(meas_lin[i][0], meas_lin[i][1], meas_lin[i][2]);
        let mut cl = [0.0f32; 3];
        for c in 0..3 {
            let mut sum = 0.0f64;
            for r in 0..7 {
                sum += phi[r] * m[r][c];
            }
            cl[c] = sum.clamp(0.0, 1.0) as f32;
        }
        let cs = [
            (lin_to_srgb_scalar(cl[0]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            (lin_to_srgb_scalar(cl[1]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
            (lin_to_srgb_scalar(cl[2]).clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        ];
        calibrated_srgb[i] = cs;

        let lab_m = lab_f64([
            (measured_srgb[i][0] / 255.0).clamp(0.0, 1.0),
            (measured_srgb[i][1] / 255.0).clamp(0.0, 1.0),
            (measured_srgb[i][2] / 255.0).clamp(0.0, 1.0),
        ]);
        let lab_c = lab_f64([
            cs[0] as f32 / 255.0,
            cs[1] as f32 / 255.0,
            cs[2] as f32 / 255.0,
        ]);
        let lab_t = lab_f64([
            tgt[i][0] as f32 / 255.0,
            tgt[i][1] as f32 / 255.0,
            tgt[i][2] as f32 / 255.0,
        ]);
        de_before.push(delta_e2000(lab_m, lab_t) as f32);
        de_after.push(delta_e2000(lab_c, lab_t) as f32);
    }

    let n = 33usize;
    let mut lut = Lut3D::identity(n);
    for ir in 0..n {
        for ig in 0..n {
            for ib in 0..n {
                let idx = (ir * n + ig) * n + ib;
                let s = lut.data[idx];
                let [rl, gl, bl] = srgb_to_lin(s);
                let phi = root_poly_basis(rl as f64, gl as f64, bl as f64);
                let mut cl = [0.0f32; 3];
                for c in 0..3 {
                    let mut sum = 0.0f64;
                    for r in 0..7 {
                        sum += phi[r] * m[r][c];
                    }
                    cl[c] = sum.clamp(0.0, 1.5) as f32;
                }
                let ok = lin_to_oklab(cl);
                let gam = oklab_to_lin_gamut(ok);
                lut.data[idx] = lin_to_srgb(gam);
            }
        }
    }

    let mean_b = de_before.iter().sum::<f32>() / 24.0;
    let mean_a = de_after.iter().sum::<f32>() / 24.0;
    let max_b = de_before.iter().cloned().fold(0.0f32, f32::max);
    let max_a = de_after.iter().cloned().fold(0.0f32, f32::max);
    let skin_de = 0.5 * (de_after[0] + de_after[1]);
    let gray_de = de_after[18..24].iter().sum::<f32>() / 6.0;

    ChartCalibration {
        centres,
        patch_radius,
        measured_srgb: *measured_srgb,
        calibrated_srgb,
        target_srgb: tgt,
        lut,
        report: ChartReport {
            de00_before: mean_b,
            de00_after: mean_a,
            max_de00_before: max_b,
            max_de00_after: max_a,
            skin_de00: skin_de,
            gray_de00: gray_de,
            patch_de00_before: de_before,
            patch_de00_after: de_after,
        },
    }
}
