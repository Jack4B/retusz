//! 256-bin R, G, B, L histogram and highlight/shadow clipping statistics.
//! Author: Jack4B.

#[derive(Debug, Clone)]
pub struct Histogram {
    pub r: [f32; 256],
    pub g: [f32; 256],
    pub b: [f32; 256],
    pub l: [f32; 256],
    pub hi_pct: f32,
    pub lo_pct: f32,
    pub hi_channels: [bool; 3],
    pub lo_channels: [bool; 3],
}

impl Default for Histogram {
    fn default() -> Self {
        Self {
            r: [0.0; 256],
            g: [0.0; 256],
            b: [0.0; 256],
            l: [0.0; 256],
            hi_pct: 0.0,
            lo_pct: 0.0,
            hi_channels: [false; 3],
            lo_channels: [false; 3],
        }
    }
}

pub fn compute_histogram(rgb8: &[u8]) -> Histogram {
    let n_px = rgb8.len() / 3;
    if n_px == 0 {
        return Histogram::default();
    }
    let mut hr = [0u32; 256];
    let mut hg = [0u32; 256];
    let mut hb = [0u32; 256];
    let mut hl = [0u32; 256];
    let mut hi_any = 0u32;
    let mut lo_any = 0u32;
    let mut hi_c = [0u32; 3];
    let mut lo_c = [0u32; 3];

    for px in rgb8.chunks_exact(3) {
        let (r, g, b) = (px[0], px[1], px[2]);
        hr[r as usize] += 1;
        hg[g as usize] += 1;
        hb[b as usize] += 1;
        let lum = ((77 * r as u32 + 150 * g as u32 + 29 * b as u32 + 128) >> 8).min(255) as usize;
        hl[lum] += 1;

        if r >= 254 || g >= 254 || b >= 254 {
            hi_any += 1;
        }
        if r <= 1 && g <= 1 && b <= 1 {
            lo_any += 1;
        }
        if r >= 254 { hi_c[0] += 1; }
        if g >= 254 { hi_c[1] += 1; }
        if b >= 254 { hi_c[2] += 1; }
        if r <= 1 { lo_c[0] += 1; }
        if g <= 1 { lo_c[1] += 1; }
        if b <= 1 { lo_c[2] += 1; }
    }

    // Normalise by 99th percentile bin (excluding endpoints 0 and 255) for clean display
    let norm_curve = |bins: &[u32; 256]| -> [f32; 256] {
        let mut inner: Vec<u32> = bins[1..255].to_vec();
        inner.sort_unstable();
        let p99 = inner[(inner.len() * 98) / 100].max(1) as f32;
        let mut out = [0.0f32; 256];
        for i in 0..256 {
            out[i] = (bins[i] as f32 / p99).min(1.0);
        }
        out
    };

    let inv_n = 100.0 / n_px as f32;
    let thr = (n_px as f32 * 0.0005) as u32;
    Histogram {
        r: norm_curve(&hr),
        g: norm_curve(&hg),
        b: norm_curve(&hb),
        l: norm_curve(&hl),
        hi_pct: hi_any as f32 * inv_n,
        lo_pct: lo_any as f32 * inv_n,
        hi_channels: [hi_c[0] > thr, hi_c[1] > thr, hi_c[2] > thr],
        lo_channels: [lo_c[0] > thr, lo_c[1] > thr, lo_c[2] > thr],
    }
}
