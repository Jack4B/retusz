//! Non-destructive crop, straighten, and maximum inscribed rectangle math.
//! Author: Jack4B.

/// Largest axis-aligned rectangle `[x0, y0, x1, y1]` in normalised coordinates `0..1`
/// inscribed inside a frame of aspect `W/H` rotated by `angle_deg` degrees.
pub fn max_inscribed_rect(w: f32, h: f32, angle_deg: f32) -> [f32; 4] {
    let a = angle_deg.abs().to_radians();
    if a < 1e-6 || w <= 0.0 || h <= 0.0 {
        return [0.0, 0.0, 1.0, 1.0];
    }
    let (sa, ca) = (a.sin(), a.cos());
    let (wr, hr);
    let short = w.min(h);
    if short <= 2.0 * sa * ca * w.max(h) {
        let x = 0.5 * short;
        if w >= h {
            wr = x / sa;
            hr = x / ca;
        } else {
            wr = x / ca;
            hr = x / sa;
        }
    } else {
        let cos2 = ca * ca - sa * sa;
        wr = (w * ca - h * sa) / cos2;
        hr = (h * ca - w * sa) / cos2;
    }
    let fw = (wr / w).clamp(0.05, 1.0);
    let fh = (hr / h).clamp(0.05, 1.0);
    [0.5 - 0.5 * fw, 0.5 - 0.5 * fh, 0.5 + 0.5 * fw, 0.5 + 0.5 * fh]
}

/// Clamp a normalised crop `[x0, y0, x1, y1]` inside the rotated inscribed rectangle
/// and optional target aspect ratio `aspect_wh` (`W / H`).
pub fn clamp_crop(crop: [f32; 4], w: f32, h: f32, angle_deg: f32, aspect_wh: Option<f32>) -> [f32; 4] {
    let lim = max_inscribed_rect(w, h, angle_deg);
    let mut x0 = crop[0].max(lim[0]);
    let mut y0 = crop[1].max(lim[1]);
    let mut x1 = crop[2].min(lim[2]);
    let mut y1 = crop[3].min(lim[3]);
    if x1 - x0 < 0.05 {
        let cx = 0.5 * (lim[0] + lim[2]);
        x0 = cx - 0.025;
        x1 = cx + 0.025;
    }
    if y1 - y0 < 0.05 {
        let cy = 0.5 * (lim[1] + lim[3]);
        y0 = cy - 0.025;
        y1 = cy + 0.025;
    }
    if let Some(target_r) = aspect_wh {
        if target_r > 1e-4 && w > 0.0 && h > 0.0 {
            let cx = 0.5 * (x0 + x1);
            let cy = 0.5 * (y0 + y1);
            let mut pw = (x1 - x0) * w;
            let mut ph = (y1 - y0) * h;
            if pw / ph > target_r {
                pw = ph * target_r;
            } else {
                ph = pw / target_r;
            }
            let nw = pw / w;
            let nh = ph / h;
            x0 = (cx - 0.5 * nw).max(lim[0]);
            y0 = (cy - 0.5 * nh).max(lim[1]);
            x1 = (x0 + nw).min(lim[2]);
            y1 = (y0 + nh).min(lim[3]);
        }
    }
    [x0, y0, x1, y1]
}
