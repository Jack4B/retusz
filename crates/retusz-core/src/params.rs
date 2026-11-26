//! Parameters, presets (`*.retusz-preset.json`), and non-destructive sidecars (`*.retusz.json`).
//! Author: Jack4B.

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

fn default_one() -> f32 { 1.0 }
fn default_dnb() -> f32 { 0.40 }
fn default_plastic() -> f32 { 0.30 }
fn default_true() -> bool { true }
fn default_aspect() -> String { "free".to_string() }
fn default_preset_cat() -> String { "Własne".to_string() }

fn default_sharpen_mode() -> String { "portrait".to_string() }

/// Non-destructive photographic and portrait adjustment parameters.
/// Fully backwards-compatible with `*.retusz.json` and `retusz-seria.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Params {
    #[serde(default = "default_one")]
    pub strength: f32,
    #[serde(default = "default_one")]
    pub skin: f32,
    #[serde(default)]
    pub skin_even: f32,
    #[serde(default)]
    pub skin_cast: f32,      //  0.0 .. +1.0 ('Czysta skóra' OKLCh cast reduction)
    #[serde(default)]
    pub skin_hue: f32,       // -1.0 .. +1.0 ('Odcień skóry' OKLCh hue rotation)
    #[serde(default)]
    pub deshine: f32,        //  0.0 .. +1.0 ('Redukcja połysku' Dichromatic SUV De-Shine)
    #[serde(default)]
    pub blemish: f32,        //  0.0 .. +1.0 ('Niedoskonałości' Two-band pore-preserving auto-heal)
    #[serde(default)]
    pub eyes_teeth: f32,     //  0.0 .. +1.0 ('Oczy i zęby' Anatomical iris/sclera/teeth enhancer)
    #[serde(default)]
    pub exposure: f32,       // -2.0 .. +2.0 EV
    #[serde(default)]
    pub contrast: f32,       // -1.0 .. +1.0
    #[serde(default)]
    pub brightness: f32,     // -1.0 .. +1.0 ('Jasność / Półtony' Capture One midtone curve)
    #[serde(default)]
    pub dehaze: f32,         // -1.0 .. +1.0 ('Zamglenie / Dehaze' OKLab veil removal)
    #[serde(default)]
    pub highlights: f32,     // -1.0 .. +1.0
    #[serde(default)]
    pub shadows: f32,        // -1.0 .. +1.0
    #[serde(default)]
    pub whites: f32,         // -1.0 .. +1.0
    #[serde(default)]
    pub blacks: f32,         // -1.0 .. +1.0
    #[serde(default)]
    pub clarity: f32,        // -1.0 .. +1.0
    #[serde(default)]
    pub texture: f32,        // -1.0 .. +1.0 ('Struktura')
    #[serde(default)]
    pub sharpen_amount: f32, //  0.0 .. +1.0 ('Ostrość')
    #[serde(default = "default_sharpen_mode")]
    pub sharpen_mode: String,// 'portrait' | 'rl' | 'edges' | 'soft'
    #[serde(default)]
    pub denoise: f32,        //  0.0 .. +1.0 ('Odszumianie')
    #[serde(default)]
    pub vibrance: f32,       // -1.0 .. +1.0
    #[serde(default)]
    pub saturation: f32,     // -1.0 .. +1.0
    #[serde(default)]
    pub vignette: f32,       // -1.0 .. +1.0
    #[serde(default = "default_dnb", alias = "dnb")]
    pub db: f32,             //  0.0 .. +1.0
    #[serde(default = "default_plastic")]
    pub plastic: f32,        //  0.0 .. +1.0
    #[serde(default)]
    pub light_angle: Option<f32>,
    #[serde(default, alias = "vivid")]
    pub wide: f32,           //  0.0 .. +1.0
    #[serde(default)]
    pub temp: f32,           // -1.0 .. +1.0
    #[serde(default)]
    pub tint: f32,           // -1.0 .. +1.0
    #[serde(default)]
    pub angle: f32,          // -45.0 .. +45.0 deg
    #[serde(default)]
    pub crop: Option<[f32; 4]>,
    #[serde(default = "default_aspect", alias = "crop_aspect")]
    pub aspect: String,
    #[serde(default)]
    pub rating: u8,
    #[serde(default)]
    pub reference_path: Option<String>,
    #[serde(default)]
    pub ref_colorchecker: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            strength: 1.0,
            skin: 1.0,
            skin_even: 0.0,
            skin_cast: 0.0,
            skin_hue: 0.0,
            deshine: 0.0,
            blemish: 0.0,
            eyes_teeth: 0.0,
            exposure: 0.0,
            contrast: 0.0,
            brightness: 0.0,
            dehaze: 0.0,
            highlights: 0.0,
            shadows: 0.0,
            whites: 0.0,
            blacks: 0.0,
            clarity: 0.0,
            texture: 0.0,
            sharpen_amount: 0.0,
            sharpen_mode: "portrait".to_string(),
            denoise: 0.0,
            vibrance: 0.0,
            saturation: 0.0,
            vignette: 0.0,
            db: 0.40,
            plastic: 0.30,
            light_angle: None,
            wide: 0.0,
            temp: 0.0,
            tint: 0.0,
            angle: 0.0,
            crop: None,
            aspect: "orig".to_string(),
            rating: 0,
            reference_path: None,
            ref_colorchecker: false,
        }
    }
}

/// Parameters for AI Grading Transfer (`Przenieś grading z AI`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TransferParams {
    #[serde(default = "default_one")]
    pub kolor: f32,
    #[serde(default = "default_one")]
    pub jasnosc: f32,
    #[serde(default = "default_true")]
    pub protect: bool,
}

impl Default for TransferParams {
    fn default() -> Self {
        Self { kolor: 1.0, jasnosc: 1.0, protect: true }
    }
}

/// A named photographic preset (`*.retusz-preset.json`).
/// Does not overwrite per-photo crop/straighten geometry when applied.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Preset {
    pub name: String,
    #[serde(default = "default_preset_cat")]
    pub category: String,
    #[serde(default)]
    pub builtin: bool,
    pub params: Params,
}

impl Preset {
    /// Apply preset's tonal, colour, and portrait parameters while keeping the photo's crop/angle and reference settings.
    pub fn apply_to(&self, current: &Params) -> Params {
        let mut out = self.params.clone();
        out.angle = current.angle;
        out.crop = current.crop;
        out.aspect = current.aspect.clone();
        out.strength = current.strength;
        out.skin = current.skin;
        out
    }

    pub fn save_to_file(&self, path: &Path) -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let txt = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        fs::write(path, txt)
    }

    pub fn load_from_file(path: &Path) -> std::io::Result<Self> {
        let txt = fs::read_to_string(path)?;
        serde_json::from_str(&txt)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

/// Built-in studio portrait presets.
pub fn builtin_presets() -> Vec<Preset> {
    vec![
        Preset {
            name: "Portret naturalny".to_string(),
            category: "Portret".to_string(),
            builtin: true,
            params: Params {
                db: 0.40,
                plastic: 0.30,
                ..Params::default()
            },
        },
        Preset {
            name: "Ciepły złoty".to_string(),
            category: "Portret".to_string(),
            builtin: true,
            params: Params {
                exposure: 0.15,
                contrast: 0.12,
                highlights: -0.20,
                shadows: 0.22,
                whites: 0.08,
                temp: 0.18,
                tint: 0.04,
                vibrance: 0.18,
                db: 0.45,
                plastic: 0.35,
                vignette: -0.15,
                ..Params::default()
            },
        },
        Preset {
            name: "Filmowy miękki".to_string(),
            category: "Filmowe".to_string(),
            builtin: true,
            params: Params {
                exposure: 0.05,
                contrast: -0.12,
                highlights: -0.30,
                shadows: 0.28,
                whites: -0.10,
                blacks: 0.15,
                clarity: -0.22,
                vibrance: 0.10,
                saturation: -0.08,
                db: 0.50,
                plastic: 0.25,
                vignette: -0.20,
                ..Params::default()
            },
        },
        Preset {
            name: "High-Key (jasny)".to_string(),
            category: "Studyjne".to_string(),
            builtin: true,
            params: Params {
                exposure: 0.45,
                contrast: -0.08,
                highlights: -0.25,
                shadows: 0.35,
                whites: 0.15,
                blacks: 0.10,
                clarity: -0.12,
                vibrance: 0.10,
                db: 0.50,
                plastic: 0.20,
                ..Params::default()
            },
        },
        Preset {
            name: "Studyjny kontrast 3D".to_string(),
            category: "Studyjne".to_string(),
            builtin: true,
            params: Params {
                contrast: 0.24,
                highlights: -0.18,
                shadows: 0.15,
                whites: 0.12,
                blacks: -0.10,
                clarity: 0.25,
                vibrance: 0.15,
                db: 0.55,
                plastic: 0.55,
                vignette: -0.22,
                ..Params::default()
            },
        },
        Preset {
            name: "B&W Klasyczny".to_string(),
            category: "Monochrom".to_string(),
            builtin: true,
            params: Params {
                exposure: 0.08,
                contrast: 0.28,
                highlights: -0.22,
                shadows: 0.18,
                whites: 0.14,
                blacks: -0.12,
                clarity: 0.20,
                saturation: -1.0,
                vibrance: -1.0,
                db: 0.50,
                plastic: 0.45,
                vignette: -0.25,
                ..Params::default()
            },
        },
    ]
}

/// Sidecar path next to the photo: `IMG_001.jpg.retusz.json`.
pub fn sidecar_path(photo: &Path) -> PathBuf {
    let mut os = photo.as_os_str().to_owned();
    os.push(".retusz.json");
    PathBuf::from(os)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SidecarFile {
    #[serde(default = "default_version")]
    pub version: u32,
    pub params: Params,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brush_png_b64: Option<String>,
}

fn default_version() -> u32 { 1 }

pub fn save_sidecar(photo: &Path, params: &Params) -> std::io::Result<()> {
    let sp = sidecar_path(photo);
    let sc = SidecarFile { version: 1, params: params.clone(), brush_png_b64: None };
    let txt = serde_json::to_string_pretty(&sc)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    fs::write(sp, txt)
}

pub fn load_sidecar(photo: &Path) -> Option<Params> {
    let sp = sidecar_path(photo);
    let txt = fs::read_to_string(sp).ok()?;
    let sc: SidecarFile = serde_json::from_str(&txt).ok()?;
    Some(sc.params)
}
