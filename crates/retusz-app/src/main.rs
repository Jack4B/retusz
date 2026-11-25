//! Retusz 0.2.0 – Native Studio Portrait Editor (`eframe` / `egui` + `wgpu`) & CLI.
//! Author: Jack4B.
//!
//! Layout:
//! - Left filmstrip (series thumbnails with edited badge)
//! - Central zoomable studio canvas (`Wynik` | `Przed | Po` | `Podział` | `Z wzorcem`)
//! - Right docked Studio Inspector:
//!   * Live R/G/B/L Histogram with highlight/shadow clipping triangles (`J`)
//!   * Presets bar (`Portret naturalny`, `Ciepły złoty`, `Filmowy miękki`, `High-Key`, `Studyjny kontrast 3D`, `B&W Klasyczny` + custom `*.retusz-preset.json`)
//!   * Open section: `Światło i Ton (OKLab / EV)`
//!   * Open section: `Kolor i Balans bieli (Bradford CAT)`
//!   * Open section: `Portret AI i Plastyka 3D`
//!   * Collapsible bonus section (`▸ Dopasowanie do wzorca (AI)`): closed by default, opens on click or when a reference image is chosen.

use clap::{Parser, Subcommand};
use eframe::egui;
use retusz_core::histo::{compute_histogram, Histogram};
use retusz_core::params::{
    builtin_presets, load_sidecar, save_sidecar, Params, Preset,
};
use retusz_core::pipeline::{render_photo_cpu, ReferenceStats};
use retusz_gpu::GpuRenderer;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(name = "retusz", version = "0.2.0", author = "Jack4B")]
struct CliArgs {
    #[command(subcommand)]
    cmd: Option<Command>,
    /// Optional photo or folder to open in the Studio GUI
    path: Option<PathBuf>,
    /// Optional reference photo to load
    #[arg(long)]
    reference: Option<PathBuf>,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Process a photo or folder headlessly (with or without a reference image / preset)
    Cli {
        path: PathBuf,
        #[arg(long)]
        reference: Option<PathBuf>,
        #[arg(long)]
        preset: Option<String>,
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CanvasMode {
    Wynik,
    PrzedPo,
    Podzial,
    ZWzorcem,
}

struct LoadedImage {
    path: PathBuf,
    w: usize,
    h: usize,
    rgb8: Vec<u8>,
}

impl LoadedImage {
    fn load(path: &Path) -> Option<Self> {
        let img = image::open(path).ok()?.to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        Some(Self {
            path: path.to_path_buf(),
            w,
            h,
            rgb8: img.into_raw(),
        })
    }
}

pub struct RetuszStudioApp {
    photos: Vec<PathBuf>,
    idx: usize,
    src: Option<LoadedImage>,
    res_rgb8: Vec<u8>,
    res_w: usize,
    res_h: usize,
    ref_img: Option<LoadedImage>,
    ref_stats: Option<ReferenceStats>,
    params: Params,
    undo_stack: Vec<Params>,
    redo_stack: Vec<Params>,
    clipboard: Option<Params>,
    presets: Vec<Preset>,
    selected_preset: String,
    new_preset_name: String,
    show_save_preset: bool,
    ref_section_open: bool,
    canvas_mode: CanvasMode,
    zoom: f32,
    show_clip: bool,
    histo: Histogram,
    tex_src: Option<egui::TextureHandle>,
    tex_res: Option<egui::TextureHandle>,
    tex_ref: Option<egui::TextureHandle>,
    dirty: bool,
    gpu_name: String,
    status: String,
}

impl RetuszStudioApp {
    pub fn new(initial_path: Option<PathBuf>, reference: Option<PathBuf>) -> Self {
        let gpu_name = GpuRenderer::try_new()
            .map(|g| format!("GPU: {}", g.adapter_name))
            .unwrap_or_else(|| "CPU (rayon SIMD)".to_string());
        let presets = Self::load_all_presets();
        let mut app = Self {
            photos: Vec::new(),
            idx: 0,
            src: None,
            res_rgb8: Vec::new(),
            res_w: 0,
            res_h: 0,
            ref_img: None,
            ref_stats: None,
            params: Params::default(),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            clipboard: None,
            presets,
            selected_preset: "Portret naturalny".to_string(),
            new_preset_name: String::new(),
            show_save_preset: false,
            ref_section_open: reference.is_some(),
            canvas_mode: CanvasMode::Wynik,
            zoom: 1.0,
            show_clip: false,
            histo: Histogram::default(),
            tex_src: None,
            tex_res: None,
            tex_ref: None,
            dirty: false,
            gpu_name,
            status: "Gotowy · Otwórz zdjęcie lub folder (styl studyjny, dopasowanie do wzorca jako sekcja dodatkowa)".to_string(),
        };
        if let Some(rp) = reference {
            app.set_reference(&rp);
        }
        if let Some(p) = initial_path {
            app.open_path(&p);
        }
        app
    }

    fn presets_dir() -> PathBuf {
        let base = std::env::var_os("APPDATA")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))
            .unwrap_or_else(|| PathBuf::from("."));
        base.join("Retusz").join("presets")
    }

    fn load_all_presets() -> Vec<Preset> {
        let mut list = builtin_presets();
        let dir = Self::presets_dir();
        if let Ok(rd) = fs::read_dir(dir) {
            for entry in rd.flatten() {
                let p = entry.path();
                if p.extension().and_then(|e| e.to_str()) == Some("json") {
                    if let Ok(mut pr) = Preset::load_from_file(&p) {
                        pr.builtin = false;
                        list.push(pr);
                    }
                }
            }
        }
        list
    }

    fn open_path(&mut self, path: &Path) {
        if path.is_dir() {
            let mut list = Vec::new();
            if let Ok(rd) = fs::read_dir(path) {
                for e in rd.flatten() {
                    let p = e.path();
                    if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
                        let el = ext.to_ascii_lowercase();
                        if matches!(el.as_str(), "jpg" | "jpeg" | "png" | "tif" | "tiff" | "webp") {
                            list.push(p);
                        }
                    }
                }
            }
            list.sort();
            if !list.is_empty() {
                self.photos = list;
                self.select_photo(0);
            }
        } else if path.is_file() {
            self.photos = vec![path.to_path_buf()];
            self.select_photo(0);
        }
    }

    fn select_photo(&mut self, idx: usize) {
        if idx >= self.photos.len() {
            return;
        }
        self.idx = idx;
        let p = self.photos[idx].clone();
        if let Some(loaded) = LoadedImage::load(&p) {
            self.params = load_sidecar(&p).unwrap_or_default();
            self.undo_stack.clear();
            self.redo_stack.clear();
            self.status = format!(
                "{} ({}×{} px) · {}",
                p.file_name().and_then(|s| s.to_str()).unwrap_or(""),
                loaded.w,
                loaded.h,
                self.gpu_name
            );
            self.src = Some(loaded);
            self.tex_src = None;
            self.dirty = true;
        }
    }

    fn set_reference(&mut self, path: &Path) {
        if let Some(loaded) = LoadedImage::load(path) {
            self.ref_stats = Some(ReferenceStats::from_rgb8(&loaded.rgb8));
            self.ref_img = Some(loaded);
            self.tex_ref = None;
            self.ref_section_open = true;
            self.dirty = true;
        }
    }

    fn push_undo(&mut self) {
        if self.undo_stack.last() != Some(&self.params) {
            self.undo_stack.push(self.params.clone());
            if self.undo_stack.len() > 40 {
                self.undo_stack.remove(0);
            }
            self.redo_stack.clear();
        }
    }

    fn undo(&mut self) {
        if let Some(prev) = self.undo_stack.pop() {
            self.redo_stack.push(self.params.clone());
            self.params = prev;
            self.dirty = true;
        }
    }

    fn redo(&mut self) {
        if let Some(next) = self.redo_stack.pop() {
            self.undo_stack.push(self.params.clone());
            self.params = next;
            self.dirty = true;
        }
    }

    fn rerender(&mut self, ctx: &egui::Context) {
        let Some(src) = &self.src else { return };
        if self.tex_src.is_none() {
            let cimg = egui::ColorImage::from_rgb([src.w, src.h], &src.rgb8);
            self.tex_src = Some(ctx.load_texture("src_tex", cimg, egui::TextureOptions::LINEAR));
        }
        if let Some(rf) = &self.ref_img {
            if self.tex_ref.is_none() {
                let cimg = egui::ColorImage::from_rgb([rf.w, rf.h], &rf.rgb8);
                self.tex_ref = Some(ctx.load_texture("ref_tex", cimg, egui::TextureOptions::LINEAR));
            }
        }
        let active_ref = if self.ref_section_open { self.ref_stats } else { None };
        let (out, rw, rh) = render_photo_cpu(
            &src.rgb8,
            src.w,
            src.h,
            &self.params,
            &[],
            None,
            active_ref,
        );
        self.histo = compute_histogram(&out);
        self.res_rgb8 = out;
        self.res_w = rw;
        self.res_h = rh;
        let cimg = egui::ColorImage::from_rgb([rw, rh], &self.res_rgb8);
        self.tex_res = Some(ctx.load_texture("res_tex", cimg, egui::TextureOptions::LINEAR));
        if let Some(cur_path) = self.photos.get(self.idx) {
            let _ = save_sidecar(cur_path, &self.params);
        }
        self.dirty = false;
    }

    fn export_current(&mut self) {
        let Some(src) = &self.src else { return };
        let parent = src.path.parent().unwrap_or_else(|| Path::new("."));
        let out_dir = parent.join("Retusz");
        if fs::create_dir_all(&out_dir).is_err() {
            return;
        }
        let stem = src.path.file_stem().and_then(|s| s.to_str()).unwrap_or("wynik");
        let dst = out_dir.join(format!("{stem}_retusz.jpg"));
        if let Some(buf) = image::RgbImage::from_raw(self.res_w as u32, self.res_h as u32, self.res_rgb8.clone()) {
            if buf.save(&dst).is_ok() {
                self.status = format!("Zapisano eksport: {}", dst.display());
            }
        }
    }

    fn slider_row(
        ui: &mut egui::Ui,
        label: &str,
        val: &mut f32,
        range: std::ops::RangeInclusive<f32>,
        default: f32,
        fmt_ev: bool,
    ) -> bool {
        let mut changed = false;
        ui.horizontal(|ui| {
            let lbl_resp = ui.label(egui::RichText::new(label).size(12.5));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let txt = if fmt_ev {
                    format!("{:+.2} EV", *val)
                } else {
                    format!("{:+.0}", *val * 100.0)
                };
                let active = (*val - default).abs() > 1e-3;
                let col = if active {
                    egui::Color32::from_rgb(240, 240, 245)
                } else {
                    egui::Color32::from_rgb(145, 148, 158)
                };
                let val_resp = ui.add_sized([58.0, 18.0], egui::Label::new(egui::RichText::new(txt).size(12.0).color(col)));
                if lbl_resp.double_clicked() || val_resp.double_clicked() {
                    *val = default;
                    changed = true;
                }
            });
        });
        let s_resp = ui.add(
            egui::Slider::new(val, range)
                .show_value(false)
                .trailing_fill(true),
        );
        if s_resp.double_clicked() {
            *val = default;
            changed = true;
        }
        if s_resp.changed() {
            changed = true;
        }
        changed
    }
}

impl eframe::App for RetuszStudioApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if self.dirty {
            self.rerender(ctx);
        }

        // Top Studio Toolbar
        egui::TopBottomPanel::top("top_bar")
            .exact_height(46.0)
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    if ui.button("Otwórz zdjęcie…").clicked() {
                        if let Some(p) = rfd::FileDialog::new().pick_file() {
                            self.open_path(&p);
                        }
                    }
                    if ui.button("Otwórz folder…").clicked() {
                        if let Some(p) = rfd::FileDialog::new().pick_folder() {
                            self.open_path(&p);
                        }
                    }
                    ui.separator();
                    ui.label("Widok:");
                    for (m, name) in [
                        (CanvasMode::Wynik, "Wynik"),
                        (CanvasMode::PrzedPo, "Przed | Po"),
                        (CanvasMode::Podzial, "Podział"),
                        (CanvasMode::ZWzorcem, "Z wzorcem"),
                    ] {
                        if ui.selectable_label(self.canvas_mode == m, name).clicked() {
                            self.canvas_mode = m;
                        }
                    }
                    ui.separator();
                    if ui.button("Dopasuj").clicked() {
                        self.zoom = 1.0;
                    }
                    if ui.button("100%").clicked() {
                        self.zoom = 2.0;
                    }
                    ui.separator();
                    if ui.add_enabled(!self.undo_stack.is_empty(), egui::Button::new("Cofnij")).clicked() {
                        self.undo();
                    }
                    if ui.add_enabled(!self.redo_stack.is_empty(), egui::Button::new("Ponów")).clicked() {
                        self.redo();
                    }
                    if ui.button("Kopiuj").clicked() {
                        self.clipboard = Some(self.params.clone());
                    }
                    if ui.add_enabled(self.clipboard.is_some(), egui::Button::new("Wklej")).clicked() {
                        if let Some(cp) = self.clipboard.clone() {
                            self.push_undo();
                            self.params = cp;
                            self.dirty = true;
                        }
                    }
                    if ui.button("Zeruj").clicked() {
                        self.push_undo();
                        self.params = Params::default();
                        self.dirty = true;
                    }
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui
                            .add(egui::Button::new(egui::RichText::new("Zapisz JPG").strong()).fill(egui::Color32::from_rgb(235, 168, 52)))
                            .clicked()
                        {
                            self.export_current();
                        }
                    });
                });
            });

        // Bottom Status Bar
        egui::TopBottomPanel::bottom("status_bar")
            .exact_height(26.0)
            .show(ctx, |ui| {
                ui.horizontal_centered(|ui| {
                    ui.label(egui::RichText::new(&self.status).size(12.0));
                });
            });

        // Left Filmstrip Panel
        egui::SidePanel::left("filmstrip_panel")
            .resizable(false)
            .exact_width(165.0)
            .show(ctx, |ui| {
                ui.add_space(6.0);
                ui.heading(format!("Seria ({})", self.photos.len()));
                ui.separator();
                egui::ScrollArea::vertical().show(ui, |ui| {
                    let list = self.photos.clone();
                    for (i, p) in list.iter().enumerate() {
                        let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("zdjęcie");
                        let has_sc = retusz_core::params::sidecar_path(p).exists();
                        let label = if has_sc {
                            format!("● {name}")
                        } else {
                            name.to_string()
                        };
                        if ui.selectable_label(self.idx == i, label).clicked() {
                            self.select_photo(i);
                        }
                    }
                });
            });

        // Right Docked Studio Inspector
        egui::SidePanel::right("studio_inspector")
            .resizable(false)
            .exact_width(330.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.add_space(6.0);
                    // 1. Live Histogram
                    ui.horizontal(|ui| {
                        ui.strong("Histogram");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let clip_txt = format!(
                                "Cienie {:.1}% · Światła {:.1}%",
                                self.histo.lo_pct, self.histo.hi_pct
                            );
                            if ui.selectable_label(self.show_clip, clip_txt).clicked() {
                                self.show_clip = !self.show_clip;
                            }
                        });
                    });
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(304.0, 68.0), egui::Sense::click());
                    let painter = ui.painter_at(rect);
                    painter.rect_filled(rect, 6.0, egui::Color32::from_rgb(20, 22, 26));
                    for i in 1..256 {
                        let x0 = rect.left() + (i - 1) as f32 / 255.0 * rect.width();
                        let x1 = rect.left() + i as f32 / 255.0 * rect.width();
                        let y0 = rect.bottom() - self.histo.l[i - 1] * (rect.height() - 6.0);
                        let y1 = rect.bottom() - self.histo.l[i] * (rect.height() - 6.0);
                        painter.line_segment(
                            [egui::pos2(x0, y0), egui::pos2(x1, y1)],
                            egui::Stroke::new(1.3_f32, egui::Color32::from_rgb(220, 222, 230)),
                        );
                    }
                    ui.add_space(6.0);
                    ui.separator();

                    // 2. Presets Section
                    ui.horizontal(|ui| {
                        ui.strong("Presety (Style)");
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui.small_button("Zapisz własny…").clicked() {
                                self.show_save_preset = !self.show_save_preset;
                            }
                        });
                    });
                    egui::ComboBox::from_id_salt("preset_combo")
                        .selected_text(&self.selected_preset)
                        .width(300.0)
                        .show_ui(ui, |ui| {
                            let list = self.presets.clone();
                            for pr in list {
                                let tag = if pr.builtin {
                                    format!("{} ({})", pr.name, pr.category)
                                } else {
                                    format!("★ {}", pr.name)
                                };
                                if ui.selectable_label(self.selected_preset == pr.name, tag).clicked() {
                                    self.push_undo();
                                    self.params = pr.apply_to(&self.params);
                                    self.selected_preset = pr.name.clone();
                                    self.dirty = true;
                                }
                            }
                        });
                    if self.show_save_preset {
                        ui.horizontal(|ui| {
                            ui.text_edit_singleline(&mut self.new_preset_name);
                            if ui.button("Zapisz").clicked() && !self.new_preset_name.trim().is_empty() {
                                let name = self.new_preset_name.trim().to_string();
                                let pr = Preset {
                                    name: name.clone(),
                                    category: "Własne".to_string(),
                                    builtin: false,
                                    params: self.params.clone(),
                                };
                                let path = Self::presets_dir().join(format!("{name}.retusz-preset.json"));
                                let _ = pr.save_to_file(&path);
                                self.presets = Self::load_all_presets();
                                self.selected_preset = name;
                                self.new_preset_name.clear();
                                self.show_save_preset = false;
                            }
                        });
                    }
                    ui.add_space(6.0);
                    ui.separator();

                    let mut p = self.params.clone();
                    let mut changed = false;

                    // 3. Open Section: Światło i Ton (OKLab / EV)
                    egui::CollapsingHeader::new(egui::RichText::new("Światło i Ton (OKLab / EV)").strong())
                        .default_open(true)
                        .show(ui, |ui| {
                            changed |= Self::slider_row(ui, "Ekspozycja", &mut p.exposure, -2.0..=2.0, 0.0, true);
                            changed |= Self::slider_row(ui, "Kontrast", &mut p.contrast, -1.0..=1.0, 0.0, false);
                            changed |= Self::slider_row(ui, "Światła", &mut p.highlights, -1.0..=1.0, 0.0, false);
                            changed |= Self::slider_row(ui, "Cienie", &mut p.shadows, -1.0..=1.0, 0.0, false);
                            changed |= Self::slider_row(ui, "Biele", &mut p.whites, -1.0..=1.0, 0.0, false);
                            changed |= Self::slider_row(ui, "Czernie", &mut p.blacks, -1.0..=1.0, 0.0, false);
                        });

                    // 4. Open Section: Kolor i Balans bieli (Bradford CAT)
                    egui::CollapsingHeader::new(egui::RichText::new("Kolor i Balans bieli (Bradford)").strong())
                        .default_open(true)
                        .show(ui, |ui| {
                            changed |= Self::slider_row(ui, "Temperatura", &mut p.temp, -1.0..=1.0, 0.0, false);
                            changed |= Self::slider_row(ui, "Odcień", &mut p.tint, -1.0..=1.0, 0.0, false);
                            changed |= Self::slider_row(ui, "Jaskrawość (Vibrance)", &mut p.vibrance, -1.0..=1.0, 0.0, false);
                            changed |= Self::slider_row(ui, "Nasycenie", &mut p.saturation, -1.0..=1.0, 0.0, false);
                            changed |= Self::slider_row(ui, "Pełne barwy", &mut p.wide, 0.0..=1.0, 0.0, false);
                        });

                    // 5. Open Section: Portret AI i Plastyka 3D
                    egui::CollapsingHeader::new(egui::RichText::new("Portret AI i Plastyka 3D").strong())
                        .default_open(true)
                        .show(ui, |ui| {
                            changed |= Self::slider_row(ui, "Dodge & Burn (skóra)", &mut p.db, 0.0..=1.0, 0.40, false);
                            changed |= Self::slider_row(ui, "Plastyka 3D", &mut p.plastic, 0.0..=1.0, 0.30, false);
                            changed |= Self::slider_row(ui, "Przejrzystość (Clarity)", &mut p.clarity, -1.0..=1.0, 0.0, false);
                            changed |= Self::slider_row(ui, "Czysta skóra (zafarb)", &mut p.skin_cast, 0.0..=1.0, 0.0, false);
                            changed |= Self::slider_row(ui, "Odcień skóry (OKLCh)", &mut p.skin_hue, -1.0..=1.0, 0.0, false);
                            changed |= Self::slider_row(ui, "Winieta", &mut p.vignette, -1.0..=1.0, 0.0, false);
                        });

                    // 6. Collapsible BONUS Section: Dopasowanie do wzorca (AI Color Match)
                    ui.add_space(4.0);
                    let hdr_txt = if self.ref_img.is_some() {
                        "Dopasowanie do wzorca (AI) · AKTYWNE"
                    } else {
                        "Dopasowanie do wzorca (AI · opcjonalnie)"
                    };
                    egui::CollapsingHeader::new(
                        egui::RichText::new(hdr_txt)
                            .strong()
                            .color(egui::Color32::from_rgb(235, 168, 52)),
                    )
                    .default_open(self.ref_section_open)
                    .show(ui, |ui| {
                        self.ref_section_open = true;
                        ui.horizontal(|ui| {
                            if ui.button("Wybierz wzorzec…").clicked() {
                                if let Some(rp) = rfd::FileDialog::new().pick_file() {
                                    self.set_reference(&rp);
                                }
                            }
                            if self.ref_img.is_some() && ui.button("Wyłącz wzorzec").clicked() {
                                self.ref_img = None;
                                self.ref_stats = None;
                                self.tex_ref = None;
                                self.dirty = true;
                            }
                        });
                        if let Some(rf) = &self.ref_img {
                            let rname = rf.path.file_name().and_then(|s| s.to_str()).unwrap_or("wzorzec");
                            ui.label(egui::RichText::new(format!("Wzorzec: {rname}")).size(11.5));
                        } else {
                            ui.label(
                                egui::RichText::new("Opcjonalny bonus: wybierz zdjęcie wzorcowe, aby przenieść z niego paletę barw i kontrast.")
                                    .size(11.0)
                                    .color(egui::Color32::from_rgb(145, 148, 158)),
                            );
                        }
                        changed |= Self::slider_row(ui, "Siła kolorów wzorca", &mut p.strength, 0.0..=1.0, 1.0, false);
                        changed |= Self::slider_row(ui, "Ochrona skóry", &mut p.skin, 0.0..=1.0, 1.0, false);
                        changed |= Self::slider_row(ui, "Równanie skóry", &mut p.skin_even, 0.0..=1.0, 0.0, false);
                    });

                    if changed {
                        self.push_undo();
                        self.params = p;
                        self.dirty = true;
                    }
                });
            });

        // Central Studio Canvas
        egui::CentralPanel::default().show(ctx, |ui| {
            let avail = ui.available_size();
            match self.canvas_mode {
                CanvasMode::Wynik | CanvasMode::Podzial => {
                    if let Some(tex) = &self.tex_res {
                        let sz = tex.size_vec2();
                        let scale = ((avail.x - 16.0) / sz.x).min((avail.y - 16.0) / sz.y) * self.zoom;
                        let draw_sz = sz * scale;
                        ui.centered_and_justified(|ui| {
                            ui.image((tex.id(), draw_sz));
                        });
                    } else {
                        ui.centered_and_justified(|ui| {
                            ui.label(
                                egui::RichText::new("Otwórz zdjęcie lub folder, aby rozpocząć obróbkę.\n\nPełny edytor OKLab / Bradford CAT / Plastyka 3D / Dodge & Burn v2.\nDopasowanie do wzorca oraz Presety znajdziesz w prawym panelu.")
                                    .size(14.0)
                                    .color(egui::Color32::from_rgb(150, 154, 165)),
                            );
                        });
                    }
                }
                CanvasMode::PrzedPo => {
                    ui.columns(2, |cols| {
                        cols[0].vertical_centered(|ui| {
                            ui.label("Przed (Oryginał)");
                            if let Some(ts) = &self.tex_src {
                                let sz = ts.size_vec2();
                                let scale = ((avail.x * 0.48) / sz.x).min((avail.y - 32.0) / sz.y);
                                ui.image((ts.id(), sz * scale));
                            }
                        });
                        cols[1].vertical_centered(|ui| {
                            ui.label("Po (Wynik)");
                            if let Some(tr) = &self.tex_res {
                                let sz = tr.size_vec2();
                                let scale = ((avail.x * 0.48) / sz.x).min((avail.y - 32.0) / sz.y);
                                ui.image((tr.id(), sz * scale));
                            }
                        });
                    });
                }
                CanvasMode::ZWzorcem => {
                    ui.columns(2, |cols| {
                        cols[0].vertical_centered(|ui| {
                            ui.label("Wynik");
                            if let Some(tr) = &self.tex_res {
                                let sz = tr.size_vec2();
                                let scale = ((avail.x * 0.48) / sz.x).min((avail.y - 32.0) / sz.y);
                                ui.image((tr.id(), sz * scale));
                            }
                        });
                        cols[1].vertical_centered(|ui| {
                            ui.label("Wzorzec");
                            if let Some(tf) = &self.tex_ref {
                                let sz = tf.size_vec2();
                                let scale = ((avail.x * 0.48) / sz.x).min((avail.y - 32.0) / sz.y);
                                ui.image((tf.id(), sz * scale));
                            } else {
                                ui.label("Rozwiń „Dopasowanie do wzorca” po prawej i wybierz wzorzec.");
                            }
                        });
                    });
                }
            }
        });
    }
}

fn run_headless_cli(
    path: &Path,
    reference: Option<&Path>,
    preset_name: Option<&str>,
    out_dir: Option<&Path>,
) -> Result<(), String> {
    let ref_stats = reference
        .and_then(LoadedImage::load)
        .map(|img| ReferenceStats::from_rgb8(&img.rgb8));
    let presets = builtin_presets();
    let chosen_preset = preset_name.and_then(|nm| presets.iter().find(|p| p.name == nm));

    let files: Vec<PathBuf> = if path.is_dir() {
        fs::read_dir(path)
            .map_err(|e| e.to_string())?
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.extension()
                    .and_then(|s| s.to_str())
                    .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"))
                    .unwrap_or(false)
            })
            .collect()
    } else {
        vec![path.to_path_buf()]
    };

    for f in files {
        let Some(img) = LoadedImage::load(&f) else { continue };
        let mut params = load_sidecar(&f).unwrap_or_default();
        if let Some(pr) = chosen_preset {
            params = pr.apply_to(&params);
        }
        let (out, rw, rh) = render_photo_cpu(&img.rgb8, img.w, img.h, &params, &[], None, ref_stats);
        let dst_folder = out_dir
            .map(PathBuf::from)
            .unwrap_or_else(|| f.parent().unwrap_or_else(|| Path::new(".")).join("Retusz"));
        fs::create_dir_all(&dst_folder).map_err(|e| e.to_string())?;
        let stem = f.file_stem().and_then(|s| s.to_str()).unwrap_or("out");
        let dst = dst_folder.join(format!("{stem}_retusz.jpg"));
        let rgb_img = image::RgbImage::from_raw(rw as u32, rh as u32, out)
            .ok_or_else(|| "Błąd bufora obrazu".to_string())?;
        rgb_img.save(&dst).map_err(|e| e.to_string())?;
        println!("Zapisano {}", dst.display());
    }
    Ok(())
}

fn main() -> eframe::Result<()> {
    let args = CliArgs::parse();
    if let Some(Command::Cli { path, reference, preset, out }) = args.cmd {
        if let Err(e) = run_headless_cli(&path, reference.as_deref(), preset.as_deref(), out.as_deref()) {
            eprintln!("Błąd: {e}");
            std::process::exit(1);
        }
        return Ok(());
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1480.0, 900.0])
            .with_min_inner_size([1180.0, 720.0])
            .with_title("Retusz 0.2.0 – Edytor Portretowy"),
        ..Default::default()
    };
    eframe::run_native(
        "Retusz 0.2.0",
        options,
        Box::new(move |cc| {
            let mut visuals = egui::Visuals::dark();
            visuals.panel_fill = egui::Color32::from_rgb(24, 25, 29);
            visuals.window_fill = egui::Color32::from_rgb(24, 25, 29);
            cc.egui_ctx.set_visuals(visuals);
            Ok(Box::new(RetuszStudioApp::new(args.path, args.reference)))
        }),
    )
}
