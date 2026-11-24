"""Retusz main window (mockup A) + Porównaj view (mockup B)."""
from __future__ import annotations
import queue, threading, traceback, sys
from pathlib import Path
from tkinter import filedialog, messagebox
import numpy as np, cv2
import customtkinter as ctk
from PIL import Image
from . import theme as T
from .imageview import ImageView
from .compare import CompareView
from .histview import HistView
from .dialogs import SettingsDialog, ModelDialog, ProgressDialog
from .. import imageio, models, batch, export, colorman as CM, __version__
from ..reference import Reference
from ..settings import Settings
from ..sidecar import Series
from ..engine.pipeline import Engine, Params
from ..engine import render as R, color as C, device as DEV, modflows as MF, relight as RL
from ..engine import histo as HI, wb as WB, geometry as G

ENGINE_LABEL = {"ai": "AI (B6)", "ai_light": "AI lekki (B0)", "classic": "bez AI (IDT)"}
SLIDERS = [  # key, label, from, to, display  (4 equal columns, fixed-width value boxes: nothing re-flows while dragging)
    ("strength", "Siła kolorów", 0, 100, "pct"), ("skin", "Skóra", 0, 100, "pct"), ("skin_even", "Równanie skóry", 0, 100, "pct"),
    ("vignette", "Winieta", 0, 100, "pct"),
    ("shadows", "Cienie", -100, 100, "num"), ("highlights", "Światła", -100, 100, "num"), ("saturation", "Nasycenie", -100, 100, "num"),
    ("vibrance", "Jaskrawość", -100, 100, "num"),
    ("wide", "Pełne barwy", 0, 100, "pct"), ("db", "Dodge&Burn", 0, 100, "pct"), ("plastic", "Plastyka 3D", 0, 100, "pct"),
    ("temp", "Temperatura", -100, 100, "num"), ("tint", "Odcień", -100, 100, "num"),
]
SLIDER_CELLS = [(i//4, i % 4) for i in range(11)] + [(3, 0), (3, 1)]   # (2, 3) = light direction, (3, 2-3) = white balance
TOOLS = ["Pędzel D&B", "Kadr"]
TOOL_W, TOOL_H = 268, 184
PHOTO_TYPES = [("Zdjęcia i RAW", "*.jpg *.jpeg *.png *.tif *.tiff *.webp *.nef *.NEF *.nrw *.NRW")]
BRUSH_MIN, BRUSH_MAX = 5, 600   # brush radius in original-image pixels
BRUSH_MODES = ["Wył.", "Rozjaśnij", "Przyciemnij"]

class RetuszApp(ctk.CTk):
    def __init__(self, path=None, reference=None):
        T.setup(); super().__init__(fg_color=T.BG)
        self.title(f"Retusz {__version__}"); self.geometry("1360x860"); self.minsize(1100, 700)
        self.settings = Settings.load(); DEV.MODE = self.settings.device
        if sys.platform == "win32":
            try:
                from ..paths import resource_dir; self.iconbitmap(str(resource_dir()/"retusz.ico"))
            except Exception: pass
        self.engine: Engine | None = None; self.engine_name = ""; self.device = ""
        self.series: Series | None = None; self.photos: list[Path] = []; self.idx = -1
        self.ref_analysis = None; self.ref_prev = None; self.reference: Reference | None = None
        self.src_icc = None; self.full_cache = None; self._detail_after = None; self._space = False; self._pan = None
        self.src_prev = None; self.src_size = None; self.fit = None; self.params = Params()
        self.result = None; self.mask_prev = None; self.thumbs = {}; self.thumb_btns = []
        self._gen = {}; self._ui = queue.Queue(); self._jobs = queue.Queue(); self._save_after = None
        self._stroke = None
        self.frame = None; self.detail_raw = None; self.show_clip = False; self.picking = False
        self.tool = "brush"; self.hline_on = False; self._hline = None; self._crop_drag = None; self._crop_live = None
        self._geo_after = None
        threading.Thread(target=self._worker, daemon=True).start()
        self._build(); self._poll()
        self.bind("<Left>", lambda e: self.step(-1)); self.bind("<Right>", lambda e: self.step(1))
        self.bind("<Control-s>", lambda e: self.save_current()); self.bind("<Control-z>", lambda e: self.undo_stroke())
        self.bind("<KeyPress-space>", lambda e: self._space_key(True)); self.bind("<KeyRelease-space>", lambda e: self._space_key(False))
        self.bind("<bracketleft>", lambda e: self._brush_step(-1)); self.bind("<bracketright>", lambda e: self._brush_step(1))
        self.bind("<Control-0>", lambda e: self.zoom_fit()); self.bind("<Control-1>", lambda e: self.zoom_100())
        for k, fn in (("w", self.toggle_pick), ("c", self.toggle_crop_tool), ("j", self.toggle_clip)):
            self.bind(f"<KeyPress-{k}>", lambda e, fn=fn: self._key(fn)); self.bind(f"<KeyPress-{k.upper()}>", lambda e, fn=fn: self._key(fn))
        self.bind("<Return>", lambda e: self._key(self._crop_done)); self.bind("<Escape>", lambda e: self._key(self._escape))
        self.submit(lambda: (R.warmup(), None)[1], None)
        self.setup_engine()
        if path: self.after(100, lambda: self.open_path(Path(path), Path(reference) if reference else None))
        self.after(700, self._first_run)

    # ================= layout =================
    def _build(self):
        top = ctk.CTkFrame(self, fg_color=T.BG); top.pack(fill="x", padx=14, pady=(12, 8))
        T.button(top, "Otwórz zdjęcie", self.open_photo, width=130).pack(side="left")
        T.button(top, "Otwórz folder", self.open_folder, width=130).pack(side="left", padx=8)
        T.button(top, "Wzorzec", self.choose_reference).pack(side="left")
        T.button(top, "Porównaj", self.show_compare).pack(side="left", padx=8)
        T.button(top, "Ustawienia", self.open_settings).pack(side="left")
        zb = ctk.CTkFrame(top, fg_color=T.BG); zb.pack(side="left", padx=(22, 0))
        T.button(zb, "Dopasuj", self.zoom_fit, width=78).pack(side="left")
        T.button(zb, "100%", self.zoom_100, width=58).pack(side="left", padx=6)
        zbox = ctk.CTkFrame(zb, fg_color=T.BG, width=64, height=30); zbox.pack(side="left"); zbox.pack_propagate(False)
        self.zoom_lbl = T.label(zbox, "", 12, T.MUTED, anchor="w"); self.zoom_lbl.pack(fill="both", expand=True)
        T.button(top, "Eksportuj serię", self.export_series, accent=True, width=140).pack(side="right")
        T.button(top, "Zapisz", self.save_current, accent=True).pack(side="right", padx=8)

        self.main = ctk.CTkFrame(self, fg_color=T.BG); self.main.pack(fill="both", expand=True)
        self.compare = CompareView(self, self.hide_compare, self.save_current, self._compare_full)

        side = ctk.CTkFrame(self.main, fg_color=T.BG, width=170); side.pack(side="left", fill="y", padx=(14, 0))
        self.series_lbl = T.label(side, "Seria", 13, weight="bold"); self.series_lbl.pack(anchor="w", pady=(0, 6))
        self.strip = ctk.CTkScrollableFrame(side, fg_color=T.PANEL, width=150, corner_radius=8)
        self.strip.pack(fill="y", expand=True)

        body = ctk.CTkFrame(self.main, fg_color=T.BG); body.pack(side="left", fill="both", expand=True, padx=14)
        imgs = ctk.CTkFrame(body, fg_color=T.BG); imgs.pack(fill="both", expand=True)
        for c in range(3): imgs.grid_columnconfigure(c, weight=1, uniform="v")
        imgs.grid_rowconfigure(1, weight=1)
        for c, t in enumerate(["Źródło", "Wynik", "Wzorzec"]):
            T.label(imgs, t, 14).grid(row=0, column=c, sticky="w", padx=4, pady=(0, 4))
        self.view_src = ImageView(imgs, "Otwórz zdjęcie lub folder")
        self.view_res = ImageView(imgs, "Tu pojawi się wynik")
        self.view_ref = ImageView(imgs, "Wybierz wzorzec,\nnp. zdjęcie obrobione w PS")
        for c, v in enumerate([self.view_src, self.view_res, self.view_ref]): v.grid(row=1, column=c, sticky="nsew", padx=4)
        self.view_res.bind("<ButtonPress-1>", self._brush_press); self.view_res.bind("<B1-Motion>", self._brush_move)
        self.view_res.bind("<ButtonRelease-1>", self._brush_release); self.view_res.bind("<Motion>", self._brush_cursor)
        self.view_res.bind("<Leave>", lambda e: self.view_res.delete("cursor"))
        self.view_res.on_redraw = self._draw_crop_overlay
        for v in (self.view_src, self.view_res):
            v.on_view_change = self._view_changed
            v.bind("<MouseWheel>", lambda e, v=v: self._wheel(v, e, 1 if e.delta > 0 else -1))
            v.bind("<Button-4>", lambda e, v=v: self._wheel(v, e, 1)); v.bind("<Button-5>", lambda e, v=v: self._wheel(v, e, -1))
            v.bind("<ButtonPress-2>", lambda e, v=v: self._pan_start(v, e)); v.bind("<B2-Motion>", lambda e, v=v: self._pan_move(v, e))
            v.bind("<ButtonRelease-2>", lambda e: self._pan_end())
        self.view_src.bind("<ButtonPress-1>", lambda e: self._pan_start(self.view_src, e))
        self.view_src.bind("<B1-Motion>", lambda e: self._pan_move(self.view_src, e)); self.view_src.bind("<ButtonRelease-1>", lambda e: self._pan_end())

        ctl = ctk.CTkFrame(body, fg_color=T.CARD, corner_radius=10, border_width=1, border_color=T.BORDER)
        ctl.pack(fill="x", pady=(12, 0))
        sl = ctk.CTkFrame(ctl, fg_color=T.CARD)
        for c in range(4): sl.grid_columnconfigure(c, weight=1, uniform="sl")
        self.sliders = {}
        def value_box(parent, width):
            box = ctk.CTkFrame(parent, fg_color=T.CARD, width=width, height=22); box.pack_propagate(False)
            lab = T.label(box, "", 13, T.MUTED, anchor="e"); lab.pack(fill="both", expand=True); return box, lab
        for i, (key, text, lo, hi, kind) in enumerate(SLIDERS):
            r_, c_ = SLIDER_CELLS[i]
            cell = ctk.CTkFrame(sl, fg_color=T.CARD); cell.grid(row=r_, column=c_, sticky="ew", padx=6, pady=3)
            cell.grid_columnconfigure(0, weight=1)
            T.label(cell, text, 13, anchor="w").grid(row=0, column=0, sticky="w")
            box, val = value_box(cell, 46); box.grid(row=1, column=1, padx=(6, 0))
            s = T.slider(cell, lo, hi, number_of_steps=hi-lo, width=60, command=lambda v, k=key: self._slider(k, v))
            s.grid(row=1, column=0, sticky="ew")
            self.sliders[key] = (s, val, kind)
        cell = ctk.CTkFrame(sl, fg_color=T.CARD); cell.grid(row=2, column=3, sticky="ew", padx=6, pady=3)
        cell.grid_columnconfigure(0, weight=1)
        T.label(cell, "Światło 3D", 13, anchor="w").grid(row=0, column=0, sticky="w")
        self.light_auto = ctk.CTkSwitch(cell, text="Auto", progress_color=T.ACCENT, text_color=T.TEXT, width=60,
                                        switch_width=30, switch_height=14, font=T.font(11), command=self._light_auto)
        self.light_auto.grid(row=0, column=1, sticky="e")
        box, self.light_val = value_box(cell, 62); box.grid(row=1, column=1, padx=(6, 0))
        self.light = T.slider(cell, 0, 355, number_of_steps=71, width=60, command=self._light); self.light.grid(row=1, column=0, sticky="ew")
        wbc = ctk.CTkFrame(sl, fg_color=T.CARD); wbc.grid(row=3, column=2, columnspan=2, sticky="ew", padx=6, pady=3)
        T.label(wbc, "Balans bieli", 13, anchor="w").grid(row=0, column=0, columnspan=3, sticky="w")
        self.pick_btn = T.button(wbc, "Pipeta (W)", self.toggle_pick, width=104); self.pick_btn.configure(height=26)
        self.pick_btn.grid(row=1, column=0, sticky="w")
        b = T.button(wbc, "Balans → seria", self.sync_wb, width=118); b.configure(height=26); b.grid(row=1, column=1, sticky="w", padx=(6, 0))
        b = T.button(wbc, "Zeruj", self.reset_wb, width=56); b.configure(height=26); b.grid(row=1, column=2, sticky="w", padx=(6, 0))

        tools = ctk.CTkFrame(ctl, fg_color=T.CARD)
        self.tool_tabs = T.segmented(tools, TOOLS, command=self._tool_tab); self.tool_tabs.set(TOOLS[0]); self.tool_tabs.pack(fill="x")
        area = ctk.CTkFrame(tools, fg_color=T.CARD, width=TOOL_W, height=TOOL_H); area.pack(pady=(8, 0)); area.pack_propagate(False)
        br = ctk.CTkFrame(area, fg_color=T.CARD); cr = ctk.CTkFrame(area, fg_color=T.CARD)
        for f in (br, cr): f.place(x=0, y=0, relwidth=1, relheight=1)
        self.tool_frames = {TOOLS[0]: br, TOOLS[1]: cr}
        self.brush_mode = T.segmented(br, BRUSH_MODES, command=lambda v: self.view_res.delete("cursor")); self.brush_mode.set(BRUSH_MODES[0])
        self.brush_mode.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 4))
        self.brush_vals = {}
        def brow(r, text, lo, hi, val, key):
            T.label(br, text, 12, T.MUTED).grid(row=r, column=0, sticky="w")
            box = ctk.CTkFrame(br, fg_color=T.CARD, width=56, height=20); box.grid(row=r, column=2, padx=(6, 0)); box.pack_propagate(False)
            lab = T.label(box, "", 12, T.MUTED, anchor="e"); lab.pack(fill="both", expand=True); self.brush_vals[key] = lab
            sl_ = T.slider(br, lo, hi, width=118, command=lambda v: self._brush_label(key)); sl_.set(val); sl_.grid(row=r, column=1)
            return sl_
        self.brush_size = brow(1, "Rozmiar", 0, 1, 0.45, "size")
        self.brush_soft = brow(2, "Miękkość", 0, 100, 50, "soft")
        self.brush_str = brow(3, "Przepływ", 5, 100, 40, "flow")
        for k in self.brush_vals: self._brush_label(k)
        bb = ctk.CTkFrame(br, fg_color=T.CARD); bb.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        T.button(bb, "Cofnij", self.undo_stroke, width=80).pack(side="left")
        T.button(bb, "Wyczyść", self.clear_strokes, width=80).pack(side="left", padx=6)
        T.label(br, "[ ] lub Alt+kółko: rozmiar · Ctrl+kółko: zoom", 11, T.MUTED).grid(row=5, column=0, columnspan=3, sticky="w", pady=(4, 0))
        # crop & straighten
        cr.grid_columnconfigure(1, weight=1)
        T.label(cr, "Proporcje", 12, T.MUTED).grid(row=0, column=0, sticky="w")
        self.aspect_menu = ctk.CTkOptionMenu(cr, values=list(G.ASPECTS.values()), command=self._aspect, width=100, height=28,
                                             fg_color=T.BTN, button_color=T.BTN, button_hover_color=T.BTN_HOVER, text_color=T.TEXT,
                                             font=T.font(12), dropdown_font=T.font(12), corner_radius=8)
        self.aspect_menu.grid(row=0, column=1, sticky="w", padx=(8, 0))
        b = T.button(cr, "Pion/Poziom", self._flip_aspect, width=88); b.configure(height=28); b.grid(row=0, column=2, sticky="e")
        T.label(cr, "Prostowanie", 12, T.MUTED).grid(row=1, column=0, sticky="w", pady=(10, 0))
        self.angle = T.slider(cr, -45, 45, number_of_steps=900, width=100, command=self._angle)
        self.angle.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(10, 0))
        box = ctk.CTkFrame(cr, fg_color=T.CARD, width=56, height=20); box.grid(row=1, column=2, sticky="e", pady=(10, 0)); box.pack_propagate(False)
        self.angle_val = T.label(box, "", 12, T.MUTED, anchor="e"); self.angle_val.pack(fill="both", expand=True)
        cb = ctk.CTkFrame(cr, fg_color=T.CARD); cb.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        self.hline_btn = T.button(cb, "Linia horyzontu", self.toggle_hline, width=124); self.hline_btn.pack(side="left")
        T.button(cb, "Resetuj kadr", self.reset_crop, width=104).pack(side="left", padx=6)
        T.label(cr, "Przeciągnij na wyniku: nowy kadr, narożniki,\nprzesuwanie · Enter: gotowe · Esc: wyjście", 11, T.MUTED,
                justify="left").grid(row=3, column=0, columnspan=3, sticky="w", pady=(8, 0))
        br.lift()

        ser = ctk.CTkFrame(ctl, fg_color=T.CARD)
        self.hist = HistView(ser, on_toggle=self.toggle_clip, width=206, height=74); self.hist.pack(anchor="w")
        hb = ctk.CTkFrame(ser, fg_color=T.CARD, width=208, height=20); hb.pack(anchor="w", pady=(2, 0)); hb.pack_propagate(False)
        self.clip_lbl = T.label(hb, "Cienie – · Światła –", 11, T.MUTED, anchor="w"); self.clip_lbl.pack(fill="both", expand=True)
        self.clip_sw = ctk.CTkSwitch(ser, text="Pokaż przepalenia (J)", progress_color=T.ACCENT, text_color=T.TEXT, font=T.font(12),
                                     switch_width=30, switch_height=14, command=lambda: self.toggle_clip(self.clip_sw.get() == 1))
        self.clip_sw.pack(anchor="w", pady=(2, 8))
        ob = ctk.CTkFrame(ser, fg_color=T.CARD); ob.pack(anchor="w", fill="x")
        T.label(ob, "To zdjęcie:", 12).pack(side="left")
        self.override_lbl = T.label(ob, "ustawienia serii", 12, T.MUTED); self.override_lbl.pack(side="left", padx=(6, 0))
        b = T.button(ser, "Przywróć ustawienia serii", self.reset_to_series, width=208); b.configure(height=28); b.pack(pady=(4, 2))
        b = T.button(ser, "Ustaw jako domyślne serii", self.make_series_default, width=208); b.configure(height=28); b.pack(pady=2)
        ser.pack(side="right", fill="y", padx=(0, 14), pady=12); tools.pack(side="right", fill="y", padx=(0, 14), pady=12)
        sl.pack(side="left", fill="x", expand=True, padx=(10, 12), pady=12)

        self.status = T.label(self, "", 12, T.MUTED); self.status.pack(fill="x", pady=8)
        self._load_params_into_ui(self.params); self._update_status()

    # ================= threading =================
    def _worker(self):
        while True:
            fn, done, kind, gen = self._jobs.get()
            if kind and self._gen.get(kind) != gen: continue  # superseded
            try:
                res = fn(); self._ui.put((done, res))
            except Exception as e:
                tb = traceback.format_exc(); self._ui.put((self._error, (e, tb)))

    def submit(self, fn, done, kind=None):
        g = None
        if kind: g = self._gen[kind] = self._gen.get(kind, 0)+1
        self._jobs.put((fn, done, kind, g))

    def _poll(self):
        try:
            while True:
                done, res = self._ui.get_nowait()
                if done: done(res)
        except queue.Empty: pass
        self.after(30, self._poll)

    def _error(self, info):
        e, tb = info; print(tb, file=sys.stderr); self._set_status(f"Błąd: {e}")

    # ================= engine / model =================
    def setup_engine(self):
        path, eff = models.resolve(self.settings.method)
        def work():
            e = Engine(path, eff); return e, e.device()
        def done(r):
            self.engine, self.device = r; self.engine_name = ENGINE_LABEL[self.engine.method]
            self.fit = None; self._update_status(); self.refit()
        self.submit(work, done)

    def _first_run(self):
        if self.settings.method == "ai" and not models.have_ai() and not self.settings.asked_download:
            self.settings.asked_download = True; self.settings.save()
            ModelDialog(self, lambda ok: ok and self.setup_engine())

    def open_settings(self):
        old = (self.settings.method, self.settings.preview_side, self.settings.device, models.have_ai())
        def saved():
            new = (self.settings.method, self.settings.preview_side, self.settings.device, models.have_ai())
            if new[2] != old[2]:
                DEV.MODE = self.settings.device; DEV.reset(); MF.reset_sessions(); RL.reset_session()
            if new != old:
                self.setup_engine()
                if self.idx >= 0: self.select(self.idx)
            self.redraw_result()
        SettingsDialog(self, self.settings, saved)

    # ================= opening =================
    def open_photo(self):
        p = filedialog.askopenfilename(title="Otwórz zdjęcie", initialdir=self.settings.last_folder or None,
                                       filetypes=PHOTO_TYPES)
        if p: self.open_path(Path(p))

    def open_folder(self):
        p = filedialog.askdirectory(title="Otwórz folder", initialdir=self.settings.last_folder or None)
        if p: self.open_path(Path(p))

    def open_path(self, path: Path, reference: Path | None = None):
        folder = path if path.is_dir() else path.parent
        self.series = Series(folder); self.settings.last_folder = str(folder); self.settings.save()
        if reference: self.series.reference = reference; self.series.save()
        ref = self.series.reference.resolve() if self.series.reference else None
        if path.is_dir():
            out_dir = (folder / self.settings.output_subfolder).resolve()
            self.photos = [p for p in imageio.list_images(folder) if p.resolve() != ref and p.parent.resolve() != out_dir]
        else:
            self.photos = [path]
        self.series_lbl.configure(text=f"Seria · {len(self.photos)}")
        self._build_strip()
        if self.series.reference and self.series.reference.exists(): self.load_reference(self.series.reference)
        if self.photos: self.select(0)
        else: self._set_status("W folderze nie ma zdjęć JPG/PNG/TIFF/NEF.")
        tw = imageio.raw_twins(folder) if path.is_dir() else 0
        if tw: self.after(1500, lambda: self._set_status(f"Pominięto {tw} JPG z parą NEF o tej samej nazwie – edytujesz pliki NEF."))

    def choose_reference(self):
        p = filedialog.askopenfilename(title="Wybierz wzorzec", initialdir=self.settings.last_folder or None,
                                       filetypes=PHOTO_TYPES)
        if not p: return
        p = Path(p)
        if self.series is None: self.series = Series(p.parent)
        self.series.reference = p; self.series.save()
        if p in self.photos:
            i = self.photos.index(p); self.photos.remove(p); self._build_strip()
            if self.photos: self.select(min(i, len(self.photos)-1))
        self.load_reference(p)

    def load_reference(self, p: Path):
        self._set_status("Wczytuję wzorzec…")
        side = self.settings.preview_side
        def work():
            return Reference(p), imageio.load_preview(p, side, with_icc=True)
        def done(r):
            self.reference, (self.ref_prev, ref_icc) = r; self.ref_analysis = None
            self.view_ref.set_image(C.crop_border(self.ref_prev, margin=2), icc=ref_icc, keep_view=False); self.fit = None; self.refit()
        self.submit(work, done)

    # ================= series strip =================
    def _build_strip(self):
        for b in self.thumb_btns: b.destroy()
        self.thumb_btns = []
        for i, p in enumerate(self.photos):
            b = ctk.CTkButton(self.strip, text=self._thumb_text(p), width=130, height=110, compound="top", corner_radius=6,
                              fg_color=T.PANEL, hover_color=T.BTN_HOVER, text_color=T.TEXT, border_width=2, border_color=T.PANEL,
                              font=T.font(11), command=lambda i=i: self.select(i))
            b.pack(pady=4, padx=4); self.thumb_btns.append(b)
        photos = list(self.photos)
        def load_all():
            for i, p in enumerate(photos):
                if p in self.thumbs: continue
                try: im = Image.fromarray(imageio.load_preview(p, 240))
                except Exception: continue
                self._ui.put((lambda r: self._set_thumb(*r), (i, p, im)))
        threading.Thread(target=load_all, daemon=True).start()

    def _set_thumb(self, i, p, im):
        im.thumbnail((124, 84)); self.thumbs[p] = ctk.CTkImage(im, size=im.size)
        if i < len(self.thumb_btns) and self.photos[i] == p: self.thumb_btns[i].configure(image=self.thumbs[p])

    def _thumb_text(self, p):
        name = p.name if len(p.name) <= 18 else p.name[:15] + "…"
        return ("● " if self.series and self.series.has_override(p) else "") + name

    def _refresh_strip_marks(self):
        for i, b in enumerate(self.thumb_btns):
            b.configure(text=self._thumb_text(self.photos[i]), border_color=T.ACCENT if i == self.idx else T.PANEL)

    # ================= selection / processing =================
    def step(self, d):
        if self.photos and not isinstance(self.focus_get(), ctk.CTkEntry):
            self.select(max(0, min(len(self.photos)-1, self.idx+d)))

    def select(self, i):
        self.idx = i; p = self.photos[i]; self.params = self.series.params_for(p)
        self._load_params_into_ui(self.params); self._refresh_strip_marks(); self.fit = None
        self.result = None; self.frame = None; self.detail_raw = None; self.full_cache = None
        self.view_res.set_image(None, keep_view=False); self._set_status(f"Wczytuję {p.name}…")
        side = self.settings.preview_side
        def work():
            return imageio.load_preview(p, side, with_icc=True), imageio.image_size(p)
        def done(r):
            if self.idx != i: return
            (self.src_prev, self.src_icc), self.src_size = r
            self.view_src.set_image(self.src_prev, icc=self.src_icc, full_wh=self.src_size, keep_view=False); self._sync_zoom_label(); self.refit()
        self.submit(work, done, kind="load")

    def refit(self):
        if self.src_prev is None or self.reference is None or self.engine is None:
            self._update_status(); return
        i, src, ref, eng, icc = self.idx, self.src_prev, self.reference, self.engine, self.src_icc
        self._set_status("Dopasowuję kolory…")
        def work():
            ra = ref.analysis(eng, icc)        # reference converted into this photo's colour space (cached)
            return ra, eng.fit(src, ra)
        def done(r):
            if self.idx != i: return
            self.ref_analysis, self.fit = r; self._show_light(); self.request_render()
        self.submit(work, done, kind="fit")

    def request_render(self):
        if self.fit is None or self.src_prev is None: return
        fit, src, p, eng, i = self.fit, self.src_prev, Params.from_dict(self.params.to_dict()), self.engine, self.idx
        def done(r):
            if self.idx != i: return
            self.result, self.mask_prev = r; self.redraw_result(); self._update_status()
            if self.compare.winfo_ismapped(): self.compare.set_images(self._disp(self.src_prev), self._disp(self.result))
            self._schedule_detail()
        self.submit(lambda: eng.render(src, fit, p, return_mask=True), done, kind="render")

    def redraw_result(self):
        """Result view = the straightened frame; outside the crop is dimmed (or shaded by the crop tool)."""
        if self.result is None: return
        a = self.params.angle
        self.frame = G.apply(self.result, a, None, interp=cv2.INTER_CUBIC) if a else self.result
        img = self.frame
        if self.settings.show_mask and self.mask_prev is not None:
            m = cv2.resize(self.mask_prev, (self.result.shape[1], self.result.shape[0]))
            if a: m = G.apply(m, a, None, interp=cv2.INTER_LINEAR)
            m = m[..., None]; img = (img*(1-0.45*m) + np.array([245, 163, 59])*0.45*m).astype(np.uint8)
        self.view_res.set_image(self._decorate(img), icc=self.src_icc, full_wh=self.src_size)
        if self.detail_raw is not None: self.view_res.detail = self._decorate(self.detail_raw); self.view_res.redraw()
        if self.view_res.zoom != self.view_src.zoom or self.view_res.center != self.view_src.center:
            self.view_res.set_view(self.view_src.zoom, self.view_src.center)
        self._sync_zoom_label(); self._update_hist()

    def _decorate(self, img):
        """Clipping overlay + crop dimming for display (the export never sees this)."""
        if self.show_clip: img = HI.overlay(img)
        c = self.params.crop
        if c is not None and self.tool != "crop":
            h, w = img.shape[:2]; x0, y0, x1, y1 = G.crop_px(c, w, h)
            out = (img*0.28).astype(np.uint8); out[y0:y1, x0:x1] = img[y0:y1, x0:x1]; img = out
        return img

    def _update_hist(self):
        if self.frame is None: self.hist.draw(None, active=self.show_clip); return
        h, w = self.frame.shape[:2]; x0, y0, x1, y1 = G.crop_px(self.params.crop, w, h)
        reg = self.frame[y0:y1, x0:x1]; lo, hi = HI.clipping(reg)[:2]
        self.hist.draw(HI.histogram(reg), lo, hi, self.show_clip)
        self.clip_lbl.configure(text=f"Cienie {lo*100:.1f}% · Światła {hi*100:.1f}%".replace(".", ","),
                                text_color=T.TEXT if max(lo, hi) > 0.005 else T.MUTED)

    def toggle_clip(self, on=None):
        self.show_clip = (not self.show_clip) if on is None else bool(on)
        (self.clip_sw.select() if self.show_clip else self.clip_sw.deselect())
        self.redraw_result()

    # ================= white balance =================
    def toggle_pick(self, on=None):
        self.picking = (not self.picking) if on is None else bool(on)
        self.pick_btn.configure(border_color=T.ACCENT if self.picking else T.BORDER, text_color=T.ACCENT if self.picking else T.TEXT)
        self.view_res.configure(cursor="crosshair" if self.picking else "")
        if self.picking:
            self.view_res.delete("cursor"); self._set_status("Pipeta: kliknij na wyniku punkt, który ma być neutralnie szary lub biały (Esc: anuluj).")

    def _pick(self, e):
        if self.frame is None: return
        u, v = self.view_res.to_norm(e.x, e.y); h, w = self.frame.shape[:2]
        x, y = int(u*w), int(v*h)
        if not (0 <= x < w and 0 <= y < h): return
        self._last_pick = (x, y); d = WB.solve(self.frame[max(0, y-3):y+4, max(0, x-3):x+4])
        if d is None:
            self._set_status("Ten punkt jest przepalony albo za ciemny – kliknij jasnoszary punkt bez przepaleń."); return
        t, ti = self.params.temp + d[0], self.params.tint + d[1]
        cut = abs(t) > 1 or abs(ti) > 1
        self._set_wb(float(np.clip(t, -1, 1)), float(np.clip(ti, -1, 1))); self.toggle_pick(False)
        self._set_status("Balans bieli ustawiony z pipety" + (" (poza zakresem suwaków – przycięty)." if cut else "."))

    def _set_wb(self, t, ti):
        self.params.temp, self.params.tint = t, ti
        for k in ("temp", "tint"):
            s_, val, kind = self.sliders[k]; v = getattr(self.params, k)*100; s_.set(v); val.configure(text=self._fmt(v, kind))
        self._changed()

    def reset_wb(self):
        self._set_wb(0.0, 0.0)

    def sync_wb(self):
        if not self.series or not self.photos: return
        if self._save_after: self.after_cancel(self._save_after); self._save_after = None
        self.series.set_override(self.photos[self.idx], Params.from_dict(self.params.to_dict())) if self.series.has_override(self.photos[self.idx]) else None
        self.series.sync_fields(self.photos, {"temp": self.params.temp, "tint": self.params.tint})
        self._refresh_strip_marks(); self._update_override_label()
        self._set_status(f"Balans bieli skopiowany do całej serii ({len(self.photos)} zdjęć).")

    # ================= crop & straighten =================
    def _frame_wh(self):
        if self.src_size: return self.src_size
        return (self.result.shape[1], self.result.shape[0]) if self.result is not None else (3, 2)

    def _crop_portrait(self):
        W, H = self._frame_wh(); c = self.params.crop
        return (c[2]-c[0])*W < (c[3]-c[1])*H if c else H > W

    def _tool_tab(self, name):
        self.tool = "crop" if name == TOOLS[1] else "brush"
        self.tool_frames[name].lift(); self.hline_on = False; self._hline = None; self._style_hline()
        if self.tool == "crop": self.toggle_pick(False); self.view_res.delete("cursor")
        self.redraw_result(); self._draw_crop_overlay()

    def toggle_crop_tool(self):
        name = TOOLS[0] if self.tool == "crop" else TOOLS[1]; self.tool_tabs.set(name); self._tool_tab(name)

    def _crop_done(self):
        if self.tool == "crop": self.toggle_crop_tool()

    def _escape(self):
        if self.picking: self.toggle_pick(False); self._update_status()
        elif self.tool == "crop": self.toggle_crop_tool()

    def _key(self, fn):
        if isinstance(self.focus_get(), ctk.CTkEntry): return
        fn()

    def toggle_hline(self):
        self.hline_on = not self.hline_on; self._style_hline()
        if self.hline_on: self._set_status("Linia horyzontu: przeciągnij wzdłuż horyzontu lub pionowej krawędzi.")

    def _style_hline(self):
        self.hline_btn.configure(border_color=T.ACCENT if self.hline_on else T.BORDER, text_color=T.ACCENT if self.hline_on else T.TEXT)

    def _crop_ui_sync(self):
        a = self.params.angle; self.angle.set(a)
        self.angle_val.configure(text=f"{a:+.1f}°".replace(".", ",") if a else "0°")
        self.aspect_menu.set(G.ASPECTS.get(self.params.aspect, G.ASPECTS["orig"]))

    def _geometry_changed(self):
        self._crop_ui_sync(); self.redraw_result(); self._changed(render=False)

    def _aspect(self, label):
        key = {v: k for k, v in G.ASPECTS.items()}[label]; self.params.aspect = key
        if key != "free":
            W, H = self._frame_wh(); self.params.crop = G.clamp_crop(G.aspect_rect(key, W, H, self._crop_portrait()))
        self._geometry_changed()

    def _flip_aspect(self):
        W, H = self._frame_wh(); c = self.params.crop
        if self.params.aspect == "free":
            if c is None: return
            cx, cy = (c[0]+c[2])/2, (c[1]+c[3])/2; hw, hh = (c[3]-c[1])*H/W/2, (c[2]-c[0])*W/H/2
            k = min(1.0, cx/hw, (1-cx)/hw, cy/hh, (1-cy)/hh); hw, hh = hw*k, hh*k
            self.params.crop = G.clamp_crop([cx-hw, cy-hh, cx+hw, cy+hh])
        else:
            self.params.crop = G.clamp_crop(G.aspect_rect(self.params.aspect, W, H, not self._crop_portrait()))
        self._geometry_changed()

    def _angle(self, v):
        self.params.angle = round(float(v), 1)
        self.angle_val.configure(text=f"{self.params.angle:+.1f}°".replace(".", ",") if self.params.angle else "0°")
        self.detail_raw = None; self.view_res.detail = None
        self.redraw_result()
        if self._geo_after: self.after_cancel(self._geo_after)
        self._geo_after = self.after(400, lambda: (self._changed(render=False), self._schedule_detail()))

    def reset_crop(self):
        self.params.angle = 0.0; self.params.crop = None; self.params.aspect = "orig"
        self.detail_raw = None; self.view_res.detail = None; self._geometry_changed()

    def _crop_press(self, e):
        v = self.view_res
        if self._space or self.result is None: self._pan_start(v, e); return
        if self.hline_on: self._hline = (e.x, e.y, e.x, e.y); self._crop_drag = ("line",); return
        u, w_ = (float(np.clip(t, 0, 1)) for t in v.to_norm(e.x, e.y))
        c = self.params.crop or [0.0, 0.0, 1.0, 1.0]; x0, y0, dw, dh = v.geom
        corners = [(c[0], c[1]), (c[2], c[1]), (c[0], c[3]), (c[2], c[3])]
        for i, (cx, cy) in enumerate(corners):
            if abs(x0+cx*dw-e.x) < 14 and abs(y0+cy*dh-e.y) < 14:
                self._crop_drag = ("resize", corners[3-i]); return
        if self.params.crop and c[0] < u < c[2] and c[1] < w_ < c[3]:
            self._crop_drag = ("move", (u, w_), list(c)); return
        self._crop_drag = ("new", (u, w_))

    def _crop_move(self, e):
        if self._pan is not None: self._pan_move(self.view_res, e); return
        d = self._crop_drag
        if not d: return
        if d[0] == "line":
            self._hline = (*self._hline[:2], e.x, e.y); self._draw_crop_overlay(); return
        u, w_ = (float(np.clip(t, 0, 1)) for t in self.view_res.to_norm(e.x, e.y)); W, H = self._frame_wh()
        if d[0] == "move":
            (pu, pv), c = d[1], d[2]
            dx = float(np.clip(u-pu, -c[0], 1-c[2])); dy = float(np.clip(w_-pv, -c[1], 1-c[3]))
            self._crop_live = [c[0]+dx, c[1]+dy, c[2]+dx, c[3]+dy]
        else:
            ax, ay = d[1]; rect = [min(ax, u), min(ay, w_), max(ax, u), max(ay, w_)]
            if self.params.aspect != "free":
                portrait = abs(u-ax)*W < abs(w_-ay)*H if d[0] == "new" else self._crop_portrait()
                r = G.ratio(self.params.aspect, W, H, portrait)
                if r: rect = G.fit_aspect(rect, r, W, H, anchor=(ax, ay))
            self._crop_live = rect
        self._draw_crop_overlay()

    def _crop_release(self, e):
        if self._pan is not None: self._pan_end(); return
        d, self._crop_drag = self._crop_drag, None
        if not d: return
        if d[0] == "line":
            x0_, y0_, x1_, y1_ = self._hline; self._hline = None
            if np.hypot(x1_-x0_, y1_-y0_) > 12:
                W, H = self._frame_wh(); v = self.view_res
                a = np.array(v.to_norm(x0_, y0_))*(W, H); b = np.array(v.to_norm(x1_, y1_))*(W, H)
                self.params.angle = float(np.clip(round(self.params.angle + G.horizon_angle(a, b), 1), -45, 45))
                self.detail_raw = None; self.view_res.detail = None
                self._set_status(f"Wyprostowano: kąt {self.params.angle:+.1f}°".replace(".", ","))
            else: self._update_status()
            self.hline_on = False; self._style_hline(); self._geometry_changed(); self._schedule_detail(); return
        live, self._crop_live = self._crop_live, None
        if live is None: self._draw_crop_overlay(); return
        if (live[2]-live[0]) < 0.01 or (live[3]-live[1]) < 0.01:
            self._draw_crop_overlay(); return
        self.params.crop = G.clamp_crop(live); self._geometry_changed()

    def _draw_crop_overlay(self):
        v = self.view_res; v.delete("crop")
        if self.result is None or self.tool != "crop": return
        x0, y0, dw, dh = v.geom
        c = self._crop_live or self.params.crop or [0.0, 0.0, 1.0, 1.0]
        X0, Y0, X1, Y1 = x0+c[0]*dw, y0+c[1]*dh, x0+c[2]*dw, y0+c[3]*dh
        for r in ((x0, y0, x0+dw, Y0), (x0, Y1, x0+dw, y0+dh), (x0, Y0, X0, Y1), (X1, Y0, x0+dw, Y1)):
            if r[2]-r[0] > 0.5 and r[3]-r[1] > 0.5: v.create_rectangle(*r, fill="#000000", stipple="gray50", outline="", tags="crop")
        for i in (1, 2):
            xx, yy = X0+(X1-X0)*i/3, Y0+(Y1-Y0)*i/3
            v.create_line(xx, Y0, xx, Y1, fill="#BDBDBD", tags="crop"); v.create_line(X0, yy, X1, yy, fill="#BDBDBD", tags="crop")
        v.create_rectangle(X0, Y0, X1, Y1, outline="#FFFFFF", width=1, tags="crop")
        L = min(16, (X1-X0)/3, (Y1-Y0)/3)
        for cx, cy, sx, sy in ((X0, Y0, 1, 1), (X1, Y0, -1, 1), (X0, Y1, 1, -1), (X1, Y1, -1, -1)):
            v.create_line(cx, cy+sy*L, cx, cy, cx+sx*L, cy, fill=T.ACCENT, width=3, tags="crop")
        if self._hline: v.create_line(*self._hline, fill=T.ACCENT, width=2, dash=(6, 3), tags="crop")

    # ================= controls =================
    def _load_params_into_ui(self, p: Params):
        for key, (s, val, kind) in self.sliders.items():
            v = getattr(p, key)*100; s.set(v); val.configure(text=self._fmt(v, kind))
        if p.light_angle is None: self.light_auto.select()
        else: self.light_auto.deselect(); self.light.set(p.light_angle)
        self._show_light(); self._crop_ui_sync()
        self._update_override_label()

    def _auto_angle(self):
        r = self.fit.relight if self.fit is not None and self.fit.relight is not None else None
        return r.auto_angle if r is not None and r.auto_angle is not None else 120.0

    @staticmethod
    def _dir_name(a):
        names = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]   # where the light comes from (short: fixed-width box)
        return names[int(((a % 360)+22.5)//45) % 8]

    def _show_light(self):
        auto = self.params.light_angle is None
        a = self._auto_angle() if auto else self.params.light_angle
        if auto: self.light.set(a)
        self.light_val.configure(text=f"{int(a)}° {self._dir_name(a)}")
        self.light.configure(state="disabled" if auto else "normal")

    def _light(self, v):
        if self.params.light_angle is None: return
        self.params.light_angle = float(v); self._show_light(); self._changed()

    def _light_auto(self):
        self.params.light_angle = None if self.light_auto.get() else float(self.light.get())
        self._show_light(); self._changed()

    @staticmethod
    def _fmt(v, kind):
        return f"{int(round(v))}%" if kind == "pct" else f"{int(round(v)):+d}" if round(v) else "0"

    def _slider(self, key, v):
        s, val, kind = self.sliders[key]; val.configure(text=self._fmt(v, kind))
        setattr(self.params, key, float(v)/100); self._changed()

    def _changed(self, render=True):
        if self.series and self.idx >= 0:
            if self._save_after: self.after_cancel(self._save_after)
            p, ph = Params.from_dict(self.params.to_dict()), self.photos[self.idx]
            self._save_after = self.after(300, lambda: (self.series.set_override(ph, p), self._refresh_strip_marks(), self._update_override_label()))
        if render: self.request_render()

    def _update_override_label(self):
        own = bool(self.series and self.idx >= 0 and self.series.has_override(self.photos[self.idx]))
        self.override_lbl.configure(text="własne (.retusz.json)" if own else "ustawienia serii",
                                    text_color=T.ACCENT if own else T.MUTED)

    def reset_to_series(self):
        if not self.series or self.idx < 0: return
        if self._save_after: self.after_cancel(self._save_after); self._save_after = None
        self.series.clear_override(self.photos[self.idx]); self.params = self.series.params_for(self.photos[self.idx])
        self._load_params_into_ui(self.params); self._refresh_strip_marks(); self.request_render()

    def make_series_default(self):
        if not self.series: return
        self.series.params = Params.from_dict({**self.params.to_dict(), "strokes": []}); self.series.save()
        if self.idx >= 0 and not self.params.strokes and self.params.crop is None and not self.params.angle:
            if self._save_after: self.after_cancel(self._save_after); self._save_after = None
            self.series.clear_override(self.photos[self.idx])
        self._refresh_strip_marks(); self._update_override_label(); self._set_status("Zapisano jako domyślne dla całej serii.")

    # ================= brush =================
    def _brush_px(self):
        """Brush radius in original-image pixels (log slider 5..600 px)."""
        return BRUSH_MIN*(BRUSH_MAX/BRUSH_MIN)**float(self.brush_size.get())

    def _brush_label(self, key):
        lab = self.brush_vals[key]
        if key == "size": lab.configure(text=f"{int(round(self._brush_px()))} px")
        elif key == "soft": lab.configure(text=f"{int(round(self.brush_soft.get()))}%")
        else: lab.configure(text=f"{int(round(self.brush_str.get()))}%")

    def _brush_step(self, d):
        if isinstance(self.focus_get(), ctk.CTkEntry): return
        self.brush_size.set(float(np.clip(self.brush_size.get() + d*0.04, 0, 1))); self._brush_label("size")
        self._redraw_cursor()

    def _brush_on(self):
        return self.brush_mode.get() != BRUSH_MODES[0] and self.result is not None

    def _brush_color(self):
        return T.ACCENT if self.brush_mode.get() == "Rozjaśnij" else "#6FA8FF"

    def _brush_cursor(self, e):
        self._mouse = (e.x, e.y); v = self.view_res; v.delete("cursor")
        if not self._brush_on() or self._space or self.picking or self.tool == "crop": return
        r = max(2.0, self._brush_px()*v.cur_zoom())
        v.create_oval(e.x-r, e.y-r, e.x+r, e.y+r, outline=self._brush_color(), width=1.5, tags="cursor")
        if r > 6: v.create_oval(e.x-1.5, e.y-1.5, e.x+1.5, e.y+1.5, outline=self._brush_color(), tags="cursor")

    def _redraw_cursor(self):
        if getattr(self, "_mouse", None):
            class E: pass
            e = E(); e.x, e.y = self._mouse; self._brush_cursor(e)

    def _brush_press(self, e):
        self.view_res.focus_set()
        if self.picking: self._pick(e); return
        if self.tool == "crop": self._crop_press(e); return
        if self._space or not self._brush_on():
            self._pan_start(self.view_res, e); return
        self._stroke = [self.view_res.to_norm(e.x, e.y)]; self._last = (e.x, e.y)

    def _brush_move(self, e):
        if self.tool == "crop": self._crop_move(e); return
        if self._pan is not None: self._pan_move(self.view_res, e); return
        self._brush_cursor(e)
        if self._stroke is None: return
        self._stroke.append(self.view_res.to_norm(e.x, e.y))
        r = self._brush_px()*self.view_res.cur_zoom()
        self.view_res.create_line(*self._last, e.x, e.y, fill=self._brush_color(), width=max(1, r*1.2), capstyle="round",
                                  stipple="gray25", tags="stroke")
        self._last = (e.x, e.y)

    def _brush_release(self, e):
        if self.tool == "crop": self._crop_release(e); return
        if self._pan is not None: self._pan_end(); return
        if self._stroke is None: return
        W, H = self._frame_wh(); a = self.params.angle
        pts = [[float(np.clip(x, 0, 1)), float(np.clip(y, 0, 1))] for x, y in (G.frame_to_src(q, a, W, H) for q in self._stroke)]
        self._stroke = None
        self.view_res.delete("stroke")
        W = W*G.straighten_scale(W, H, a)                        # brush size in source pixels
        self.params.strokes.append({"mode": "dodge" if self.brush_mode.get() == "Rozjaśnij" else "burn",
                                    "size": round(self._brush_px()/W, 5), "strength": round(self.brush_str.get()/100, 3),
                                    "softness": round(self.brush_soft.get()/100, 3),
                                    "points": pts[::max(1, len(pts)//300)]})
        self._changed()

    def undo_stroke(self):
        if self.params.strokes: self.params.strokes.pop(); self._changed()

    def clear_strokes(self):
        if self.params.strokes: self.params.strokes = []; self._changed()

    # ================= zoom / pan =================
    def _space_key(self, down):
        if self.compare.winfo_ismapped(): self.compare.key_space(down); return
        if isinstance(self.focus_get(), ctk.CTkEntry): return
        self._space = down
        for v in (self.view_src, self.view_res): v.configure(cursor="fleur" if down else "")
        if down: self.view_res.delete("cursor")

    def _wheel(self, v, e, d):
        st = e.state
        alt = bool(st & 0x20000) if sys.platform == "win32" else bool(st & 0x8)
        if alt: self._brush_step(d); return "break"
        if st & 0x4:                                   # Ctrl + wheel: zoom around the cursor
            v.zoom_at(1.25 if d > 0 else 0.8, e.x, e.y); self._redraw_cursor(); return "break"
        if v.zoom is not None:                         # wheel alone pans a zoomed image (Shift: sideways)
            (v.pan(d*80, 0) if st & 0x1 else v.pan(0, d*80)); return "break"

    def _pan_start(self, v, e):
        if v.zoom is None and not self._space: self._pan = None; return
        self._pan = (e.x, e.y)

    def _pan_move(self, v, e):
        if self._pan is None: return
        v.pan(e.x-self._pan[0], e.y-self._pan[1]); self._pan = (e.x, e.y)

    def _pan_end(self):
        self._pan = None

    def _view_changed(self, src_view):
        other = self.view_res if src_view is self.view_src else self.view_src
        other.set_view(src_view.zoom, src_view.center)
        self._sync_zoom_label(); self._schedule_detail()

    def zoom_fit(self):
        if isinstance(self.focus_get(), ctk.CTkEntry): return
        self.view_src.set_view(None, (0.5, 0.5), notify=True)

    def zoom_100(self):
        if isinstance(self.focus_get(), ctk.CTkEntry): return
        v = self.view_src; c = v.center if v.zoom else (0.5, 0.5)
        v.set_view(1.0, c, notify=True)

    def _sync_zoom_label(self):
        v = self.view_src
        self.zoom_lbl.configure(text="" if v.rgb is None else f"{int(round(v.cur_zoom()*100))}%")

    def _schedule_detail(self):
        """At zoom above the preview resolution, render the visible photo at full resolution (background)."""
        if self._detail_after: self.after_cancel(self._detail_after); self._detail_after = None
        if self.fit is None or not self.view_src.needs_detail(): return
        self._detail_after = self.after(250, self._render_detail)

    def _render_detail(self):
        self._detail_after = None
        if self.fit is None or self.idx < 0: return
        i, ph, fit, eng = self.idx, self.photos[self.idx], self.fit, self.engine
        params = Params.from_dict(self.params.to_dict())
        def work():
            fc = self.full_cache
            if fc is None or fc[0] != ph:
                rgb, _ = imageio.load(ph); fc = self.full_cache = (ph, rgb)
            res = eng.render(fc[1], fit, params)
            return fc[1], (G.apply(res, params.angle, None, interp=cv2.INTER_LINEAR) if params.angle else res)
        def done(r):
            if self.idx != i or params.angle != self.params.angle: return
            self.detail_raw = r[1]; self.view_src.set_detail(r[0]); self.view_res.set_detail(self._decorate(r[1]))
        self._set_status("Liczę pełną rozdzielczość…")
        self.submit(work, lambda r: (done(r), self._update_status()), kind="detail")

    # ================= compare =================
    def _disp(self, rgb):
        return CM.for_display(rgb, self.src_icc)

    def show_compare(self):
        if self.result is None: self._set_status("Najpierw otwórz zdjęcie i wybierz wzorzec."); return
        self.main.pack_forget(); self.compare.pack(fill="both", expand=True, before=self.status)
        self.compare.set_images(self._disp(self.src_prev), self._disp(self.result))

    def hide_compare(self):
        self.compare.pack_forget(); self.main.pack(fill="both", expand=True, before=self.status)

    def _compare_full(self):
        if self.fit is None: return
        p, fit, eng, params = self.photos[self.idx], self.fit, self.engine, Params.from_dict(self.params.to_dict())
        def work():
            rgb, _ = imageio.load(p); return self._disp(rgb), self._disp(eng.render(rgb, fit, params))
        self.submit(work, lambda r: self.compare.set_full(*r), kind="full")

    # ================= export =================
    def _out_dir(self, photo):
        return Path(photo).parent / self.settings.output_subfolder

    def save_current(self):
        if self.fit is None: self._set_status("Najpierw otwórz zdjęcie i wybierz wzorzec."); return
        p, eng, ref, fit = self.photos[self.idx], self.engine, self.reference, self.fit
        params, opts = Params.from_dict(self.params.to_dict()), self.settings.export_options()
        self._set_status(f"Zapisuję {p.name}…")
        def work():
            return batch.export_one(eng, ref, p, params, self._out_dir(p), opts, fit=fit)   # same fit as the preview
        self.submit(work, lambda dst: self._set_status(f"Zapisano: {dst}"))

    def export_series(self):
        if not self.photos or self.reference is None or self.engine is None:
            self._set_status("Najpierw otwórz folder i wybierz wzorzec."); return
        dlg = ProgressDialog(self, f"Eksport serii · {len(self.photos)} zdjęć")
        photos, out_dir = list(self.photos), self._out_dir(self.photos[0])
        eng, ref, series, opts = self.engine, self.reference, self.series, self.settings.export_options()
        def prog(i, n, ph, sec): self._ui.put((lambda _: dlg.winfo_exists() and dlg.update_progress(i, n, f"{i}/{n} · {Path(ph).name} · {sec:.1f} s"), None))
        def work():
            res = batch.export_series(eng, ref, series, photos, out_dir, opts, prog, lambda: dlg.cancelled)
            ok = sum(1 for r in res if r[1]); err = [r for r in res if r[2]]
            msg = f"Gotowe: {ok} z {len(photos)} zapisanych w {out_dir}" + (f" · błędy: {len(err)}" if err else "")
            self._ui.put((lambda _: dlg.winfo_exists() and dlg.done(msg), None)); self._ui.put((lambda _: self._set_status(msg), None))
        threading.Thread(target=work, daemon=True).start()

    # ================= status =================
    def _set_status(self, text):
        self.status.configure(text=text)

    def _update_status(self):
        parts = []
        if self.fit is not None:
            parts.append(f"Twarze: {len(self.fit.faces_n)}")
            if self.fit.relight is not None: parts.append(f"3D: {self.fit.relight.method}")
        if self.src_size: parts.append(f"{self.src_size[0]}×{self.src_size[1]}")
        parts.append(f"podgląd {self.settings.preview_side} px")
        if self.engine_name: parts.append(f"silnik: {self.engine_name} · {DEV.summary() if DEV.ACTIVE else self.device}")
        if self.src_icc is not None: parts.append(f"profil: {CM.description(self.src_icc)}")
        if self.reference is None: parts.append("brak wzorca")
        self._set_status("  ·  ".join(parts))

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="Retusz")
    ap.add_argument("path", nargs="?", help="zdjęcie lub folder")
    ap.add_argument("--reference", "-r", help="wzorzec")
    a = ap.parse_args(argv)
    RetuszApp(a.path, a.reference).mainloop()
