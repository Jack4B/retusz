"""Retusz main window (mockup A) + Porównaj view (mockup B)."""
from __future__ import annotations
import queue, threading, traceback, sys
from pathlib import Path
from tkinter import filedialog, messagebox
import numpy as np
import customtkinter as ctk
from PIL import Image
from . import theme as T
from .imageview import ImageView
from .compare import CompareView
from .dialogs import SettingsDialog, ModelDialog, ProgressDialog
from .. import imageio, models, batch, export, colorman as CM, __version__
from ..reference import Reference
from ..settings import Settings
from ..sidecar import Series
from ..engine.pipeline import Engine, Params
from ..engine import render as R, color as C, device as DEV, modflows as MF, relight as RL

ENGINE_LABEL = {"ai": "AI (B6)", "ai_light": "AI lekki (B0)", "classic": "bez AI (IDT)"}
SLIDERS = [  # key, label, from, to, display  (4 equal columns, fixed-width value boxes: nothing re-flows while dragging)
    ("strength", "Siła kolorów", 0, 100, "pct"), ("skin", "Skóra", 0, 100, "pct"), ("skin_even", "Równanie skóry", 0, 100, "pct"),
    ("vignette", "Winieta", 0, 100, "pct"),
    ("shadows", "Cienie", -100, 100, "num"), ("highlights", "Światła", -100, 100, "num"), ("saturation", "Nasycenie", -100, 100, "num"),
    ("vibrance", "Jaskrawość", -100, 100, "num"),
    ("wide", "Pełne barwy", 0, 100, "pct"), ("db", "Dodge&Burn", 0, 100, "pct"), ("plastic", "Plastyka 3D", 0, 100, "pct"),
]
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
        threading.Thread(target=self._worker, daemon=True).start()
        self._build(); self._poll()
        self.bind("<Left>", lambda e: self.step(-1)); self.bind("<Right>", lambda e: self.step(1))
        self.bind("<Control-s>", lambda e: self.save_current()); self.bind("<Control-z>", lambda e: self.undo_stroke())
        self.bind("<KeyPress-space>", lambda e: self._space_key(True)); self.bind("<KeyRelease-space>", lambda e: self._space_key(False))
        self.bind("<bracketleft>", lambda e: self._brush_step(-1)); self.bind("<bracketright>", lambda e: self._brush_step(1))
        self.bind("<Control-0>", lambda e: self.zoom_fit()); self.bind("<Control-1>", lambda e: self.zoom_100())
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
            cell = ctk.CTkFrame(sl, fg_color=T.CARD); cell.grid(row=i//4, column=i % 4, sticky="ew", padx=8, pady=3)
            cell.grid_columnconfigure(0, weight=1)
            T.label(cell, text, 13, anchor="w").grid(row=0, column=0, sticky="w")
            box, val = value_box(cell, 46); box.grid(row=1, column=1, padx=(6, 0))
            s = T.slider(cell, lo, hi, number_of_steps=hi-lo, width=60, command=lambda v, k=key: self._slider(k, v))
            s.grid(row=1, column=0, sticky="ew")
            self.sliders[key] = (s, val, kind)
        cell = ctk.CTkFrame(sl, fg_color=T.CARD); cell.grid(row=2, column=3, sticky="ew", padx=8, pady=3)
        cell.grid_columnconfigure(0, weight=1)
        T.label(cell, "Światło 3D", 13, anchor="w").grid(row=0, column=0, sticky="w")
        self.light_auto = ctk.CTkSwitch(cell, text="Auto", progress_color=T.ACCENT, text_color=T.TEXT, width=60,
                                        switch_width=30, switch_height=14, font=T.font(11), command=self._light_auto)
        self.light_auto.grid(row=0, column=1, sticky="e")
        box, self.light_val = value_box(cell, 62); box.grid(row=1, column=1, padx=(6, 0))
        self.light = T.slider(cell, 0, 355, number_of_steps=71, width=60, command=self._light); self.light.grid(row=1, column=0, sticky="ew")
        br = ctk.CTkFrame(ctl, fg_color=T.CARD)
        T.label(br, "Pędzel dodge/burn", 13).grid(row=0, column=0, columnspan=3, sticky="w")
        self.brush_mode = T.segmented(br, BRUSH_MODES, command=lambda v: self.view_res.delete("cursor")); self.brush_mode.set(BRUSH_MODES[0])
        self.brush_mode.grid(row=1, column=0, columnspan=3, sticky="ew", pady=4)
        self.brush_vals = {}
        def brow(r, text, lo, hi, val, key):
            T.label(br, text, 12, T.MUTED).grid(row=r, column=0, sticky="w")
            box = ctk.CTkFrame(br, fg_color=T.CARD, width=56, height=20); box.grid(row=r, column=2, padx=(6, 0)); box.pack_propagate(False)
            lab = T.label(box, "", 12, T.MUTED, anchor="e"); lab.pack(fill="both", expand=True); self.brush_vals[key] = lab
            sl_ = T.slider(br, lo, hi, width=130, command=lambda v: self._brush_label(key)); sl_.set(val); sl_.grid(row=r, column=1)
            return sl_
        self.brush_size = brow(2, "Rozmiar", 0, 1, 0.45, "size")
        self.brush_soft = brow(3, "Miękkość", 0, 100, 50, "soft")
        self.brush_str = brow(4, "Przepływ", 5, 100, 40, "flow")
        for k in self.brush_vals: self._brush_label(k)
        T.label(br, "[ ] lub Alt+kółko: rozmiar · Ctrl+kółko: zoom", 11, T.MUTED).grid(row=6, column=0, columnspan=3, sticky="w", pady=(4, 0))
        bb = ctk.CTkFrame(br, fg_color=T.CARD); bb.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        T.button(bb, "Cofnij", self.undo_stroke, width=80).pack(side="left")
        T.button(bb, "Wyczyść", self.clear_strokes, width=80).pack(side="left", padx=6)
        ser = ctk.CTkFrame(ctl, fg_color=T.CARD)
        T.label(ser, "To zdjęcie", 13).pack(anchor="w")
        self.override_lbl = T.label(ser, "ustawienia serii", 12, T.MUTED); self.override_lbl.pack(anchor="w", pady=(0, 6))
        T.button(ser, "Przywróć ustawienia serii", self.reset_to_series, width=200).pack(pady=2)
        T.button(ser, "Ustaw jako domyślne serii", self.make_series_default, width=200).pack(pady=2)
        ser.pack(side="right", fill="y", padx=(0, 16), pady=12); br.pack(side="right", fill="y", padx=(0, 16), pady=12)
        sl.pack(side="left", fill="x", expand=True, padx=16, pady=12)

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
                                       filetypes=[("Zdjęcia", "*.jpg *.jpeg *.png *.tif *.tiff *.webp")])
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
        else: self._set_status("W folderze nie ma zdjęć JPG/PNG/TIFF.")

    def choose_reference(self):
        p = filedialog.askopenfilename(title="Wybierz wzorzec", initialdir=self.settings.last_folder or None,
                                       filetypes=[("Zdjęcia", "*.jpg *.jpeg *.png *.tif *.tiff *.webp")])
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
        self.result = None; self.full_cache = None; self.view_res.set_image(None, keep_view=False); self._set_status(f"Wczytuję {p.name}…")
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
        if self.result is None: return
        img = self.result
        if self.settings.show_mask and self.mask_prev is not None:
            import cv2
            m = cv2.resize(self.mask_prev, (img.shape[1], img.shape[0]))[..., None]
            img = (img*(1-0.45*m) + np.array([245, 163, 59])*0.45*m).astype(np.uint8)
        self.view_res.set_image(img, icc=self.src_icc, full_wh=self.src_size)
        if self.view_res.zoom != self.view_src.zoom or self.view_res.center != self.view_src.center:
            self.view_res.set_view(self.view_src.zoom, self.view_src.center)
        self._sync_zoom_label()

    # ================= controls =================
    def _load_params_into_ui(self, p: Params):
        for key, (s, val, kind) in self.sliders.items():
            v = getattr(p, key)*100; s.set(v); val.configure(text=self._fmt(v, kind))
        if p.light_angle is None: self.light_auto.select()
        else: self.light_auto.deselect(); self.light.set(p.light_angle)
        self._show_light()
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

    def _changed(self):
        if self.series and self.idx >= 0:
            if self._save_after: self.after_cancel(self._save_after)
            p, ph = Params.from_dict(self.params.to_dict()), self.photos[self.idx]
            self._save_after = self.after(300, lambda: (self.series.set_override(ph, p), self._refresh_strip_marks(), self._update_override_label()))
        self.request_render()

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
        if self.idx >= 0 and not self.params.strokes:
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
        if not self._brush_on() or self._space: return
        r = max(2.0, self._brush_px()*v.cur_zoom())
        v.create_oval(e.x-r, e.y-r, e.x+r, e.y+r, outline=self._brush_color(), width=1.5, tags="cursor")
        if r > 6: v.create_oval(e.x-1.5, e.y-1.5, e.x+1.5, e.y+1.5, outline=self._brush_color(), tags="cursor")

    def _redraw_cursor(self):
        if getattr(self, "_mouse", None):
            class E: pass
            e = E(); e.x, e.y = self._mouse; self._brush_cursor(e)

    def _brush_press(self, e):
        self.view_res.focus_set()
        if self._space or not self._brush_on():
            self._pan_start(self.view_res, e); return
        self._stroke = [self.view_res.to_norm(e.x, e.y)]; self._last = (e.x, e.y)

    def _brush_move(self, e):
        if self._pan is not None: self._pan_move(self.view_res, e); return
        self._brush_cursor(e)
        if self._stroke is None: return
        self._stroke.append(self.view_res.to_norm(e.x, e.y))
        r = self._brush_px()*self.view_res.cur_zoom()
        self.view_res.create_line(*self._last, e.x, e.y, fill=self._brush_color(), width=max(1, r*1.2), capstyle="round",
                                  stipple="gray25", tags="stroke")
        self._last = (e.x, e.y)

    def _brush_release(self, e):
        if self._pan is not None: self._pan_end(); return
        if self._stroke is None: return
        pts = [[float(np.clip(x, 0, 1)), float(np.clip(y, 0, 1))] for x, y in self._stroke]; self._stroke = None
        self.view_res.delete("stroke")
        W = (self.src_size or (self.result.shape[1], 0))[0]
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
            return fc[1], eng.render(fc[1], fit, params)
        def done(r):
            if self.idx != i: return
            self.view_src.set_detail(r[0]); self.view_res.set_detail(r[1])
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
