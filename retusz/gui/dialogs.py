"""Ustawienia, pobieranie modelu AI, postęp eksportu."""
import threading
from tkinter import filedialog
import customtkinter as ctk
from . import theme as T
from .. import models, export as EX, colorman as CM
from ..settings import DEVICES
from ..engine import device

class _Dialog(ctk.CTkToplevel):
    def __init__(self, master, title, w=460, h=360):
        super().__init__(master, fg_color=T.BG); self.title(title)
        self.geometry(f"{w}x{h}+{master.winfo_rootx()+120}+{master.winfo_rooty()+90}")
        self.resizable(False, False); self.transient(master)
        self.after(50, self._grab)

    def _grab(self):
        try: self.grab_set(); self.focus_force()
        except Exception: pass

class ModelDialog(_Dialog):
    """First-run download of the AI model; 'Później' keeps the bundled light model."""
    def __init__(self, master, on_done):
        super().__init__(master, "Model AI", 480, 250); self.on_done = on_done; self._cancel = False
        T.label(self, "Model AI do najlepszej jakości", 16, weight="bold").pack(anchor="w", padx=20, pady=(18, 4))
        T.label(self, f"Jednorazowe pobranie {models.AI.size_mb} MB, potem działa offline.\n"
                      "Do tego czasu Retusz używa lekkiego modelu wbudowanego w aplikację.",
                12, T.MUTED, justify="left").pack(anchor="w", padx=20)
        self.bar = ctk.CTkProgressBar(self, progress_color=T.ACCENT, fg_color=T.TRACK); self.bar.set(0)
        self.bar.pack(fill="x", padx=20, pady=(18, 4))
        self.status = T.label(self, "", 12, T.MUTED); self.status.pack(anchor="w", padx=20)
        row = ctk.CTkFrame(self, fg_color=T.BG); row.pack(fill="x", padx=20, pady=14)
        self.btn = T.button(row, "Pobierz", self.start, accent=True); self.btn.pack(side="right")
        T.button(row, "Później", self.later).pack(side="right", padx=8)
        T.button(row, "Wskaż plik…", self.pick, width=120).pack(side="left")
        self.protocol("WM_DELETE_WINDOW", self.later)

    def start(self):
        self.btn.configure(state="disabled", text="Pobieram…")
        def prog(done, total): self.after(0, lambda: (self.bar.set(min(1, done/total)), self.status.configure(text=f"{done/1e6:.0f} / {total/1e6:.0f} MB")))
        def work():
            try:
                models.download(prog, lambda: self._cancel); self.after(0, lambda: self._finish(True, "Model zainstalowany."))
            except models.DownloadError as e:
                self.after(0, lambda: self._finish(False, str(e)))
        threading.Thread(target=work, daemon=True).start()

    def pick(self):
        p = filedialog.askopenfilename(parent=self, title="Plik modelu", filetypes=[("Model ONNX", "*.onnx")])
        if not p: return
        try:
            models.install(p); self._finish(True, "Model zainstalowany.")
        except models.DownloadError as e:
            self.status.configure(text=str(e))

    def _finish(self, ok, msg):
        self.status.configure(text=msg)
        if ok:
            self.on_done(True); self.after(600, self.destroy)
        else:
            self.btn.configure(state="normal", text="Spróbuj ponownie")

    def later(self):
        self._cancel = True; self.on_done(False); self.destroy()

class SettingsDialog(_Dialog):
    """Ustawienia: Ogólne (podgląd, silnik, urządzenie, model) and Eksport (Format, JPG, profil, DPI, rozmiar, wyostrzanie, metadane)."""
    def __init__(self, master, settings, on_save):
        super().__init__(master, "Ustawienia", 560, 700); self.s = settings; self.on_save = on_save
        row_b = ctk.CTkFrame(self, fg_color=T.BG); row_b.pack(side="bottom", fill="x", padx=20, pady=(6, 14))
        T.button(row_b, "Zapisz", self.save, accent=True).pack(side="right")
        T.button(row_b, "Anuluj", self.destroy).pack(side="right", padx=8)
        tabs = ctk.CTkTabview(self, fg_color=T.BG, segmented_button_selected_color="#5A4120",
                              segmented_button_selected_hover_color="#6A4C24", segmented_button_fg_color=T.BTN,
                              segmented_button_unselected_color=T.BTN, text_color=T.TEXT)
        tabs.pack(fill="both", expand=True, padx=16, pady=(6, 0))
        self._general(tabs.add("Ogólne")); self._export(tabs.add("Eksport"))

    @staticmethod
    def _grid(tab):
        g = ctk.CTkFrame(tab, fg_color=T.BG); g.pack(fill="both", expand=True, padx=6, pady=6)
        g.grid_columnconfigure(1, weight=1); return g

    @staticmethod
    def _menu(g, values, current, row, command=None):
        m = ctk.CTkOptionMenu(g, values=values, fg_color=T.BTN, button_color=T.BTN_HOVER, text_color=T.TEXT, width=280, command=command)
        m.set(current); m.grid(row=row, column=1, columnspan=2, sticky="w", pady=5); return m

    @staticmethod
    def _lbl(g, text, row):
        T.label(g, text, 13).grid(row=row, column=0, sticky="w", pady=5, padx=(0, 14))

    def _slider(self, g, row, text, lo, hi, val, fmt=str):
        self._lbl(g, text, row)
        box = ctk.CTkFrame(g, fg_color=T.BG, width=56, height=24); box.grid(row=row, column=2, padx=(8, 0)); box.grid_propagate(False)
        lab = T.label(box, fmt(val), 13, T.MUTED, anchor="e"); lab.place(relx=1, rely=0.5, anchor="e")
        sl = T.slider(g, lo, hi, number_of_steps=int(round((hi-lo)*(10 if fmt is not str else 1))), command=lambda v: lab.configure(text=fmt(v)))
        sl.set(val); sl.grid(row=row, column=1, sticky="ew"); return sl

    def _general(self, tab):
        s = self.s; g = self._grid(tab); r = 0
        self._lbl(g, "Folder wyjściowy", r)
        self.out = ctk.CTkEntry(g, fg_color=T.CARD, border_color=T.BORDER, text_color=T.TEXT)
        self.out.insert(0, s.output_subfolder); self.out.grid(row=r, column=1, columnspan=2, sticky="ew"); r += 1
        T.label(g, "podfolder obok zdjęć; oryginały nie są nadpisywane", 11, T.MUTED).grid(row=r, column=1, columnspan=2, sticky="w"); r += 1
        self._lbl(g, "Podgląd", r); self.prev = self._menu(g, ["1200 px", "1600 px", "2400 px"], f"{s.preview_side} px", r); r += 1
        self._lbl(g, "Silnik", r); self.meth = self._menu(g, list(models.METHODS.values()), models.METHODS.get(s.method, models.METHODS["ai"]), r); r += 1
        self._lbl(g, "Obliczenia AI", r); self.dev = self._menu(g, list(DEVICES.values()), DEVICES.get(s.device, DEVICES["auto"]), r); r += 1
        self.dev_lbl = T.label(g, device.summary(), 11, T.MUTED, justify="left"); self.dev_lbl.grid(row=r, column=1, columnspan=2, sticky="w"); r += 1
        self._lbl(g, "Model AI", r); self.model_lbl = T.label(g, "", 12, T.MUTED); self.model_lbl.grid(row=r, column=1, sticky="w"); r += 1
        mb = ctk.CTkFrame(g, fg_color=T.BG); mb.grid(row=r, column=1, columnspan=2, sticky="w"); r += 1
        T.button(mb, "Pobierz model", self.download, width=130).pack(side="left")
        T.button(mb, "Wskaż plik…", self.pick, width=120).pack(side="left", padx=8)
        self._lbl(g, "Maska skóry", r)
        self.mask = ctk.CTkSwitch(g, text="pokazuj na wyniku", progress_color=T.ACCENT, text_color=T.TEXT)
        if s.show_mask: self.mask.select()
        self.mask.grid(row=r, column=1, sticky="w")
        self._model_status()

    def _export(self, tab):
        o = self.s.export_options(); g = self._grid(tab); r = 0
        self._lbl(g, "Format pliku", r); self.fmt = self._menu(g, list(EX.FORMATS.values()), EX.FORMATS[o.format], r); r += 1
        self.q = self._slider(g, r, "Jakość JPG", 50, 100, o.quality, lambda v: str(int(round(float(v))))); r += 1
        self._lbl(g, "Próbkowanie koloru", r); self.sub = self._menu(g, list(EX.SUBSAMPLING), o.subsampling, r); r += 1
        T.label(g, "4:4:4 = najlepsza jakość, 4:2:0 = mniejszy plik", 11, T.MUTED).grid(row=r, column=1, columnspan=2, sticky="w"); r += 1
        self._lbl(g, "Profil barwny", r); self.prof = self._menu(g, list(CM.OUTPUT_PROFILES.values()), CM.OUTPUT_PROFILES[o.profile], r); r += 1
        self._lbl(g, "DPI", r); self.dpi = ctk.CTkEntry(g, width=90, fg_color=T.CARD, border_color=T.BORDER, text_color=T.TEXT)
        self.dpi.insert(0, str(o.dpi)); self.dpi.grid(row=r, column=1, sticky="w", pady=5); r += 1
        self._lbl(g, "Dłuższy bok (px)", r); self.edge = ctk.CTkEntry(g, width=90, fg_color=T.CARD, border_color=T.BORDER, text_color=T.TEXT)
        self.edge.insert(0, str(o.long_edge or "")); self.edge.grid(row=r, column=1, sticky="w", pady=5)
        T.label(g, "puste = oryginał", 11, T.MUTED).grid(row=r, column=1, padx=(100, 0), sticky="w"); r += 1
        self._lbl(g, "Przyrostek nazwy", r); self.suf = ctk.CTkEntry(g, width=140, fg_color=T.CARD, border_color=T.BORDER, text_color=T.TEXT,
                                                                    placeholder_text="np. _retusz")
        if o.suffix: self.suf.insert(0, o.suffix)
        self.suf.grid(row=r, column=1, sticky="w", pady=5); r += 1
        self._lbl(g, "Metadane", r); self.meta = self._menu(g, list(EX.METADATA.values()), EX.METADATA[o.metadata], r); r += 1
        self._lbl(g, "Wyostrzanie", r); self.sh = self._menu(g, list(EX.SHARPEN.values()), EX.SHARPEN[o.sharpen], r, command=lambda _: self._sync_sharpen_state()); r += 1
        f1 = lambda v: f"{float(v):.1f}"
        self.rad = self._slider(g, r, "  promień (px)", 0.3, 3.0, o.radius, f1); r += 1
        self.amt = self._slider(g, r, "  ilość", 0.0, 3.0, o.amount, f1); r += 1
        self.thr = self._slider(g, r, "  próg", 0.0, 10.0, o.threshold, f1); r += 1
        self.shs = self._slider(g, r, "  mniej na skórze", 0, 100, o.sharpen_skin*100, lambda v: f"{int(round(float(v)))}%"); r += 1
        T.label(g, "Ekran/Druk to gotowe ustawienia; suwaki działają dla »Własne«", 11, T.MUTED).grid(row=r, column=1, columnspan=2, sticky="w")
        self._sync_sharpen_state()

    def _sync_sharpen_state(self):
        cur = self.sh.get()
        custom = cur == EX.SHARPEN["custom"]
        active = cur != EX.SHARPEN["off"]
        for sl in (self.rad, self.amt, self.thr):
            sl.configure(state="normal" if custom else "disabled",
                         progress_color=T.ACCENT if custom else T.BORDER,
                         button_color=T.ACCENT if custom else T.MUTED)
        self.shs.configure(state="normal" if active else "disabled",
                           progress_color=T.ACCENT if active else T.BORDER,
                           button_color=T.ACCENT if active else T.MUTED)

    def _model_status(self):
        self.model_lbl.configure(text=("zainstalowany (119 MB)" if models.have_ai() else "nie pobrany · używany lekki model"))

    def download(self):
        ModelDialog(self, lambda ok: self._model_status())

    def pick(self):
        p = filedialog.askopenfilename(parent=self, title="Plik modelu", filetypes=[("Model ONNX", "*.onnx")])
        if p:
            try: models.install(p)
            except models.DownloadError as e: self.model_lbl.configure(text=str(e)); return
            self._model_status()

    @staticmethod
    def _int(entry, default):
        try: return int(entry.get().strip() or default)
        except ValueError: return default

    def save(self):
        s = self.s; inv = lambda d, v: {b: a for a, b in d.items()}[v]
        s.output_subfolder = self.out.get().strip() or "Retusz"
        s.preview_side = int(self.prev.get().split()[0])
        s.method = inv(models.METHODS, self.meth.get()); s.show_mask = bool(self.mask.get())
        s.device = inv(DEVICES, self.dev.get())
        o = EX.ExportOptions.from_dict({
            "format": inv(EX.FORMATS, self.fmt.get()),
            "quality": int(round(self.q.get())), "subsampling": self.sub.get(), "profile": inv(CM.OUTPUT_PROFILES, self.prof.get()),
            "dpi": self._int(self.dpi, 300), "long_edge": self._int(self.edge, 0), "suffix": self.suf.get().strip(),
            "metadata": inv(EX.METADATA, self.meta.get()), "sharpen": inv(EX.SHARPEN, self.sh.get()),
            "radius": round(self.rad.get(), 2), "amount": round(self.amt.get(), 2), "threshold": round(self.thr.get(), 2),
            "sharpen_skin": round(self.shs.get()/100, 2)})
        s.export = o.to_dict(); s.jpg_quality = o.quality
        s.save(); self.on_save(); self.destroy()

class ProgressDialog(_Dialog):
    def __init__(self, master, title):
        super().__init__(master, title, 460, 190); self.cancelled = False
        self.title_lbl = T.label(self, title, 15, weight="bold"); self.title_lbl.pack(anchor="w", padx=20, pady=(18, 6))
        self.bar = ctk.CTkProgressBar(self, progress_color=T.ACCENT, fg_color=T.TRACK); self.bar.set(0); self.bar.pack(fill="x", padx=20)
        self.status = T.label(self, "", 12, T.MUTED); self.status.pack(anchor="w", padx=20, pady=6)
        self.btn = T.button(self, "Przerwij", self.cancel); self.btn.pack(anchor="e", padx=20, pady=8)
        self.protocol("WM_DELETE_WINDOW", self.cancel)

    def update_progress(self, i, n, text):
        self.bar.set(i/max(n, 1)); self.status.configure(text=text)

    def done(self, text):
        self.bar.set(1); self.status.configure(text=text); self.btn.configure(text="Zamknij", command=self.destroy)

    def cancel(self):
        if self.btn.cget("text") == "Zamknij": self.destroy(); return
        self.cancelled = True; self.status.configure(text="Przerywam po bieżącym zdjęciu…")


class ExportAsDialog(_Dialog):
    """Okno 'Eksportuj jako… (Ctrl+Shift+E)': szybki wybór zakresu (bieżące zdjęcie / wybrane ★ / cała seria),
    formatu (JPG 8-bit, TIFF 16-bit, TIFF 8-bit, PNG 16-bit, PNG 8-bit, WebP), przestrzeni barwnej ICC
    (Jak oryginał, sRGB, Display P3, Adobe RGB 1998, ProPhoto RGB), rozmiaru, wyostrzania i folderu docelowego."""
    SCOPES = {
        "current": "Bieżące zdjęcie",
        "starred": "Wybrane zdjęcia (★)",
        "series": "Cała seria",
    }

    def __init__(self, master, settings, default_dir, on_export, default_scope="current"):
        super().__init__(master, "Eksportuj jako…", 560, 560)
        self.s = settings
        self.on_export = on_export
        o = settings.export_options()

        row_b = ctk.CTkFrame(self, fg_color=T.BG)
        row_b.pack(side="bottom", fill="x", padx=20, pady=(6, 14))
        T.button(row_b, "Eksportuj teraz", self._run, accent=True, width=140).pack(side="right")
        T.button(row_b, "Anuluj", self.destroy, width=90).pack(side="right", padx=8)

        g = SettingsDialog._grid(self)
        r = 0
        T.label(g, "Eksportuj zdjęcie lub serię (Eksportuj jako…)", 15, weight="bold").grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(4, 10)
        )
        r += 1

        SettingsDialog._lbl(g, "Zakres eksportu", r)
        self.scope = SettingsDialog._menu(g, list(self.SCOPES.values()), self.SCOPES.get(default_scope, self.SCOPES["current"]), r)
        r += 1

        SettingsDialog._lbl(g, "Format i głębia", r)
        self.fmt = SettingsDialog._menu(g, list(EX.FORMATS.values()), EX.FORMATS.get(o.format, EX.FORMATS["jpg"]), r)
        r += 1

        SettingsDialog._lbl(g, "Przestrzeń barwna ICC", r)
        self.prof = SettingsDialog._menu(g, list(CM.OUTPUT_PROFILES.values()), CM.OUTPUT_PROFILES.get(o.profile, CM.OUTPUT_PROFILES["keep"]), r)
        r += 1

        SettingsDialog._lbl(g, "Jakość (JPG / WebP)", r)
        self.q = ctk.CTkSlider(g, from_=50, to=100, number_of_steps=50, progress_color=T.ACCENT, button_color=T.ACCENT)
        self.q.set(o.quality)
        self.q.grid(row=r, column=1, sticky="ew", pady=5)
        r += 1

        SettingsDialog._lbl(g, "Próbkowanie koloru", r)
        self.sub = SettingsDialog._menu(g, list(EX.SUBSAMPLING), o.subsampling, r)
        r += 1

        SettingsDialog._lbl(g, "Dłuższy bok (px)", r)
        self.edge = ctk.CTkEntry(g, width=110, fg_color=T.CARD, border_color=T.BORDER, text_color=T.TEXT)
        self.edge.insert(0, str(o.long_edge or ""))
        self.edge.grid(row=r, column=1, sticky="w", pady=5)
        T.label(g, "puste = pełna rozdzielczość (światło liniowe)", 11, T.MUTED).grid(
            row=r, column=1, columnspan=2, padx=(118, 0), sticky="w"
        )
        r += 1

        SettingsDialog._lbl(g, "DPI", r)
        self.dpi = ctk.CTkEntry(g, width=90, fg_color=T.CARD, border_color=T.BORDER, text_color=T.TEXT)
        self.dpi.insert(0, str(o.dpi))
        self.dpi.grid(row=r, column=1, sticky="w", pady=5)
        r += 1

        SettingsDialog._lbl(g, "Wyostrzanie wyjściowe", r)
        self.sh = SettingsDialog._menu(g, list(EX.SHARPEN.values()), EX.SHARPEN.get(o.sharpen, EX.SHARPEN["screen"]), r)
        r += 1

        SettingsDialog._lbl(g, "Przyrostek nazwy", r)
        self.suf = ctk.CTkEntry(g, width=140, fg_color=T.CARD, border_color=T.BORDER, text_color=T.TEXT, placeholder_text="np. _studio")
        if o.suffix:
            self.suf.insert(0, o.suffix)
        self.suf.grid(row=r, column=1, sticky="w", pady=5)
        r += 1

        SettingsDialog._lbl(g, "Folder docelowy", r)
        d_row = ctk.CTkFrame(g, fg_color=T.BG)
        d_row.grid(row=r, column=1, columnspan=2, sticky="ew", pady=5)
        d_row.grid_columnconfigure(0, weight=1)
        self.dir_entry = ctk.CTkEntry(d_row, fg_color=T.CARD, border_color=T.BORDER, text_color=T.TEXT)
        self.dir_entry.insert(0, str(default_dir or ""))
        self.dir_entry.grid(row=0, column=0, sticky="ew")
        T.button(d_row, "Wybierz…", self._pick_dir, width=84).grid(row=0, column=1, padx=(6, 0))

    def _pick_dir(self):
        p = filedialog.askdirectory(parent=self, title="Wybierz folder eksportu", initialdir=self.dir_entry.get() or None)
        if p:
            self.dir_entry.delete(0, "end")
            self.dir_entry.insert(0, p)

    def _run(self):
        inv = lambda d, v: {b: a for a, b in d.items()}[v]
        base_o = self.s.export_options()
        opts = EX.ExportOptions.from_dict({
            **base_o.to_dict(),
            "format": inv(EX.FORMATS, self.fmt.get()),
            "profile": inv(CM.OUTPUT_PROFILES, self.prof.get()),
            "quality": int(round(self.q.get())),
            "subsampling": self.sub.get(),
            "long_edge": SettingsDialog._int(self.edge, 0),
            "dpi": SettingsDialog._int(self.dpi, 300),
            "sharpen": inv(EX.SHARPEN, self.sh.get()),
            "suffix": self.suf.get().strip(),
        })
        scope_key = inv(self.SCOPES, self.scope.get())
        out_dir = self.dir_entry.get().strip()
        self.destroy()
        self.on_export(scope_key, opts, out_dir)


COPY_GROUPS = (
    ("wb", "Balans bieli (Temperatura, Odcień)", ("temp", "tint"), True),
    ("tone", "Tonacja i Obecność (Podstawowe)", (
        "exposure", "contrast", "brightness", "highlights", "shadows", "whites", "blacks", "vibrance", "saturation"
    ), True),
    ("hsl", "Kolor HSL (8 kanałów) i Color Grading", (
        "shadow_tone", "midtone_tone", "highlight_tone", "grading_balance",
        "hsl_green", "hsl_blue", "hsl_hue", "hsl_sat", "hsl_lum"
    ), True),
    ("portrait", "Portret, Skóra, Światło 3D i Światło mieszane", (
        "db", "plastic", "light_angle", "light_x", "light_y", "bg_exposure",
        "skin_hue", "mixed_light", "skin_cast", "skin_even", "deshine", "blemish", "eyes_teeth"
    ), True),
    ("detail", "Detal, Ostrość, Przejrzystość i Winieta", (
        "sharpen_amount", "sharpen_mode", "texture", "clarity", "denoise", "dehaze", "vignette"
    ), True),
    ("ref", "Siła wzorca i Maska AI", ("strength", "skin", "mask_hue_tol", "mask_protect", "wide"), True),
    ("crop", "Kadr i Prostowanie (ostrożnie!)", ("angle", "crop", "aspect"), False),
)


class SelectiveCopyDialog(_Dialog):
    """Lightroom Classic style 'Kopiuj ustawienia…' dialog with group checkboxes (crop unchecked by default)."""
    def __init__(self, master, on_copy):
        super().__init__(master, "Kopiuj wybrane ustawienia (Lightroom Copy)", 470, 390)
        self.on_copy = on_copy
        T.label(self, "Wybierz grupy parametrów do skopiowania", 15, weight="bold").pack(anchor="w", padx=20, pady=(14, 4))
        T.label(self, "Kadr oraz lokalny pędzel nie są zaznaczone domyślnie, aby chronić kompozycję innych zdjęć.",
                11, T.MUTED, wraplength=430, justify="left").pack(anchor="w", padx=20, pady=(0, 8))
        box = ctk.CTkFrame(self, fg_color=T.CARD, corner_radius=8)
        box.pack(fill="both", expand=True, padx=20, pady=4)
        self.vars: dict[str, ctk.BooleanVar] = {}
        for gk, label, _, default_on in COPY_GROUPS:
            v = ctk.BooleanVar(value=default_on)
            self.vars[gk] = v
            cb = ctk.CTkCheckBox(box, text=label, variable=v, fg_color=T.ACCENT, hover_color=T.ACCENT_HOVER,
                                 text_color=T.TEXT, font=T.font(12), checkbox_width=18, checkbox_height=18)
            cb.pack(anchor="w", padx=14, pady=5)

        row = ctk.CTkFrame(self, fg_color=T.BG)
        row.pack(fill="x", padx=20, pady=12)
        T.button(row, "Wszystkie", lambda: [v.set(True) for v in self.vars.values()], width=86).pack(side="left")
        T.button(row, "Bez kadru", lambda: [v.set(k != "crop") for k, v in self.vars.items()], width=86).pack(side="left", padx=6)
        T.button(row, "Kopiuj", self._apply, accent=True, width=96).pack(side="right")
        T.button(row, "Anuluj", self.destroy, width=82).pack(side="right", padx=6)

    def _apply(self):
        keys: list[str] = []
        for gk, _, gkeys, _ in COPY_GROUPS:
            if self.vars[gk].get():
                keys.extend(gkeys)
        self.destroy()
        self.on_copy(keys)


class ShortcutsDialog(_Dialog):
    """Lightroom Classic keyboard & mouse ergonomics cheat-sheet ('?')."""
    def __init__(self, master):
        super().__init__(master, "Skróty klawiszowe i ergonomia (Lightroom UX)", 560, 480)
        T.label(self, "Skróty i szybka obsługa — Retusz Studio Pro", 15, weight="bold").pack(anchor="w", padx=20, pady=(14, 6))
        scroll = ctk.CTkScrollableFrame(self, fg_color=T.CARD, corner_radius=8)
        scroll.pack(fill="both", expand=True, padx=20, pady=4)
        items = (
            ("G  /  D  /  Y", "Moduł Biblioteka (G) · Edycja (D) · Widok Obok siebie Przed|Po (Y)"),
            ("\\  (backslash)", "Szybkie przełączenie Przed / Po na głównym podglądzie"),
            ("F6", "Pokaż / Ukryj boczny pasek miniatur serii (maksymalne miejsce na portret)"),
            ("R lub C", "Narzędzie Kadr i Prostowanie (Enter / Esc — zamknij szufladę)"),
            ("Q", "Pędzel Usuń / Popraw (inpainting + transfer prawdziwej tekstury porów)"),
            ("K", "Pędzel lokalny Dodge & Burn ([ / ] lub Alt+kółko — zmiana rozmiaru)"),
            ("W  /  M  /  J", "Pipeta balansu bieli (W) · Diagnostyka Maska AI (M) · Przepalenia i cienie (J)"),
            ("P  /  X  /  U", "Flagi selekcji: Wybrane ⚑ (P) · Odrzucone ✕ (X) · Usuń flagę (U)"),
            ("0 – 5", "Ocena gwiazdkowa bieżącego zdjęcia (0..5 ★)"),
            ("Kółko nad suwakiem", "Precyzyjna korekta ±2 bez odrywania wzroku od twarzy (Shift+kółko: ±10)"),
            ("Przeciągnij nazwę suwaka", "Płynny scrubbing mikro-korekty (0,25× czułości myszy)"),
            ("Klik w liczbę / Dwuklik", "Kliknięcie wartości pozwala wpisać dokładną liczbę; dwuklik zeruje suwak"),
            ("Przeciągnij po Histogramie", "Bezpośrednia regulacja stref: Czernie · Cienie · Ekspozycja · Światła · Biele"),
            ("Ctrl+Z / Ctrl+Y", "Cofnij / Ponów zmianę"),
            ("Ctrl+Shift+C / Ctrl+Shift+V", "Selektywne Kopiuj ustawienia… / Wklej ustawienia"),
            ("Ctrl+Shift+E / Ctrl+S", "Eksportuj jako… (TIFF/PNG 16-bit, ICC) / Szybki zapis"),
        )
        scroll.grid_columnconfigure(1, weight=1)
        for idx, (k_txt, desc) in enumerate(items):
            T.label(scroll, k_txt, 11, T.ACCENT, weight="bold", anchor="w").grid(row=idx, column=0, sticky="nw", padx=(10, 14), pady=4)
            T.label(scroll, desc, 11, T.TEXT, anchor="w", justify="left", wraplength=330).grid(row=idx, column=1, sticky="w", pady=4)
        row = ctk.CTkFrame(self, fg_color=T.BG)
        row.pack(fill="x", padx=20, pady=10)
        T.button(row, "Zamknij", self.destroy, accent=True, width=100).pack(side="right")


