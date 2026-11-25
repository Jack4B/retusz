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
    """Ustawienia: Ogólne (podgląd, silnik, urządzenie, model) and Eksport (JPG, profil, DPI, rozmiar, wyostrzanie, metadane)."""
    def __init__(self, master, settings, on_save):
        super().__init__(master, "Ustawienia", 560, 660); self.s = settings; self.on_save = on_save
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
        m = ctk.CTkOptionMenu(g, values=values, fg_color=T.BTN, button_color=T.BTN_HOVER, text_color=T.TEXT, width=230, command=command)
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
