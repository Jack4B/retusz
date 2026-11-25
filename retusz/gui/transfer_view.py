"""Okno „Przenieś grading z AI”: oryginał w pełnej rozdzielczości + wersja z narzędzia AI (mniejsza,
przesunięta, przycięta) -> kolor i światło z AI, szczegóły i ostrość z oryginału."""
from __future__ import annotations
import queue, threading, traceback, sys
from pathlib import Path
from tkinter import filedialog
import numpy as np, cv2
import customtkinter as ctk
from . import theme as T
from .imageview import ImageView
from .. import imageio, grading as GR
from ..engine import transfer as TR
from ..engine.align import AlignmentError

PREVIEW = 1600
MODES = ["Wynik", "Oryginał", "AI", "Miganie", "Podział"]
IMG_TYPES = [("Zdjęcia", "*.jpg *.jpeg *.png *.tif *.tiff *.webp *.nef *.nrw *.NEF"), ("Wszystkie pliki", "*.*")]

class TransferWindow(ctk.CTkToplevel):
    def __init__(self, master, orig=None, ai=None):
        super().__init__(master, fg_color=T.BG); self.title("Przenieś grading z AI")
        self.geometry(f"1360x860+{master.winfo_rootx()+40}+{master.winfo_rooty()+30}")
        self.minsize(1100, 700)
        self.orig_p = self.ai_p = None; self.orig = self.meta = self.prev = None; self.ai = None
        self.prep = None; self.result = None; self.ai_view = None; self.mask = None
        self.params = TR.TransferParams(); self._blink_on = False; self._split = 0.5; self._busy = 0
        self._jobs = queue.Queue(); self._ui = queue.Queue(); self._gen = 0
        threading.Thread(target=self._worker, daemon=True).start()
        self._build(); self.after(30, self._poll)
        self.bind("<Control-s>", lambda e: self.save("jpg"))
        if orig: self.after(50, lambda: self.load_orig(Path(orig)))
        if ai: self.after(80, lambda: self.load_ai(Path(ai)))

    # ---------------------------------------------------------------- layout
    def _build(self):
        top = ctk.CTkFrame(self, fg_color=T.BG); top.pack(fill="x", padx=14, pady=(12, 8))
        T.button(top, "Oryginał…", self.pick_orig, width=110).pack(side="left")
        self.orig_lbl = self._name_box(top); T.button(top, "Wersja AI…", self.pick_ai, width=110).pack(side="left", padx=(12, 0))
        self.ai_lbl = self._name_box(top)
        T.button(top, "Seria…", self.batch, width=90).pack(side="left", padx=(12, 0))
        T.button(top, "Zapisz TIFF 16-bit", lambda: self.save("tif"), width=150).pack(side="right")
        T.button(top, "Zapisz JPG", lambda: self.save("jpg"), accent=True, width=120).pack(side="right", padx=8)

        body = ctk.CTkFrame(self, fg_color=T.BG); body.pack(fill="both", expand=True, padx=14)
        side = ctk.CTkFrame(body, fg_color=T.CARD, width=300, corner_radius=10); side.pack(side="right", fill="y", padx=(12, 0)); side.pack_propagate(False)
        left = ctk.CTkFrame(body, fg_color=T.BG); left.pack(side="left", fill="both", expand=True)
        self.mode = T.segmented(left, MODES, command=lambda v: self._mode_changed()); self.mode.set("Wynik"); self.mode.pack(anchor="w", pady=(0, 8))
        self.view = ImageView(left, "Wybierz oryginał (pełna rozdzielczość, JPG/TIFF/NEF)\ni wersję z AI (np. z Gemini).\n\n"
                                    "Program sam nałoży obie wersje, przeniesie kolor i światło z AI,\na szczegóły i ostrość zostawi z oryginału.")
        self.view.pack(fill="both", expand=True)
        self.view.on_redraw = self._overlay
        self.view.bind("<ButtonPress-1>", self._press); self.view.bind("<B1-Motion>", self._drag); self.view.bind("<ButtonRelease-1>", self._release)
        self.view.bind("<Motion>", self._cursor)

        pad = dict(padx=16, anchor="w")
        T.label(side, "Przeniesienie z AI", 15, weight="bold").pack(pady=(16, 8), **pad)
        self.sliders = {}
        for key, name in (("kolor", "Kolor"), ("jasnosc", "Jasność (światło)")):
            row = ctk.CTkFrame(side, fg_color=T.CARD); row.pack(fill="x", padx=16, pady=(6, 0))
            T.label(row, name, 13).pack(side="left")
            vb = ctk.CTkFrame(row, fg_color=T.CARD, width=48, height=22); vb.pack(side="right"); vb.pack_propagate(False)
            val = T.label(vb, "100%", 12, T.MUTED, anchor="e"); val.pack(fill="both", expand=True)
            s = T.slider(side, 0, 100, command=lambda v, k=key: self._slider(k, v), number_of_steps=100, width=268); s.set(100); s.pack(padx=16, pady=(4, 2))
            self.sliders[key] = (s, val)
        T.label(side, "Kolor = barwy z AI, Jasność = nowe światło z AI. Szczegóły i ostrość zawsze z oryginału.",
                11, T.MUTED, justify="left", wraplength=264).pack(pady=(2, 10), **pad)
        sw = lambda text, cmd, on: self._switch(side, text, cmd, on)
        self.protect_sw = sw("Chroń miejsca zmienione przez AI", self._protect, True)
        self.changes_sw = sw("Pokaż mapę zmian", lambda: self.redraw(), False)

        T.label(side, "Pędzel jasności", 13, weight="bold").pack(pady=(14, 4), **pad)
        self.brush_sw = sw("Maluj maskę (B)", self._brush_toggle, False)
        self.brush_mode = T.segmented(side, ["Zmniejsz", "Przywróć"]); self.brush_mode.set("Zmniejsz"); self.brush_mode.pack(padx=16, pady=(4, 6), anchor="w")
        row = ctk.CTkFrame(side, fg_color=T.CARD); row.pack(fill="x", padx=16)
        T.label(row, "Rozmiar", 12).pack(side="left")
        bb = ctk.CTkFrame(row, fg_color=T.CARD, width=48, height=22); bb.pack(side="right"); bb.pack_propagate(False)
        self.brush_val = T.label(bb, "6%", 12, T.MUTED, anchor="e"); self.brush_val.pack(fill="both", expand=True)
        self.brush_size = T.slider(side, 1, 20, command=lambda v: self.brush_val.configure(text=f"{int(v)}%"), number_of_steps=19, width=268)
        self.brush_size.set(6); self.brush_size.pack(padx=16, pady=(4, 4))
        b = T.button(side, "Wyczyść maskę", self.clear_mask, width=268); b.configure(height=28); b.pack(padx=16, pady=(2, 10))

        T.label(side, "Dopasowanie", 13, weight="bold").pack(pady=(8, 2), **pad)
        self.info = T.label(side, "—", 12, T.MUTED, justify="left", anchor="w", width=268, height=96); self.info.pack(padx=16, anchor="w")
        self.warn = T.label(side, "", 12, T.ACCENT, justify="left", anchor="w", width=268, height=54, wraplength=264); self.warn.pack(padx=16, anchor="w")
        self.status = T.label(self, "", 12, T.MUTED, anchor="w"); self.status.pack(fill="x", padx=16, pady=8)
        self.bind("<KeyPress-b>", lambda e: self._key_brush()); self.bind("<KeyPress-B>", lambda e: self._key_brush())

    def _name_box(self, master):
        f = ctk.CTkFrame(master, fg_color=T.BG, width=230, height=30); f.pack(side="left", padx=(8, 0)); f.pack_propagate(False)
        l = T.label(f, "—", 12, T.MUTED, anchor="w"); l.pack(fill="both", expand=True); return l

    def _switch(self, master, text, cmd, on):
        s = ctk.CTkSwitch(master, text=text, font=T.font(12), progress_color=T.ACCENT, switch_width=30, switch_height=14, command=cmd)
        if on: s.select()
        s.pack(padx=16, pady=3, anchor="w"); return s

    # ---------------------------------------------------------------- threading
    def _worker(self):
        while True:
            fn, done, gen = self._jobs.get()
            if gen is not None and gen != self._gen:
                self._ui.put((None, None)); continue
            try: self._ui.put((done, fn()))
            except Exception as e:  # noqa: BLE001
                print(traceback.format_exc(), file=sys.stderr); self._ui.put((self._fail, e))

    def submit(self, fn, done, render=False):
        if render: self._gen += 1
        self._busy += 1; self._jobs.put((fn, done, self._gen if render else None))

    def _poll(self):
        try:
            while True:
                done, res = self._ui.get_nowait(); self._busy = max(0, self._busy-1)
                if done: done(res)
        except queue.Empty: pass
        if self.mode.get() == "Miganie" and self.result is not None:
            t = (self.tk.call("clock", "milliseconds") // 600) % 2 == 1
            if t != self._blink_on: self._blink_on = t; self.redraw()
        if self.winfo_exists(): self.after(30, self._poll)

    def _fail(self, e):
        msg = str(e) if isinstance(e, (AlignmentError, ValueError)) else f"Błąd: {e}"
        self.warn.configure(text=msg); self.status.configure(text=msg)

    # ---------------------------------------------------------------- loading
    def pick_orig(self):
        p = filedialog.askopenfilename(parent=self, title="Oryginał (pełna rozdzielczość)", filetypes=IMG_TYPES)
        if p: self.load_orig(Path(p))

    def pick_ai(self):
        p = filedialog.askopenfilename(parent=self, title="Wersja z AI", filetypes=IMG_TYPES)
        if p: self.load_ai(Path(p))

    def load_orig(self, p: Path):
        self.orig_p = p; self.orig_lbl.configure(text=p.name); self.status.configure(text=f"Wczytuję {p.name}…")
        self.prep = self.result = None; self.mask = None
        def work():
            o, meta = GR.load_original(p, bits16=True)
            h, w = o.shape[:2]; s = min(1.0, PREVIEW/max(h, w))
            prev = cv2.resize(o, (max(1, round(w*s)), max(1, round(h*s))), interpolation=cv2.INTER_AREA)
            return o, meta, prev
        def done(r):
            self.orig, self.meta, self.prev = r
            self.view.set_image(self._to8(self.prev), self.meta.get("icc_profile"), (self.orig.shape[1], self.orig.shape[0]), keep_view=False)
            self.status.configure(text=f"Oryginał {self.orig.shape[1]}×{self.orig.shape[0]} px" + ("" if self.ai is not None else " · wybierz wersję AI"))
            if self.ai_p is not None: self.load_ai(self.ai_p)
        self.submit(work, done)

    def load_ai(self, p: Path):
        self.ai_p = p; self.ai_lbl.configure(text=p.name)
        if self.orig is None: return
        self.status.configure(text="Nakładam wersję AI na oryginał…"); self.warn.configure(text="")
        icc = self.meta.get("icc_profile"); orig = self.orig; protect = self.params.protect
        def work():
            ai = GR.load_ai(p, icc)
            return ai, TR.prepare(orig, ai, protect=protect)
        def done(r):
            self.ai, self.prep = r; self._make_ai_view(); self._info(); self.render()
        self.submit(work, done)

    @staticmethod
    def _to8(a):
        return (a >> 8).astype(np.uint8) if a.dtype == np.uint16 else a

    def _make_ai_view(self):
        """AI version resampled onto the preview of the original (to check the alignment)."""
        al = self.prep.al; ph, pw = self.prev.shape[:2]; W, H = self.prep.full_wh
        yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
        fx, fy = (xx+0.5)*W/pw - 0.5, (yy+0.5)*H/ph - 0.5
        P = al.full_to_ai(np.c_[fx.ravel(), fy.ravel()]).astype(np.float32)
        v = cv2.remap(self.ai, P[:, 0].reshape(ph, pw), P[:, 1].reshape(ph, pw), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        self.ai_view = v

    def _info(self):
        p = self.prep; al = p.al
        self.info.configure(text=f"Błąd dopasowania: {al.residual_px:.2f} px\nSkala AI: 1 : {1/al.scale:.2f}\n"
                                 f"Pokrycie oryginału: {p.coverage_frac*100:.0f}%\nZmienione przez AI: {p.changed_frac*100:.1f}%"
                                 + ("\nKorekta lokalnych przesunięć: tak" if al.flow is not None else ""))
        w = []
        if al.residual_px > 1.0: w.append("Wersja AI nie pokrywa się dokładnie – sprawdź tryb Miganie.")
        if p.coverage_frac < 0.9: w.append("AI przycięło kadr – brzegi dostają ogólny grading.")
        if p.changed_frac > 0.05: w.append("AI zmieniło treść w części zdjęcia (mapa zmian) – tam zostaje oryginał.")
        self.warn.configure(text="\n".join(w))

    # ---------------------------------------------------------------- rendering
    def render(self):
        if self.prep is None: return
        prep, prev, prm, mask = self.prep, self.prev, TR.TransferParams(**vars(self.params)), None if self.mask is None else self.mask.copy()
        def done(r):
            self.result = r; self.redraw()
            self.status.configure(text=f"Oryginał {self.orig.shape[1]}×{self.orig.shape[0]} px · podgląd {prev.shape[1]}×{prev.shape[0]} · zapis w pełnej rozdzielczości")
        self.submit(lambda: TR.render(prep, prev, prm, mask, out_dtype=np.uint8), done, render=True)

    def _current(self):
        m = self.mode.get(); base = self._to8(self.prev)
        if self.result is None: return base
        if m == "Oryginał": return base
        if m == "AI": return self.ai_view
        if m == "Miganie": return self.ai_view if self._blink_on else self.result
        if m == "Podział":
            x = int(self._split*base.shape[1]); out = self.result.copy(); out[:, :x] = base[:, :x]; return out
        return self.result

    def redraw(self):
        if self.prev is None: return
        img = self._current()
        if self.changes_sw.get() and self.prep is not None and self.mode.get() in ("Wynik", "Podział"):
            ch = cv2.resize(self.prep.changed, (img.shape[1], img.shape[0]))[..., None]
            img = (img*(1-0.55*ch) + np.array([245, 60, 60])*0.55*ch).astype(np.uint8)
        if self.mask is not None and self.brush_sw.get():
            m = cv2.resize(self.mask, (img.shape[1], img.shape[0]))[..., None]
            img = (img*(0.45+0.55*m) + np.array([60, 140, 255])*0.55*(1-m)).astype(np.uint8)
        self.view.set_image(img, self.meta.get("icc_profile"), (self.orig.shape[1], self.orig.shape[0]))

    def _overlay(self):
        self.view.delete("ov")
        if self.mode.get() == "Podział" and self.result is not None:
            x0, y0, dw, dh = self.view.geom; x = x0 + self._split*dw
            self.view.create_line(x, y0, x, y0+dh, fill=T.ACCENT, width=2, tags="ov")
            for tx, t, anc in ((x-8, "oryginał", "ne"), (x+8, "wynik", "nw")):
                self.view.create_text(tx, y0+10, text=t, fill="white", anchor=anc, font=(T.FAMILY, 11), tags="ov")

    def _mode_changed(self):
        self._blink_on = False; self.redraw()

    # ---------------------------------------------------------------- controls
    def _slider(self, key, v):
        setattr(self.params, key, float(v)/100); self.sliders[key][1].configure(text=f"{int(round(v))}%"); self.render()

    def _protect(self):
        self.params.protect = bool(self.protect_sw.get()); self.render()

    def _brush_toggle(self):
        if self.brush_sw.get() and self.mask is None and self.prev is not None:
            self.mask = np.ones((max(1, self.prev.shape[0]//2), max(1, self.prev.shape[1]//2)), np.float32)
        self.redraw()

    def _key_brush(self):
        if self.brush_sw.get(): self.brush_sw.deselect()
        else: self.brush_sw.select()
        self._brush_toggle()

    def clear_mask(self):
        self.mask = None; self.brush_sw.deselect(); self.redraw(); self.render()

    def _paint(self, e):
        u, v = self.view.to_norm(e.x, e.y)
        if self.mask is None or not (0 <= u <= 1 and 0 <= v <= 1): return
        mh, mw = self.mask.shape; r = self.brush_size.get()/100*max(mh, mw)
        yy, xx = np.ogrid[0:mh, 0:mw]
        d = np.sqrt((xx - u*mw)**2 + (yy - v*mh)**2)
        a = np.clip(1 - d/r, 0, 1)**2 * 0.35                  # soft round brush
        tgt = 0.0 if self.brush_mode.get() == "Zmniejsz" else 1.0
        self.mask += (tgt - self.mask)*a

    def _press(self, e):
        if self.mode.get() == "Podział": self._drag(e); return
        if self.brush_sw.get(): self._paint(e); self.redraw()

    def _drag(self, e):
        if self.mode.get() == "Podział":
            x0, _, dw, _ = self.view.geom; self._split = float(np.clip((e.x-x0)/max(dw, 1), 0, 1)); self.redraw(); return
        if self.brush_sw.get(): self._paint(e); self.redraw()

    def _release(self, e):
        if self.brush_sw.get() and self.mode.get() != "Podział": self.render()

    def _cursor(self, e):
        self.view.delete("cur")
        if self.brush_sw.get() and self.prev is not None:
            r = self.brush_size.get()/100*self.view.geom[2]*max(1, self.prev.shape[0]/self.prev.shape[1]) if self.prev.shape[0] > self.prev.shape[1] else self.brush_size.get()/100*self.view.geom[2]
            self.view.create_oval(e.x-r, e.y-r, e.x+r, e.y+r, outline="white", tags="cur")

    # ---------------------------------------------------------------- output
    def save(self, fmt):
        if self.prep is None: self.status.configure(text="Najpierw wybierz oryginał i wersję AI."); return
        d = GR.default_out(self.orig_p, fmt)
        p = filedialog.asksaveasfilename(parent=self, title="Zapisz wynik", initialdir=str(d.parent), initialfile=d.name,
                                         defaultextension="." + fmt, filetypes=[("TIFF 16-bit", "*.tif")] if fmt == "tif" else [("JPG", "*.jpg")])
        if p: self.save_to(Path(p))

    def save_to(self, path: Path):
        GR._guard(path, self.orig_p, self.ai_p)
        prep, orig, meta, prm = self.prep, self.orig, self.meta, TR.TransferParams(**vars(self.params))
        mask = None if self.mask is None else self.mask.copy()
        self.status.configure(text=f"Zapisuję {path.name} ({orig.shape[1]}×{orig.shape[0]})…")
        self.submit(lambda: GR.save(path, prep, orig, meta, prm, mask), lambda r: self.status.configure(text=f"Zapisano {path}"))

    def batch(self):
        a = filedialog.askdirectory(parent=self, title="Folder z oryginałami")
        if not a: return
        b = filedialog.askdirectory(parent=self, title="Folder z wersjami AI (może być ten sam)", initialdir=a)
        if not b: return
        self.run_batch(Path(a), Path(b), Path(a) / "Retusz")

    def run_batch(self, a, b, out, fmt="jpg"):
        pairs = GR.pair_files(a, b)
        if not pairs: self.status.configure(text="Nie znaleziono par: wersja AI musi mieć nazwę oryginału (np. IMG_1_ai.png)."); return
        prm = TR.TransferParams(**vars(self.params))
        def prog(i, n, o, r, err):
            self._ui.put((lambda _: self.status.configure(text=f"Seria {i}/{n}: {Path(o).name} " + ("OK" if r else f"BŁĄD {err}")), None)); self._busy += 1
        def done(res):
            bad = [o.name for o, _, e in res if e]
            self.status.configure(text=f"Seria: zapisano {len(res)-len(bad)} z {len(res)} w {out}" + (f" · błędy: {', '.join(bad)}" if bad else ""))
        self.submit(lambda: GR.batch(pairs, out, prm, fmt, progress=prog), done)
