"""Okno „Przenieś grading z AI”: oryginał w pełnej rozdzielczości + wersja z narzędzia AI (mniejsza,
przesunięta, przycięta) -> kolor i światło z AI, szczegóły i ostrość z oryginału.

Widok Obok siebie (Oryginał/AI | Wynik) z zsynchronizowanym przybliżeniem (do 400%, pełna rozdzielczość 1:1),
Podział, Wynik, Miganie, natychmiastowe suwaki z cache Lab, pasek postępu i pędzel jasności z Cofnij."""
from __future__ import annotations
import queue, threading, traceback, sys
from pathlib import Path
from tkinter import filedialog
import numpy as np, cv2
import customtkinter as ctk
from . import theme as T
from .imageview import ImageView
from .. import grading as GR
from ..engine import transfer as TR
from ..engine.align import AlignmentError

PREVIEW = 1600
MODES = ["Obok siebie", "Podział", "Wynik", "Miganie", "Oryginał", "AI"]
CMP_SRCS = ["Oryginał", "AI"]
IMG_TYPES = [("Zdjęcia", "*.jpg *.jpeg *.png *.tif *.tiff *.webp *.nef *.nrw *.NEF"), ("Wszystkie pliki", "*.*")]
_STAGES = {
    "rejestracja": (0.18, "Dopasowuję kadr (SIFT + ECC)…"),
    "dopasowanie": (0.45, "Wyrównuję obraz AI…"),
    "mapa_zmian": (0.62, "Wykrywam miejsca zmienione przez AI…"),
    "siatka": (0.82, "Buduję lokalną mapę koloru i światła…"),
    "gotowe": (1.0, "Gotowe"),
}


class TransferWindow(ctk.CTkToplevel):
    def __init__(self, master, orig=None, ai=None, on_open=None):
        super().__init__(master, fg_color=T.BG); self.title("Przenieś grading z AI")
        self.geometry(f"1440x880+{master.winfo_rootx()+30}+{master.winfo_rooty()+20}")
        self.minsize(1160, 720)
        self.on_open = on_open
        self.orig_p = self.ai_p = None
        self.orig = self.orig_8 = self.meta = self.prev = None; self.ai = None
        self.prep = None; self.cache = None; self.result = None; self.ai_view = None
        self.detail_res = None; self.detail_key = None; self._detail_after = None
        self.mask = None; self._mask_undo = []; self._stroke_snap = None
        self.params = TR.TransferParams()
        self._blink_on = False; self._split = 0.5; self._before_on = False
        self._pan = None; self._space = False; self._syncing_view = False
        self._busy = 0; self._jobs = queue.Queue(); self._ui = queue.Queue()
        self._gen = 0; self._detail_gen = 0
        threading.Thread(target=self._worker, daemon=True).start()
        self._build(); self.after(30, self._poll)
        self._bind_keys()
        if orig: self.after(50, lambda: self.load_orig(Path(orig)))
        if ai: self.after(80, lambda: self.load_ai(Path(ai)))

    # ---------------------------------------------------------------- layout
    def _build(self):
        top = ctk.CTkFrame(self, fg_color=T.BG); top.pack(fill="x", padx=14, pady=(12, 8))
        T.button(top, "Oryginał…", self.pick_orig, width=102).pack(side="left")
        self.orig_lbl = self._name_box(top, 185)
        T.button(top, "Wersja AI…", self.pick_ai, width=102).pack(side="left", padx=(10, 0))
        self.ai_lbl = self._name_box(top, 185)
        T.button(top, "Seria…", self.batch, width=82).pack(side="left", padx=(10, 0))
        T.button(top, "Zapisz TIFF 16-bit", lambda: self.save("tif"), width=138).pack(side="right")
        if self.on_open is not None:
            T.button(top, "Zapisz i edytuj", self.save_and_edit, width=124).pack(side="right", padx=(0, 8))
        T.button(top, "Zapisz JPG", lambda: self.save("jpg"), accent=True, width=110).pack(side="right", padx=8)

        body = ctk.CTkFrame(self, fg_color=T.BG); body.pack(fill="both", expand=True, padx=14)
        side = ctk.CTkFrame(body, fg_color=T.CARD, width=310, corner_radius=10, border_width=1, border_color=T.BORDER)
        side.pack(side="right", fill="y", padx=(12, 0)); side.pack_propagate(False)

        left = ctk.CTkFrame(body, fg_color=T.BG); left.pack(side="left", fill="both", expand=True)
        bar = ctk.CTkFrame(left, fg_color=T.BG); bar.pack(fill="x", pady=(0, 6))
        self.mode = T.segmented(bar, MODES, command=lambda v: self._mode_changed())
        self.mode.set("Obok siebie"); self.mode.pack(side="left")
        T.label(bar, "Porównaj z:", 12, T.MUTED).pack(side="left", padx=(12, 4))
        self.cmp_src = T.segmented(bar, CMP_SRCS, command=lambda v: self._cmp_changed())
        self.cmp_src.set("Oryginał"); self.cmp_src.pack(side="left")

        zb = ctk.CTkFrame(bar, fg_color=T.BG); zb.pack(side="right")
        zbox = ctk.CTkFrame(zb, fg_color=T.BG, width=48, height=28); zbox.pack(side="right"); zbox.pack_propagate(False)
        self.zoom_lbl = T.label(zbox, "", 12, T.MUTED, anchor="e"); self.zoom_lbl.pack(fill="both", expand=True)
        self.before_btn = T.button(zb, "Przed/Po", self.toggle_before, width=78)
        self.before_btn.configure(height=28, font=T.font(12)); self.before_btn.pack(side="right", padx=(0, 6))
        for txt, cmd, w in (("200%", lambda: self.set_zoom(2.0), 50),
                            ("100%", lambda: self.set_zoom(1.0), 50),
                            ("Dopasuj", self.zoom_fit, 68)):
            b = T.button(zb, txt, cmd, width=w); b.configure(height=28, font=T.font(12)); b.pack(side="right", padx=(0, 4))

        self.views = ctk.CTkFrame(left, fg_color=T.BG); self.views.pack(fill="both", expand=True)
        self.views.grid_rowconfigure(1, weight=1)
        self.hdr_cmp = T.label(self.views, "Oryginał (przed)", 12, T.MUTED, anchor="w")
        self.hdr_main = T.label(self.views, "Wynik (po)", 12, T.TEXT, anchor="w")
        self.view_cmp = ImageView(self.views, "Oryginał (przed)")
        self.view = ImageView(self.views, "Wybierz oryginał (pełna rozdzielczość, JPG/TIFF/NEF)\ni wersję z AI (np. z Gemini).\n\n"
                                          "Program nałoży obie wersje, przeniesie kolor i światło z AI,\na szczegóły i ostrość zostawi z oryginału.")
        self.view.on_redraw = self._overlay
        for v in (self.view_cmp, self.view):
            v.on_view_change = self._view_changed
            v.bind("<MouseWheel>", lambda e, v=v: self._wheel(v, e, 1 if e.delta > 0 else -1))
            v.bind("<Button-4>", lambda e, v=v: self._wheel(v, e, 1))
            v.bind("<Button-5>", lambda e, v=v: self._wheel(v, e, -1))
            v.bind("<ButtonPress-2>", lambda e, v=v: self._pan_start(v, e))
            v.bind("<B2-Motion>", lambda e, v=v: self._pan_move(v, e))
            v.bind("<ButtonRelease-2>", lambda e: self._pan_end())
        self.view_cmp.bind("<ButtonPress-1>", lambda e: self._pan_start(self.view_cmp, e))
        self.view_cmp.bind("<B1-Motion>", lambda e: self._pan_move(self.view_cmp, e))
        self.view_cmp.bind("<ButtonRelease-1>", lambda e: self._pan_end())
        self.view.bind("<ButtonPress-1>", self._press); self.view.bind("<B1-Motion>", self._drag)
        self.view.bind("<ButtonRelease-1>", self._release); self.view.bind("<Motion>", self._cursor)
        self.view.bind("<Leave>", lambda e: self.view.delete("cur"))
        self._apply_pane_layout()

        pad = dict(padx=16, anchor="w")
        T.label(side, "Przeniesienie z AI", 15, weight="bold").pack(pady=(14, 6), **pad)
        self.sliders = {}
        for key, name in (("kolor", "Kolor"), ("jasnosc", "Jasność (światło)")):
            row = ctk.CTkFrame(side, fg_color=T.CARD); row.pack(fill="x", padx=16, pady=(6, 0))
            lbl = T.label(row, name, 13); lbl.pack(side="left")
            vb = ctk.CTkFrame(row, fg_color=T.CARD, width=48, height=22); vb.pack(side="right"); vb.pack_propagate(False)
            val = T.label(vb, "100%", 12, T.MUTED, anchor="e"); val.pack(fill="both", expand=True)
            s = T.slider(side, 0, 100, command=lambda v, k=key: self._slider(k, v), number_of_steps=100, width=278)
            s.set(100); s.pack(padx=16, pady=(4, 2))
            for w in (lbl, val, s): w.bind("<Double-Button-1>", lambda e, k=key: self._reset_slider(k))
            self.sliders[key] = (s, val)
        T.label(side, "Kolor = barwy z AI, Jasność = światło z AI. Szczegóły i ostrość zawsze z oryginału. Dwuklik resetuje.",
                11, T.MUTED, justify="left", wraplength=274).pack(pady=(2, 8), **pad)
        sw = lambda text, cmd, on: self._switch(side, text, cmd, on)
        self.protect_sw = sw("Chroń miejsca zmienione przez AI", self._protect, True)
        self.changes_sw = sw("Pokaż mapę zmian", lambda: self.redraw(), False)

        T.label(side, "Pędzel jasności", 13, weight="bold").pack(pady=(12, 4), **pad)
        self.brush_sw = sw("Maluj maskę (B)", self._brush_toggle, False)
        self.brush_mode = T.segmented(side, ["Zmniejsz", "Przywróć"]); self.brush_mode.set("Zmniejsz")
        self.brush_mode.pack(padx=16, pady=(4, 6), anchor="w")
        row = ctk.CTkFrame(side, fg_color=T.CARD); row.pack(fill="x", padx=16)
        blbl = T.label(row, "Rozmiar ([ / ])", 12); blbl.pack(side="left")
        bb = ctk.CTkFrame(row, fg_color=T.CARD, width=48, height=22); bb.pack(side="right"); bb.pack_propagate(False)
        self.brush_val = T.label(bb, "6%", 12, T.MUTED, anchor="e"); self.brush_val.pack(fill="both", expand=True)
        self.brush_size = T.slider(side, 1, 20, command=lambda v: self._set_brush_val(v), number_of_steps=19, width=278)
        self.brush_size.set(6); self.brush_size.pack(padx=16, pady=(4, 6))
        for w in (blbl, self.brush_val, self.brush_size):
            w.bind("<Double-Button-1>", lambda e: (self.brush_size.set(6), self._set_brush_val(6)))
        brow = ctk.CTkFrame(side, fg_color=T.CARD); brow.pack(fill="x", padx=16, pady=(0, 8))
        self.undo_btn = T.button(brow, "Cofnij", self.undo_mask, width=128)
        self.undo_btn.configure(height=28, font=T.font(12), state="disabled"); self.undo_btn.pack(side="left")
        b = T.button(brow, "Wyczyść maskę", self.clear_mask, width=142)
        b.configure(height=28, font=T.font(12)); b.pack(side="right")

        T.label(side, "Dopasowanie", 13, weight="bold").pack(pady=(6, 4), **pad)
        self.prog = ctk.CTkProgressBar(side, width=278, height=6, progress_color=T.PANEL, fg_color=T.PANEL)
        self.prog.set(0); self.prog.pack(padx=16, pady=(0, 6), anchor="w")
        self.info = T.label(side, "—", 12, T.MUTED, justify="left", anchor="w", width=278, height=96)
        self.info.pack(padx=16, anchor="w")
        self.warn = T.label(side, "", 12, T.ACCENT, justify="left", anchor="w", width=278, height=54, wraplength=274)
        self.warn.pack(padx=16, anchor="w")
        self.status = T.label(self, "", 12, T.MUTED, anchor="w"); self.status.pack(fill="x", padx=16, pady=8)

    def _bind_keys(self):
        self.bind("<Control-s>", lambda e: self.save("jpg"))
        self.bind("<Control-z>", lambda e: self.undo_mask())
        self.bind("<Control-0>", lambda e: self.zoom_fit())
        self.bind("<Control-1>", lambda e: self.set_zoom(1.0))
        self.bind("<Control-2>", lambda e: self.set_zoom(2.0))
        self.bind("<backslash>", lambda e: self.toggle_before())
        self.bind("<KeyPress-b>", lambda e: self._key_brush()); self.bind("<KeyPress-B>", lambda e: self._key_brush())
        self.bind("<bracketleft>", lambda e: self._brush_step(-1))
        self.bind("<bracketright>", lambda e: self._brush_step(1))
        self.bind("<KeyPress-space>", lambda e: self._space_key(True))
        self.bind("<KeyRelease-space>", lambda e: self._space_key(False))
        for i, m in enumerate(MODES[:4], 1):
            self.bind(f"<KeyPress-{i}>", lambda e, m=m: (self.mode.set(m), self._mode_changed()))

    def _name_box(self, master, w=200):
        f = ctk.CTkFrame(master, fg_color=T.BG, width=w, height=30); f.pack(side="left", padx=(6, 0)); f.pack_propagate(False)
        l = T.label(f, "—", 12, T.MUTED, anchor="w"); l.pack(fill="both", expand=True); return l

    def _switch(self, master, text, cmd, on):
        s = ctk.CTkSwitch(master, text=text, font=T.font(12), progress_color=T.ACCENT, switch_width=30, switch_height=14, command=cmd)
        if on: s.select()
        s.pack(padx=16, pady=3, anchor="w"); return s

    def _apply_pane_layout(self):
        dual = self.mode.get() == "Obok siebie"
        if dual:
            self.views.grid_columnconfigure(0, weight=1, uniform="tv")
            self.views.grid_columnconfigure(1, weight=1, uniform="tv")
            self.hdr_cmp.grid(row=0, column=0, sticky="w", padx=4, pady=(0, 3))
            self.view_cmp.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
            self.hdr_main.grid(row=0, column=1, sticky="w", padx=4, pady=(0, 3))
            self.view.grid(row=1, column=1, sticky="nsew", padx=(4, 0))
        else:
            self.hdr_cmp.grid_remove(); self.view_cmp.grid_remove()
            self.views.grid_columnconfigure(0, weight=0, uniform="")
            self.views.grid_columnconfigure(1, weight=1, uniform="")
            self.hdr_main.grid(row=0, column=1, sticky="w", padx=4, pady=(0, 3))
            self.view.grid(row=1, column=1, sticky="nsew", padx=0)
        self._update_headers()

    def _update_headers(self):
        m = self.mode.get(); cmp_lbl = "Oryginał (przed)" if self.cmp_src.get() == "Oryginał" else "Wersja AI (nałożona)"
        self.hdr_cmp.configure(text=cmp_lbl)
        if self._before_on and m in ("Wynik", "Obok siebie"):
            self.hdr_main.configure(text=f"Podgląd: {cmp_lbl}  (\\ wraca do wyniku)", text_color=T.ACCENT)
        elif m == "Obok siebie":
            self.hdr_main.configure(text="Wynik (po)", text_color=T.TEXT)
        elif m == "Podział":
            self.hdr_main.configure(text=f"Podział: {self.cmp_src.get()}  |  Wynik  (przeciągnij linię)", text_color=T.TEXT)
        elif m == "Miganie":
            self.hdr_main.configure(text=f"Miganie: Wynik ↔ {self.cmp_src.get()}", text_color=T.TEXT)
        else:
            self.hdr_main.configure(text=m, text_color=T.TEXT)

    # ---------------------------------------------------------------- threading
    def _worker(self):
        while True:
            fn, done, kind, gen = self._jobs.get()
            if kind == "render" and gen != self._gen:
                self._ui.put((None, None)); continue
            if kind == "detail" and gen != self._detail_gen:
                self._ui.put((None, None)); continue
            try: self._ui.put((done, fn()))
            except Exception as e:  # noqa: BLE001
                print(traceback.format_exc(), file=sys.stderr); self._ui.put((self._fail, e))

    def submit(self, fn, done, render=False, kind=None):
        if render: kind = "render"
        gen = None
        if kind == "render":
            self._gen += 1; gen = self._gen
        elif kind == "detail":
            self._detail_gen += 1; gen = self._detail_gen
        self._busy += 1; self._jobs.put((fn, done, kind, gen))

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

    def _set_prog(self, frac):
        self.prog.configure(progress_color=T.ACCENT if frac > 0 else T.PANEL)
        self.prog.set(max(0.0, min(1.0, float(frac))))

    def _fail(self, e):
        self._set_prog(0)
        msg = str(e) if isinstance(e, (AlignmentError, ValueError)) else f"Błąd: {e}"
        self.warn.configure(text=msg); self.status.configure(text=msg)

    def _post_progress(self, frac, text):
        self._busy += 1
        self._ui.put((lambda _: (self._set_prog(frac), self.status.configure(text=text)), None))

    # ---------------------------------------------------------------- loading
    def pick_orig(self):
        p = filedialog.askopenfilename(parent=self, title="Oryginał (pełna rozdzielczość)", filetypes=IMG_TYPES)
        if p: self.load_orig(Path(p))

    def pick_ai(self):
        p = filedialog.askopenfilename(parent=self, title="Wersja z AI", filetypes=IMG_TYPES)
        if p: self.load_ai(Path(p))

    def load_orig(self, p: Path):
        self.orig_p = p; self.orig_lbl.configure(text=p.name); self.status.configure(text=f"Wczytuję {p.name}…")
        self.prep = self.cache = self.result = self.detail_res = self.detail_key = None
        self.mask = None; self._mask_undo.clear(); self._sync_undo_btn()
        self._set_prog(0.1)
        def work():
            o, meta = GR.load_original(p, bits16=True)
            o8 = self._to8(o)
            h, w = o.shape[:2]; s = min(1.0, PREVIEW/max(h, w))
            prev = cv2.resize(o, (max(1, round(w*s)), max(1, round(h*s))), interpolation=cv2.INTER_AREA)
            return o, o8, meta, prev
        def done(r):
            self.orig, self.orig_8, self.meta, self.prev = r
            self._set_prog(0)
            p8 = self._to8(self.prev); fwh = (self.orig.shape[1], self.orig.shape[0])
            self.view_cmp.set_image(p8, self.meta.get("icc_profile"), fwh, keep_view=False)
            self.view_cmp.set_detail(self.orig_8)
            self.view.set_image(p8, self.meta.get("icc_profile"), fwh, keep_view=False)
            self._sync_zoom_label()
            self.status.configure(text=f"Oryginał {fwh[0]}×{fwh[1]} px" + ("" if self.ai is not None else " · wybierz wersję AI"))
            if self.ai_p is not None: self.load_ai(self.ai_p)
        self.submit(work, done)

    def load_ai(self, p: Path):
        self.ai_p = p; self.ai_lbl.configure(text=p.name)
        if self.orig is None: return
        self.status.configure(text="Nakładam wersję AI na oryginał…"); self.warn.configure(text="")
        self._set_prog(0.08); self.detail_res = None; self.detail_key = None
        icc = self.meta.get("icc_profile"); orig = self.orig; prev = self.prev
        def cb(stage, info=None):
            frac, txt = _STAGES.get(stage, (0.5, stage))
            self._post_progress(frac, txt)
        def work():
            ai = GR.load_ai(p, icc)
            prep = TR.prepare(orig, ai, protect=True, progress=cb)
            cache = TR.make_preview_cache(prep, prev)
            return ai, prep, cache
        def done(r):
            self.ai, self.prep, self.cache = r
            self._set_prog(0)
            self._make_ai_view(); self._info(); self.render()
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
        self.ai_view = cv2.remap(self.ai, P[:, 0].reshape(ph, pw), P[:, 1].reshape(ph, pw),
                                 cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def _info(self):
        p = self.prep; al = p.al
        self.info.configure(text=f"Błąd dopasowania: {al.residual_px:.2f} px\nSkala AI: 1 : {1/al.scale:.2f}\n"
                                 f"Pokrycie oryginału: {p.coverage_frac*100:.0f}%\nZmienione przez AI: {p.changed_frac*100:.1f}%"
                                 + ("\nKorekta lokalnych przesunięć: tak" if al.flow is not None else ""))
        w = []
        if al.residual_px > 1.0: w.append("Wersja AI nie pokrywa się dokładnie – sprawdź Miganie lub Obok siebie.")
        if p.coverage_frac < 0.9: w.append("AI przycięło kadr – brzegi dostają ogólny grading.")
        if p.changed_frac > 0.05: w.append("AI zmieniło treść w części zdjęcia (mapa zmian) – tam zostaje oryginał.")
        self.warn.configure(text="\n".join(w))

    # ---------------------------------------------------------------- rendering
    def _render_key(self):
        mk = None if self.mask is None else (float(self.mask.min()), float(self.mask.mean()), float(self.mask.sum()))
        return (self.params.key(), mk)

    def render(self):
        if self.prep is None: return
        prep, prev, cache = self.prep, self.prev, self.cache
        prm = TR.TransferParams(**vars(self.params))
        mask = None if self.mask is None else self.mask.copy()
        rkey = self._render_key()
        def work():
            if cache is not None: return TR.mix_cached(cache, prm, mask, out_dtype=np.uint8)
            return TR.render(prep, prev, prm, mask, out_dtype=np.uint8)
        def done(r):
            self.result = r
            if self.detail_key != rkey: self.detail_res = None
            self.redraw(); self._update_status(); self._schedule_detail()
        self.submit(work, done, kind="render")

    def _update_status(self, extra=None):
        if self.orig is None or self.prev is None: return
        ow, oh = self.orig.shape[1], self.orig.shape[0]
        pw, ph = self.prev.shape[1], self.prev.shape[0]
        det = " · 1:1 pełna rozdzielczość" if (self.detail_res is not None and self.view.needs_detail()) else ""
        tail = f" · {extra}" if extra else ""
        self.status.configure(text=f"Oryginał {ow}×{oh} px · podgląd {pw}×{ph}{det} · zapis w pełnej rozdzielczości{tail}")

    def _cmp_image(self):
        base = self._to8(self.prev)
        if self.cmp_src.get() == "AI" and self.ai_view is not None: return self.ai_view
        return base

    def _current(self):
        m = self.mode.get(); base = self._to8(self.prev); cmp_img = self._cmp_image()
        if self.result is None: return base
        if self._before_on and m in ("Wynik", "Obok siebie"): return cmp_img
        if m == "Oryginał": return base
        if m == "AI": return self.ai_view if self.ai_view is not None else base
        if m == "Miganie": return cmp_img if self._blink_on else self.result
        if m == "Podział":
            x = int(self._split*base.shape[1]); out = self.result.copy(); out[:, :x] = cmp_img[:, :x]; return out
        return self.result

    def _has_overlays(self):
        return bool((self.changes_sw.get() and self.prep is not None) or (self.mask is not None and self.brush_sw.get()))

    def redraw(self):
        if self.prev is None: return
        icc = self.meta.get("icc_profile"); fwh = (self.orig.shape[1], self.orig.shape[0])
        m = self.mode.get()
        if m == "Obok siebie":
            cmp_img = self._cmp_image()
            self.view_cmp.set_image(cmp_img, icc, fwh)
            self.view_cmp.set_detail(self.orig_8 if self.cmp_src.get() == "Oryginał" else None)
        img = self._current()
        if self.changes_sw.get() and self.prep is not None and m in ("Obok siebie", "Wynik", "Podział") and not self._before_on:
            ch = cv2.resize(self.prep.changed, (img.shape[1], img.shape[0]))[..., None]
            img = (img*(1-0.55*ch) + np.array([245, 60, 60])*0.55*ch).astype(np.uint8)
        if self.mask is not None and self.brush_sw.get() and m in ("Obok siebie", "Wynik", "Podział") and not self._before_on:
            mk = cv2.resize(self.mask, (img.shape[1], img.shape[0]))[..., None]
            img = (img*(0.45+0.55*mk) + np.array([60, 140, 255])*0.55*(1-mk)).astype(np.uint8)
        self.view.set_image(img, icc, fwh)
        self._apply_main_detail()
        self._sync_zoom_label()

    def _apply_main_detail(self):
        m = self.mode.get()
        if self._has_overlays() or m in ("Podział", "AI"):
            self.view.set_detail(None); return
        if m == "Oryginał" or (self._before_on and self.cmp_src.get() == "Oryginał") or (m == "Miganie" and self._blink_on and self.cmp_src.get() == "Oryginał"):
            self.view.set_detail(self.orig_8); return
        if m == "Miganie" and self._blink_on:
            self.view.set_detail(None); return
        if self._before_on:
            self.view.set_detail(None); return
        self.view.set_detail(self.detail_res)

    def _schedule_detail(self):
        if self._detail_after: self.after_cancel(self._detail_after); self._detail_after = None
        if self.prep is None or self.orig is None or not self.view.needs_detail(): return
        rkey = self._render_key()
        if self.detail_res is not None and self.detail_key == rkey:
            self._apply_main_detail(); self._update_status(); return
        self._detail_after = self.after(180, self._render_detail)

    def _render_detail(self):
        self._detail_after = None
        if self.prep is None or self.orig is None or not self.view.needs_detail(): return
        prep, orig = self.prep, self.orig
        prm = TR.TransferParams(**vars(self.params))
        mask = None if self.mask is None else self.mask.copy()
        rkey = self._render_key()
        self._update_status("liczę 1:1…")
        def done(res):
            if rkey != self._render_key(): return
            self.detail_res = res; self.detail_key = rkey
            self._apply_main_detail(); self._update_status()
        self.submit(lambda: TR.render(prep, orig, prm, mask, out_dtype=np.uint8), done, kind="detail")

    def _overlay(self):
        self.view.delete("ov")
        if self.mode.get() == "Podział" and self.result is not None:
            x0, y0, dw, dh = self.view.geom; x = x0 + self._split*dw
            self.view.create_line(x, y0, x, y0+dh, fill=T.ACCENT, width=2, tags="ov")
            left_txt = "oryginał" if self.cmp_src.get() == "Oryginał" else "wersja AI"
            for tx, t, anc in ((x-8, left_txt, "ne"), (x+8, "wynik", "nw")):
                self.view.create_text(tx, y0+10, text=t, fill="white", anchor=anc, font=(T.FAMILY, 11), tags="ov")

    def _mode_changed(self):
        self._blink_on = False; self._before_on = False; self._sync_before_btn()
        self._apply_pane_layout(); self.redraw()
        if self.mode.get() == "Obok siebie":
            self.view_cmp.set_view(self.view.zoom, self.view.center)
        self._schedule_detail()

    def _cmp_changed(self):
        self._update_headers(); self.redraw()

    def toggle_before(self):
        if self.result is None: return
        self._before_on = not self._before_on
        self._sync_before_btn(); self._update_headers(); self.redraw()

    def _sync_before_btn(self):
        self.before_btn.configure(fg_color=T.ACCENT if self._before_on else T.BTN,
                                  text_color="#111" if self._before_on else T.TEXT)

    # ---------------------------------------------------------------- zoom & pan
    def _view_changed(self, src):
        if self._syncing_view: return
        self._syncing_view = True
        try:
            other = self.view_cmp if src is self.view else self.view
            if other.rgb is not None: other.set_view(src.zoom, src.center)
            self._sync_zoom_label(); self._schedule_detail()
        finally:
            self._syncing_view = False

    def _sync_zoom_label(self):
        v = self.view
        self.zoom_lbl.configure(text="" if v.rgb is None else f"{int(round(v.cur_zoom()*100))}%")

    def zoom_fit(self):
        if self.view.rgb is None: return
        self.view.set_view(None, (0.5, 0.5), notify=True)

    def set_zoom(self, z):
        if self.view.rgb is None: return
        c = self.view.center if self.view.zoom else (0.5, 0.5)
        self.view.set_view(z, c, notify=True)

    def _wheel(self, v, e, d):
        if v.rgb is None: return
        v.zoom_at(1.25 if d > 0 else 1/1.25, e.x, e.y)

    def _pan_start(self, v, e):
        if v.zoom is not None: self._pan = (v, e.x, e.y)

    def _pan_move(self, v, e):
        if self._pan is None: return
        pv, px, py = self._pan; pv.pan(e.x - px, e.y - py); self._pan = (pv, e.x, e.y)

    def _pan_end(self):
        self._pan = None

    def _space_key(self, down):
        self._space = down

    # ---------------------------------------------------------------- controls
    def _slider(self, key, v):
        iv = int(round(v))
        setattr(self.params, key, float(v)/100)
        self.sliders[key][1].configure(text=f"{iv}%", text_color=T.TEXT if iv != 100 else T.MUTED)
        self.render()

    def _reset_slider(self, key):
        self.sliders[key][0].set(100); self._slider(key, 100)

    def _protect(self):
        self.params.protect = bool(self.protect_sw.get()); self.render()

    def _set_brush_val(self, v):
        iv = int(round(v))
        self.brush_val.configure(text=f"{iv}%", text_color=T.TEXT if iv != 6 else T.MUTED)

    def _brush_step(self, d):
        v = int(np.clip(round(self.brush_size.get()) + d, 1, 20))
        self.brush_size.set(v); self._set_brush_val(v)

    def _brush_toggle(self):
        if self.brush_sw.get() and self.mask is None and self.prev is not None:
            self.mask = np.ones((max(1, self.prev.shape[0]//2), max(1, self.prev.shape[1]//2)), np.float32)
        self.redraw()

    def _key_brush(self):
        if self.brush_sw.get(): self.brush_sw.deselect()
        else: self.brush_sw.select()
        self._brush_toggle()

    def _sync_undo_btn(self):
        self.undo_btn.configure(state="normal" if self._mask_undo else "disabled")

    def undo_mask(self):
        if not self._mask_undo: return
        self.mask = self._mask_undo.pop()
        self._sync_undo_btn(); self.redraw(); self.render()

    def clear_mask(self):
        if self.mask is not None:
            self._mask_undo.append(self.mask.copy())
            if len(self._mask_undo) > 20: self._mask_undo.pop(0)
            self._sync_undo_btn()
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
        if self._space and self.view.zoom is not None:
            self._pan_start(self.view, e); return
        if self.mode.get() == "Podział": self._drag(e); return
        if self.brush_sw.get():
            self._stroke_snap = None if self.mask is None else self.mask.copy()
            self._paint(e); self.redraw(); return
        if self.view.zoom is not None:
            self._pan_start(self.view, e)

    def _drag(self, e):
        if self._pan is not None: self._pan_move(self.view, e); return
        if self.mode.get() == "Podział":
            x0, _, dw, _ = self.view.geom; self._split = float(np.clip((e.x-x0)/max(dw, 1), 0, 1)); self.redraw(); return
        if self.brush_sw.get(): self._paint(e); self.redraw()

    def _release(self, e):
        if self._pan is not None: self._pan_end(); return
        if self.brush_sw.get() and self.mode.get() != "Podział":
            if self._stroke_snap is not None:
                self._mask_undo.append(self._stroke_snap)
                if len(self._mask_undo) > 20: self._mask_undo.pop(0)
                self._stroke_snap = None; self._sync_undo_btn()
            self.render()

    def _cursor(self, e):
        self.view.delete("cur")
        if self.brush_sw.get() and self.prev is not None:
            r = (self.brush_size.get()/100*self.view.geom[2]*max(1, self.prev.shape[0]/self.prev.shape[1])
                 if self.prev.shape[0] > self.prev.shape[1] else self.brush_size.get()/100*self.view.geom[2])
            self.view.create_oval(e.x-r, e.y-r, e.x+r, e.y+r, outline="white", tags="cur")

    # ---------------------------------------------------------------- output
    def save(self, fmt):
        if self.prep is None: self.status.configure(text="Najpierw wybierz oryginał i wersję AI."); return
        d = GR.default_out(self.orig_p, fmt)
        p = filedialog.asksaveasfilename(parent=self, title="Zapisz wynik", initialdir=str(d.parent), initialfile=d.name,
                                         defaultextension="." + fmt, filetypes=[("TIFF 16-bit", "*.tif")] if fmt == "tif" else [("JPG", "*.jpg")])
        if p: self.save_to(Path(p))

    def save_to(self, path: Path, on_done=None):
        GR._guard(path, self.orig_p, self.ai_p)
        prep, orig, meta, prm = self.prep, self.orig, self.meta, TR.TransferParams(**vars(self.params))
        mask = None if self.mask is None else self.mask.copy()
        self._set_prog(0.35)
        self.status.configure(text=f"Zapisuję {path.name} ({orig.shape[1]}×{orig.shape[0]})…")
        def done(r):
            self._set_prog(0)
            self.status.configure(text=f"Zapisano {path}")
            if on_done: on_done(path)
        self.submit(lambda: GR.save(path, prep, orig, meta, prm, mask), done)

    def save_and_edit(self):
        if self.prep is None: self.status.configure(text="Najpierw wybierz oryginał i wersję AI."); return
        dst = GR.default_out(self.orig_p, "jpg")
        self.save_to(dst, on_done=lambda p: (self.on_open(p) if self.on_open else None))

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
            frac = i / max(n, 1)
            msg = f"Seria {i}/{n}: {Path(o).name} " + ("OK" if r else f"BŁĄD {err}")
            self._post_progress(frac, msg)
        def done(res):
            self._set_prog(0)
            bad = [o.name for o, _, e in res if e]
            self.status.configure(text=f"Seria: zapisano {len(res)-len(bad)} z {len(res)} w {out}" + (f" · błędy: {', '.join(bad)}" if bad else ""))
        self.submit(lambda: GR.batch(pairs, out, prm, fmt, progress=prog), done)
