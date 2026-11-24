"""Porównaj: Podział (draggable line), Obok siebie, Przełącz (hold space); Dopasuj or 100%."""
import tkinter as tk
import numpy as np
import customtkinter as ctk
from PIL import Image, ImageTk
from . import theme as T

MODES = ["Podział", "Obok siebie", "Przełącz"]

class CompareView(ctk.CTkFrame):
    def __init__(self, master, on_back, on_save, on_full):
        super().__init__(master, fg_color=T.BG)
        self.on_full = on_full
        bar = ctk.CTkFrame(self, fg_color=T.BG); bar.pack(fill="x", padx=14, pady=(10, 8))
        self.mode = T.segmented(bar, MODES, command=lambda v: self.redraw()); self.mode.set(MODES[0])
        self.mode.pack(side="left")
        self.zoom = T.segmented(bar, ["Dopasuj", "100%"], command=self._zoom); self.zoom.set("Dopasuj")
        self.zoom.pack(side="left", padx=14)
        T.button(bar, "Zapisz", on_save, accent=True).pack(side="right")
        T.button(bar, "Wróć", on_back).pack(side="right", padx=8)
        self.canvas = tk.Canvas(self, bg=T.PANEL, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True, padx=14)
        self.hint = T.label(self, "Przeciągnij linię lub przytrzymaj spację", 12, T.MUTED); self.hint.pack(pady=8)
        self.before = self.after_ = None; self.full = None; self.split = 0.5; self.show_before = False
        self.pan = [0.5, 0.5]; self._drag = None; self._photo = None
        c = self.canvas
        c.bind("<Configure>", lambda e: self.redraw())
        c.bind("<ButtonPress-1>", self._press); c.bind("<B1-Motion>", self._motion); c.bind("<ButtonRelease-1>", self._release)

    # ----- data -----
    def set_images(self, before, after):
        self.before, self.after_ = before, after; self.full = None
        if self.zoom.get() == "100%": self.on_full()
        self.redraw()

    def set_full(self, before, after):
        self.full = (before, after); self.redraw()

    def _zoom(self, v):
        if v == "100%" and self.full is None:
            self.hint.configure(text="Liczę pełną rozdzielczość…"); self.on_full()
        self.redraw()

    def key_space(self, down):
        if self.mode.get() == "Przełącz" or not down:
            self.show_before = down and self.mode.get() == "Przełącz"; self.redraw()

    # ----- drawing -----
    def _view(self, img, w, h):
        """Region of img shown in a w x h box for the current zoom."""
        if self.zoom.get() == "100%" and self.full is not None:
            H, W = img.shape[:2]
            x0 = int(np.clip(self.pan[0]*W - w/2, 0, max(0, W-w))); y0 = int(np.clip(self.pan[1]*H - h/2, 0, max(0, H-h)))
            return Image.fromarray(np.ascontiguousarray(img[y0:y0+h, x0:x0+w]))
        H, W = img.shape[:2]; s = min(w/W, h/H)
        return Image.fromarray(img).resize((max(1, int(W*s)), max(1, int(H*s))), Image.LANCZOS)

    def redraw(self):
        c = self.canvas; c.delete("all"); cw, ch = max(2, c.winfo_width()), max(2, c.winfo_height())
        if self.before is None: return
        full = self.zoom.get() == "100%" and self.full is not None
        b, a = (self.full if full else (self.before, self.after_))
        mode = self.mode.get()
        if mode == "Obok siebie":
            half = (cw-6)//2
            ib, ia = self._view(b, half, ch), self._view(a, half, ch)
            canvas = Image.new("RGB", (cw, ch), T.PANEL)
            canvas.paste(ib, ((half-ib.width)//2, (ch-ib.height)//2)); canvas.paste(ia, (half+6+(half-ia.width)//2, (ch-ia.height)//2))
            self._show(canvas); self._pill(16, 14, "Przed"); self._pill(half+22, 14, "Po")
            self.hint.configure(text="Obok siebie" + (" · przeciągnij, aby przesunąć" if full else ""))
            return
        ia = self._view(b if (mode == "Przełącz" and self.show_before) else a, cw, ch)
        x0, y0 = (cw-ia.width)//2, (ch-ia.height)//2
        canvas = Image.new("RGB", (cw, ch), T.PANEL); canvas.paste(ia, (x0, y0))
        if mode == "Podział":
            ib = self._view(b, cw, ch); sx = int(ia.width*self.split)
            canvas.paste(ib.crop((0, 0, sx, ib.height)), (x0, y0))
            self._show(canvas)
            lx = x0+sx; c.create_line(lx, y0, lx, y0+ia.height, fill=T.ACCENT, width=2)
            cy = y0+ia.height/2; c.create_oval(lx-16, cy-16, lx+16, cy+16, fill=T.ACCENT, outline="")
            c.create_text(lx, cy, text="⇔", fill=T.ON_ACCENT, font=(T.FAMILY, 13, "bold"))
            self._pill(x0+14, y0+14, "Przed"); self._pill(x0+ia.width-60, y0+14, "Po")
            self._box = (x0, y0, ia.width, ia.height)
            self.hint.configure(text="Przeciągnij linię" + (" · przeciągnij obraz, aby przesunąć" if full else "") + " · Przełącz: przytrzymaj spację")
        else:
            self._show(canvas); self._pill(x0+14, y0+14, "Przed" if self.show_before else "Po")
            self.hint.configure(text="Przytrzymaj spację, aby zobaczyć Przed")

    def _show(self, im):
        self._photo = ImageTk.PhotoImage(im); self.canvas.create_image(0, 0, image=self._photo, anchor="nw")

    def _pill(self, x, y, text):
        c = self.canvas; w = 10+8*len(text)
        c.create_rectangle(x, y, x+w, y+26, fill="#202020", outline=T.BORDER)
        c.create_text(x+w/2, y+13, text=text, fill=T.TEXT, font=(T.FAMILY, 11))

    # ----- mouse -----
    def _press(self, e):
        if self.mode.get() == "Podział" and hasattr(self, "_box"):
            x0, y0, w, h = self._box
            if abs(e.x-(x0+w*self.split)) < 24: self._drag = ("split",); return
        if self.mode.get() == "Przełącz" and self.zoom.get() != "100%":
            self.show_before = True; self.redraw(); self._drag = ("toggle",); return
        self._drag = ("pan", e.x, e.y, list(self.pan))

    def _motion(self, e):
        if not self._drag: return
        if self._drag[0] == "split":
            x0, y0, w, h = self._box; self.split = float(np.clip((e.x-x0)/max(w, 1), 0, 1)); self.redraw()
        elif self._drag[0] == "pan" and self.full is not None and self.zoom.get() == "100%":
            H, W = self.full[0].shape[:2]; _, sx, sy, p0 = self._drag
            self.pan = [float(np.clip(p0[0]-(e.x-sx)/W, 0, 1)), float(np.clip(p0[1]-(e.y-sy)/H, 0, 1))]; self.redraw()

    def _release(self, e):
        if self._drag and self._drag[0] == "toggle": self.show_before = False; self.redraw()
        self._drag = None
