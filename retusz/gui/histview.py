"""Histogram of the result (crop region) with clipping indicators; click toggles the clipping overlay."""
import tkinter as tk
import numpy as np
from . import theme as T
from ..engine import histo as HI

def _hex(c): return "#%02X%02X%02X" % c
LO_HEX, HI_HEX = _hex(HI.LO_COLOR), _hex(HI.HI_COLOR)

class HistView(tk.Canvas):
    def __init__(self, master, on_toggle=None, width=224, height=74):
        super().__init__(master, width=width, height=height, bg=T.PANEL, highlightthickness=1,
                         highlightbackground=T.BORDER, bd=0, cursor="hand2")
        self.W, self.H = width, height; self.active = False
        self.bind("<Button-1>", lambda e: on_toggle and on_toggle())
        self.draw(None)

    def draw(self, h, lo=0.0, hi=0.0, active=None):
        if active is not None: self.active = active
        self.delete("all"); W, H, pad = self.W, self.H, 4; top = pad + 12
        if h is not None:
            xs = np.linspace(pad, W-pad, 128)
            def ys(a):
                a = np.sqrt(np.asarray(a).reshape(128, 2).sum(1)); m = max(float(a[1:127].max()), 1e-9)
                return H-pad - np.clip(a/m, 0, 1)*(H-pad-top)
            l = ys(h["l"])
            self.create_polygon(pad, H-pad, *np.column_stack([xs, l]).ravel(), W-pad, H-pad, fill="#4A4A4A", outline="")
            for k, c in (("r", "#E06060"), ("g", "#5CC27A"), ("b", "#6A96FF")):
                self.create_line(*np.column_stack([xs, ys(h[k])]).ravel(), fill=c, width=1)
        for frac, col, x, d in ((lo, LO_HEX, pad+1, 10), (hi, HI_HEX, W-pad-1, -10)):
            on = frac > 1e-4
            self.create_polygon(x, pad, x+d, pad, x, pad+10, fill=col if on else T.PANEL, outline=col if on else T.MUTED)
        if self.active: self.create_rectangle(1, 1, W-1, H-1, outline=T.ACCENT, width=1)
