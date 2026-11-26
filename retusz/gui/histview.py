"""Histogram of the result (crop region) + live OKLCh Skin-Tone Vectorscope (52° Skin Line)
with clipping indicators; click toggles the clipping overlay."""
import math
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

    def draw(self, h, lo=0.0, hi=0.0, active=None, vec=None):
        if active is not None: self.active = active
        self.delete("all"); W, H, pad = self.W, self.H, 4; top = pad + 12
        # Left pane: RGB + Luminance histogram (0 .. hw)
        hw = W - 68 if W >= 190 else W
        if h is not None:
            xs = np.linspace(pad, hw - pad, 128)
            def ys(a):
                a = np.sqrt(np.asarray(a).reshape(128, 2).sum(1)); m = max(float(a[1:127].max()), 1e-9)
                return H - pad - np.clip(a / m, 0, 1) * (H - pad - top)
            l = ys(h["l"])
            self.create_polygon(pad, H - pad, *np.column_stack([xs, l]).ravel(), hw - pad, H - pad, fill="#4A4A4A", outline="")
            for k, c in (("r", "#E06060"), ("g", "#5CC27A"), ("b", "#6A96FF")):
                self.create_line(*np.column_stack([xs, ys(h[k])]).ravel(), fill=c, width=1)
        for frac, col, x, d in ((lo, LO_HEX, pad + 1, 10), (hi, HI_HEX, hw - pad - 1, -10)):
            on = frac > 1e-4
            self.create_polygon(x, pad, x + d, pad, x, pad + 10, fill=col if on else T.PANEL, outline=col if on else T.MUTED)

        # Right pane: OKLCh Skin-Tone Vectorscope (52° Skin Line)
        if W >= 190:
            self.create_line(hw + 1, pad, hw + 1, H - pad, fill=T.BORDER, width=1)
            cx = hw + (W - hw) // 2
            cy = 32
            r = 23
            self.create_oval(cx - r, cy - r, cx + r, cy + r, outline="#353840", width=1)
            self.create_line(cx - r, cy, cx + r, cy, fill="#282A30", width=1)
            self.create_line(cx, cy - r, cx, cy + r, fill="#282A30", width=1)
            # Golden 52° OKLCh Skin-Tone Line
            ang_sk = math.radians(HI.SKIN_LINE_DEG)
            sx = cx + r * math.cos(ang_sk)
            sy = cy - r * math.sin(ang_sk)
            self.create_line(cx, cy, sx, sy, fill=T.ACCENT, width=1)

            if vec is not None:
                hue_deg = float(vec.get("hue_deg", HI.SKIN_LINE_DEG))
                mean_c = float(vec.get("mean_c", 0.05))
                rad_c = min(r - 2.0, max(4.0, (mean_c / 0.14) * r))
                ang_cur = math.radians(hue_deg)
                px = cx + rad_c * math.cos(ang_cur)
                py = cy - rad_c * math.sin(ang_cur)
                self.create_line(cx, cy, px, py, fill="#F5C27C", width=1)
                self.create_oval(px - 2.5, py - 2.5, px + 2.5, py + 2.5, fill="#FFD08A", outline=T.ACCENT)
                self.create_text(cx, H - 7, text=f"Skóra {hue_deg:.0f}°", fill=T.MUTED, font=(T.FAMILY, 8))
            else:
                self.create_text(cx, H - 7, text="Skóra 52°", fill=T.MUTED, font=(T.FAMILY, 8))

        if self.active: self.create_rectangle(1, 1, W - 1, H - 1, outline=T.ACCENT, width=1)
