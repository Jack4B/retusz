"""Lightroom Classic Histogram + OKLCh Skin Vectorscope (52° Skin Line) with:
- Interactive tonal-zone dragging (Czernie | Cienie | Ekspozycja | Światła | Biele)
- Corner clipping warning triangles (J)
- Circular colour-wheel OKLCh Skin Vectorscope matching the studio mockup.

Author: Jack4B.
"""
from __future__ import annotations
import math
import tkinter as tk
import numpy as np
from . import theme as T
from ..engine import histo as HI

def _hex(c): return "#%02X%02X%02X" % c
LO_HEX, HI_HEX = _hex(HI.LO_COLOR), _hex(HI.HI_COLOR)

ZONES = (
    (0.00, 0.16, "blacks", "Czernie"),
    (0.16, 0.38, "shadows", "Cienie"),
    (0.38, 0.62, "exposure", "Ekspozycja"),
    (0.62, 0.84, "highlights", "Światła"),
    (0.84, 1.00, "whites", "Biele"),
)

# Subtle OKLCh colour-wheel rim ticks (angle_deg, hex_color)
_WHEEL_TICKS = (
    (26, "#D85A5A"),
    (52, "#F5A33B"),
    (95, "#C8BA4E"),
    (142, "#5CB874"),
    (195, "#4CA8B8"),
    (250, "#5C7EE0"),
    (305, "#9A64D8"),
    (342, "#D45C96"),
)


class HistView(tk.Canvas):
    def __init__(self, master, on_toggle=None, on_zone_drag=None, on_zone_end=None, width=366, height=84):
        super().__init__(master, width=width, height=height, bg=T.PANEL, highlightthickness=1,
                         highlightbackground=T.BORDER, bd=0, cursor="sb_h_double_arrow")
        self.W, self.H = width, height
        self.active = False
        self.on_toggle = on_toggle
        self.on_zone_drag = on_zone_drag
        self.on_zone_end = on_zone_end
        self._last_args = (None, 0.0, 0.0, False, None)
        self._hover_zone: tuple[str, str] | None = None
        self._drag: tuple[str, str, int] | None = None
        self._moved = False

        self.bind("<Motion>", self._on_motion)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_release)
        self.draw(None)

    def _hw(self) -> int:
        return self.W - 96 if self.W >= 240 else self.W

    def _zone_at(self, x: int, y: int) -> tuple[str, str] | None:
        hw = self._hw()
        if x < 0 or x >= hw or y < 16:
            return None
        u = float(np.clip(x / max(hw, 1), 0.0, 0.9999))
        for u0, u1, key, label in ZONES:
            if u0 <= u < u1:
                return key, label
        return None

    def _on_motion(self, e):
        hw = self._hw()
        if e.x >= hw or (e.y <= 18 and (e.x <= 22 or e.x >= hw - 22)):
            self.configure(cursor="hand2")
            z = None
        else:
            self.configure(cursor="sb_h_double_arrow" if self.on_zone_drag else "hand2")
            z = self._zone_at(e.x, e.y)
        if z != self._hover_zone:
            self._hover_zone = z
            self.draw(*self._last_args)

    def _on_leave(self, e):
        if self._hover_zone is not None and self._drag is None:
            self._hover_zone = None
            self.draw(*self._last_args)

    def _on_press(self, e):
        hw = self._hw()
        self._moved = False
        if e.y <= 20 and (e.x <= 24 or (hw - 24 <= e.x <= hw)):
            if self.on_toggle:
                self.on_toggle()
            self._drag = None
            return
        z = self._zone_at(e.x, e.y)
        if z and self.on_zone_drag:
            self._drag = (z[0], z[1], e.x)
        else:
            self._drag = ("toggle", "", e.x)

    def _on_drag(self, e):
        if not self._drag or self._drag[0] == "toggle":
            return
        key, label, last_x = self._drag
        dx = e.x - last_x
        if abs(dx) >= 1:
            self._moved = True
            self._drag = (key, label, e.x)
            if self.on_zone_drag:
                self.on_zone_drag(key, dx)

    def _on_release(self, e):
        d, self._drag = self._drag, None
        if not d:
            return
        if d[0] == "toggle" or not self._moved:
            if self.on_toggle:
                self.on_toggle()
        elif self.on_zone_end:
            self.on_zone_end()

    def draw(self, h, lo=0.0, hi=0.0, active=None, vec=None):
        if active is not None:
            self.active = active
        self._last_args = (h, lo, hi, self.active, vec)
        self.delete("all")
        W, H, pad = self.W, self.H, 4
        hw = self._hw()
        top = 18

        # Subtle header captions ('Histogram' | 'OKLCh Skin')
        self.create_text(pad + 14, 9, text="Histogram", anchor="w", fill=T.MUTED, font=(T.FAMILY, 8))

        # Highlight hovered / dragged tonal zone in histogram
        z_cur = (self._drag[0], self._drag[1]) if (self._drag and self._drag[0] != "toggle") else self._hover_zone
        if z_cur is not None:
            for u0, u1, zk, zlbl in ZONES:
                if zk == z_cur[0]:
                    x0 = pad + u0 * (hw - 2 * pad)
                    x1 = pad + u1 * (hw - 2 * pad)
                    self.create_rectangle(x0, top, x1, H - pad, fill="#282830", outline="")
                    self.create_text(0.5 * (x0 + x1), top + 7, text=f"◂ {zlbl} ▸", fill=T.ACCENT, font=(T.FAMILY, 8, "bold"))
                    break

        # Subtle vertical zone dividers (Lightroom style)
        for u_div in (0.16, 0.38, 0.62, 0.84):
            xd = pad + u_div * (hw - 2 * pad)
            self.create_line(xd, top + 2, xd, H - pad, fill="#26262C", width=1)

        if h is not None:
            xs = np.linspace(pad, hw - pad, 128)
            def ys(a):
                a = np.sqrt(np.asarray(a).reshape(128, 2).sum(1))
                m = max(float(a[1:127].max()), 1e-9)
                return H - pad - np.clip(a / m, 0, 1) * (H - pad - top - 2)
            # Filled channel silhouettes + crisp top curves
            for k, fill_c, line_c in (
                ("b", "#243252", "#628EFF"),
                ("g", "#223E2E", "#58C078"),
                ("r", "#48262A", "#E26262"),
            ):
                yk = ys(h[k])
                self.create_polygon(pad, H - pad, *np.column_stack([xs, yk]).ravel(), hw - pad, H - pad,
                                    fill=fill_c, stipple="gray50", outline="")
                self.create_line(*np.column_stack([xs, yk]).ravel(), fill=line_c, width=1)
            yl = ys(h["l"])
            self.create_polygon(pad, H - pad, *np.column_stack([xs, yl]).ravel(), hw - pad, H - pad,
                                fill="#585862", stipple="gray50", outline="")
            self.create_line(*np.column_stack([xs, yl]).ravel(), fill="#C8C8D0", width=1)

        # Corner clipping indicator triangles (J)
        for frac, col, x, d in ((lo, LO_HEX, pad + 1, 9), (hi, HI_HEX, hw - pad - 1, -9)):
            on = frac > 1e-4
            self.create_polygon(x, 4, x + d, 4, x, 13, fill=col if on else "#2A2A30", outline=col if on else "#52525C")

        # Right pane: Circular OKLCh Skin Vectorscope (52° Skin Line)
        if W >= 240:
            self.create_line(hw + 1, 3, hw + 1, H - 3, fill=T.BORDER, width=1)
            cx = hw + (W - hw) // 2
            self.create_text(cx, 9, text="OKLCh Skóra", fill=T.MUTED, font=(T.FAMILY, 8))
            cy = (H + 14) // 2 - 3
            r = min(28, (H - 24) // 2)
            self.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#19191D", outline="#383842", width=1)
            self.create_oval(cx - r * 0.52, cy - r * 0.52, cx + r * 0.52, cy + r * 0.52, outline="#26262E", width=1)
            self.create_line(cx - r, cy, cx + r, cy, fill="#26262E", width=1)
            self.create_line(cx, cy - r, cx, cy + r, fill="#26262E", width=1)

            # Subtle colour-wheel rim markers
            for ang_d, hex_c in _WHEEL_TICKS:
                ar = math.radians(ang_d)
                x_in = cx + (r - 3) * math.cos(ar)
                y_in = cy - (r - 3) * math.sin(ar)
                x_out = cx + r * math.cos(ar)
                y_out = cy - r * math.sin(ar)
                self.create_line(x_in, y_in, x_out, y_out, fill=hex_c, width=2)

            # Golden 52° OKLCh Skin-Tone Line
            ang_sk = math.radians(HI.SKIN_LINE_DEG)
            sx = cx + r * math.cos(ang_sk)
            sy = cy - r * math.sin(ang_sk)
            self.create_line(cx, cy, sx, sy, fill=T.ACCENT, width=1)

            if vec is not None:
                hue_deg = float(vec.get("hue_deg", HI.SKIN_LINE_DEG))
                mean_c = float(vec.get("mean_c", 0.05))
                rad_c = min(r - 3.0, max(4.0, (mean_c / 0.14) * r))
                ang_cur = math.radians(hue_deg)
                px = cx + rad_c * math.cos(ang_cur)
                py = cy - rad_c * math.sin(ang_cur)
                self.create_line(cx, cy, px, py, fill="#F5C27C", width=1.5)
                self.create_oval(px - 3, py - 3, px + 3, py + 3, fill="#FFD89A", outline=T.ACCENT)
                self.create_text(cx, H - 6, text=f"{hue_deg:.0f}° (cel 52°)", fill=T.MUTED, font=(T.FAMILY, 8))
            else:
                self.create_text(cx, H - 6, text="osie 52°", fill=T.MUTED, font=(T.FAMILY, 8))

        if self.active:
            self.create_rectangle(1, 1, W - 1, H - 1, outline=T.ACCENT, width=1)
