"""Colour-managed, zoomable image canvas.

Zoom is in % of ORIGINAL image pixels (25-400%) or 'fit'. When the zoom needs more pixels than the
preview has, a full-resolution 'detail' image (set by the app) is used. to_norm() maps canvas
coordinates to normalised image coordinates at any zoom, so the brush stays accurate."""
import tkinter as tk
import numpy as np, cv2
from PIL import Image, ImageTk
from . import theme as T
from .. import colorman as CM

ZMIN, ZMAX = 0.25, 4.0

class ImageView(tk.Canvas):
    def __init__(self, master, placeholder="", **kw):
        super().__init__(master, bg=T.PANEL, highlightthickness=1, highlightbackground=T.BORDER, bd=0, **kw)
        self.rgb = None; self.icc = None; self.full_wh = None; self.detail = None
        self.split_rgb = None; self.split_detail = None; self.split_u = None; self._split_disp = None
        self.zoom = None; self.center = [0.5, 0.5]; self.geom = (0, 0, 1, 1)
        self._photo = None; self._disp = None; self.placeholder = placeholder
        self.on_view_change = None; self.on_redraw = None
        self.bind("<Configure>", lambda e: self.redraw())

    # ----- content -----
    def set_image(self, rgb, icc=None, full_wh=None, keep_view=True):
        self.rgb = rgb; self.icc = icc; self.full_wh = full_wh; self._disp = None
        if not keep_view: self.zoom = None; self.center = [0.5, 0.5]; self.detail = None
        self.redraw()

    def set_split(self, before_rgb=None, split_u=None, before_detail=None):
        self.split_rgb = before_rgb; self.split_u = split_u; self.split_detail = before_detail
        self._split_disp = None
        self.redraw()

    def set_detail(self, rgb):
        self.detail = rgb; self.redraw()

    def _base(self):
        """Colour-managed preview (converted once per image)."""
        if self._disp is None and self.rgb is not None:
            self._disp = CM.for_display(self.rgb, self.icc)
        return self._disp

    def _base_split(self):
        if self._split_disp is None and self.split_rgb is not None:
            self._split_disp = CM.for_display(self.split_rgb, self.icc)
        return self._split_disp

    def size_wh(self):
        if self.full_wh: return self.full_wh
        if self.rgb is None: return (1, 1)
        return (self.rgb.shape[1], self.rgb.shape[0])

    # ----- geometry -----
    def fit_zoom(self):
        cw, ch = max(2, self.winfo_width()), max(2, self.winfo_height()); W, H = self.size_wh()
        return min((cw-8)/W, (ch-8)/H)

    def cur_zoom(self):
        return self.fit_zoom() if self.zoom is None else self.zoom

    def needs_detail(self):
        return self.rgb is not None and self.cur_zoom()*self.size_wh()[0] > self.rgb.shape[1]*1.05

    def to_norm(self, x, y):
        x0, y0, dw, dh = self.geom
        return (x-x0)/dw, (y-y0)/dh

    def scale(self):
        """Displayed width of the whole image in canvas pixels."""
        return self.geom[2]

    def set_view(self, zoom, center, notify=False):
        self.zoom = None if zoom is None else float(np.clip(zoom, ZMIN, ZMAX))
        self.center = [float(np.clip(center[0], 0, 1)), float(np.clip(center[1], 0, 1))]
        if self.zoom is None: self.center = [0.5, 0.5]
        self.redraw()
        if notify and self.on_view_change: self.on_view_change(self)

    def zoom_at(self, factor, x, y):
        if self.rgb is None: return
        cw, ch = self.winfo_width(), self.winfo_height(); W, H = self.size_wh()
        u, v = self.to_norm(x, y); z = float(np.clip(self.cur_zoom()*factor, ZMIN, ZMAX))
        if z <= self.fit_zoom()*1.001 and factor < 1: self.set_view(None, (0.5, 0.5), True); return
        self.set_view(z, (u + (cw/2-x)/(W*z), v + (ch/2-y)/(H*z)), True)

    def pan(self, dx, dy):
        if self.zoom is None: return
        x0, y0, dw, dh = self.geom
        self.set_view(self.zoom, (self.center[0]-dx/dw, self.center[1]-dy/dh), True)

    # ----- drawing -----
    def redraw(self):
        self.delete("img"); self.delete("ph"); self.delete("split_line")
        cw, ch = max(2, self.winfo_width()), max(2, self.winfo_height())
        if self.rgb is None:
            self.create_text(cw/2, ch/2, text=self.placeholder, fill=T.MUTED, font=(T.FAMILY, 12), tags="ph", justify="center")
            return
        W, H = self.size_wh(); z = self.cur_zoom(); dw, dh = W*z, H*z
        cx, cy = (0.5, 0.5) if self.zoom is None else self.center
        x0, y0 = cw/2 - cx*dw, ch/2 - cy*dh
        if self.zoom is not None:   # keep the image on screen
            if dw <= cw: x0 = (cw-dw)/2
            else: x0 = min(0, max(cw-dw, x0))
            if dh <= ch: y0 = (ch-dh)/2
            else: y0 = min(0, max(ch-dh, y0))
            self.center = [(cw/2-x0)/dw, (ch/2-y0)/dh]
        self.geom = (x0, y0, dw, dh)
        u0, u1 = max(0.0, -x0/dw), min(1.0, (cw-x0)/dw); v0, v1 = max(0.0, -y0/dh), min(1.0, (ch-y0)/dh)
        if u1 <= u0 or v1 <= v0: return
        use_detail = self.detail is not None and self.needs_detail()
        src = self.detail if use_detail else self._base()
        sh, sw = src.shape[:2]
        a0, a1 = int(u0*sw), max(int(u0*sw)+1, int(np.ceil(u1*sw))); b0, b1 = int(v0*sh), max(int(v0*sh)+1, int(np.ceil(v1*sh)))
        crop = src[b0:b1, a0:a1]
        tw, th = max(1, int(round((a1-a0)/sw*dw))), max(1, int(round((b1-b0)/sh*dh)))
        interp = cv2.INTER_AREA if tw < crop.shape[1] else (cv2.INTER_NEAREST if tw > 3*crop.shape[1] else cv2.INTER_LINEAR)
        img = cv2.resize(np.ascontiguousarray(crop), (tw, th), interpolation=interp)
        if use_detail: img = CM.for_display(img, self.icc)

        # Interactive Before | After split curtain ("Podział")
        if self.split_u is not None and self.split_rgb is not None:
            b_src = self.split_detail if (use_detail and self.split_detail is not None) else self._base_split()
            if b_src is not None and b_src.shape[:2] == (sh, sw):
                b_crop = b_src[b0:b1, a0:a1]
                b_img = cv2.resize(np.ascontiguousarray(b_crop), (tw, th), interpolation=interp)
                if use_detail and self.split_detail is not None:
                    b_img = CM.for_display(b_img, self.icc)
                rel_u = (float(self.split_u) - a0 / sw) / max(1e-6, (a1 - a0) / sw)
                sx_px = int(np.clip(round(rel_u * tw), 0, tw))
                if sx_px > 0:
                    img = img.copy()
                    img[:, :sx_px] = b_img[:, :sx_px]

        ix0 = x0 + a0/sw*dw
        iy0 = y0 + b0/sh*dh
        self._photo = ImageTk.PhotoImage(Image.fromarray(img))
        self.create_image(ix0, iy0, image=self._photo, anchor="nw", tags="img")
        self.tag_lower("img")

        if self.split_u is not None and self.split_rgb is not None:
            lx = x0 + float(self.split_u) * dw
            if ix0 <= lx <= ix0 + tw:
                self.create_line(lx, iy0, lx, iy0 + th, fill=T.ACCENT, width=2, tags="split_line")
                my = iy0 + th * 0.5
                self.create_oval(lx - 11, my - 11, lx + 11, my + 11, fill=T.CARD, outline=T.ACCENT, width=2, tags="split_line")
                self.create_line(lx - 3, my - 5, lx - 3, my + 5, fill=T.TEXT, width=1, tags="split_line")
                self.create_line(lx + 3, my - 5, lx + 3, my + 5, fill=T.TEXT, width=1, tags="split_line")

        if self.on_redraw: self.on_redraw()
