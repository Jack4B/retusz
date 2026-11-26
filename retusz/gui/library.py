"""Moduł Biblioteka (G): widok siatki zdjęć serii (Contact Sheet / Grid View) w stylu Lightroom / Capture One.

Funkcje:
- Filtrowanie serii: Wszystkie | Wybrane ★ (rating >= 1) | Edytowane ● (posiadające własny sidecar)
- Ocena gwiazdkowa 1..5 ★ dla każdego zdjęcia
- Podgląd i zmiana indywidualnego wzorca kolorystycznego dla pojedynczego zdjęcia lub zaznaczonej grupy zdjęć
- Zastosowanie presetu do zaznaczonych zdjęć
- Dwuklik lub przycisk „Przejdź do Edycji (D)” otwiera wybrane zdjęcie bezpośrednio w module Edycja (Develop).

Author: Jack4B.
"""
from __future__ import annotations
from pathlib import Path
from tkinter import filedialog
import customtkinter as ctk
from . import theme as T


class LibraryView(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, fg_color=T.BG)
        self.app = app
        self.filter_mode = "Wszystkie"
        self.selected: set[int] = set()
        self.card_widgets: list[ctk.CTkFrame] = []

        # Top toolbar of Library module
        bar = ctk.CTkFrame(self, fg_color=T.BG)
        bar.pack(fill="x", padx=14, pady=(12, 8))

        self.mod_switch = T.segmented(bar, ["Biblioteka (G)", "Edycja (D)"], command=self._on_mod)
        self.mod_switch.set("Biblioteka (G)")
        self.mod_switch.pack(side="left")

        self.filter_seg = T.segmented(bar, ["Wszystkie", "Wybrane ★", "Edytowane ●"], command=self._on_filter)
        self.filter_seg.set("Wszystkie")
        self.filter_seg.pack(side="left", padx=16)

        T.button(bar, "Zaznacz widoczne", self.select_visible, width=130).pack(side="left", padx=(4, 4))
        T.button(bar, "Odznacz", self.clear_selection, width=82).pack(side="left")

        T.button(bar, "Eksportuj jako…", self.app.open_export_as, accent=True, width=132).pack(side="right")
        T.button(bar, "Przejdź do Edycji (D)", self.app.hide_library, accent=True, width=156).pack(side="right", padx=8)

        # Body: scrollable contact-sheet grid on the left, batch inspector on the right
        body = ctk.CTkFrame(self, fg_color=T.BG)
        body.pack(fill="both", expand=True, padx=14, pady=(0, 10))

        self.grid_scroll = ctk.CTkScrollableFrame(body, fg_color=T.PANEL, corner_radius=10)
        self.grid_scroll.pack(side="left", fill="both", expand=True, padx=(0, 12))
        for c in range(4):
            self.grid_scroll.grid_columnconfigure(c, weight=1, uniform="lib_col")

        side = ctk.CTkFrame(body, fg_color=T.CARD, width=286, corner_radius=10, border_width=1, border_color=T.BORDER)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)

        T.label(side, "Inspektor Serii i Wzorców", 14, weight="bold").pack(anchor="w", padx=14, pady=(14, 4))
        self.sel_lbl = T.label(side, "Zaznaczone zdjęcia: 0", 12, T.MUTED)
        self.sel_lbl.pack(anchor="w", padx=14, pady=(0, 10))

        T.label(side, "Wzorzec dla zaznaczonych zdjęć:", 12, anchor="w").pack(anchor="w", padx=14, pady=(4, 4))
        T.button(side, "Własny wzorzec z pliku…", self.assign_ref_file, accent=True, width=256).pack(padx=14, pady=3)
        T.button(side, "Tablica ColorChecker 24…", self.assign_cc_file, width=256).pack(padx=14, pady=3)
        T.button(side, "Domyślny wzorzec serii", self.assign_series_ref, width=256).pack(padx=14, pady=3)
        T.button(side, "Brak wzorca (samodzielny)", self.assign_no_ref, width=256).pack(padx=14, pady=(3, 12))

        T.label(side, "Ocena zaznaczonych (★):", 12, anchor="w").pack(anchor="w", padx=14, pady=(4, 4))
        r_row = ctk.CTkFrame(side, fg_color=T.CARD)
        r_row.pack(fill="x", padx=14, pady=(0, 12))
        for stars in range(6):
            txt = "0" if stars == 0 else f"{stars}★"
            b = T.button(r_row, txt, lambda s=stars: self.set_rating_selected(s), width=36)
            b.configure(height=26, font=T.font(11))
            b.pack(side="left", padx=(0, 3))

        T.label(side, "Zastosuj preset do zaznaczonych:", 12, anchor="w").pack(anchor="w", padx=14, pady=(4, 4))
        self.preset_menu = ctk.CTkOptionMenu(
            side, values=[p.name for p in self.app.presets],
            command=self.apply_preset_selected, width=240, height=28,
            fg_color=T.BTN, button_color=T.BTN, button_hover_color=T.BTN_HOVER, text_color=T.TEXT
        )
        self.preset_menu.pack(padx=14, pady=(0, 14))

        self.info_lbl = T.label(
            side,
            "Wskazówka: Kliknij kartę, aby zaznaczyć zdjęcie.\n"
            "Kliknij „Edytuj” na karcie, aby otworzyć zdjęcie w module Edycja (D).\n"
            "Każde zdjęcie w serii może mieć przypisany własny wzorzec kolorystyczny.",
            11, T.MUTED, justify="left", wraplength=240
        )
        self.info_lbl.pack(anchor="w", padx=14, pady=(8, 10))

    def _on_mod(self, val: str):
        if val.startswith("Edycja"):
            self.mod_switch.set("Biblioteka (G)")
            self.app.hide_library()

    def _on_filter(self, val: str):
        self.filter_mode = val
        self.refresh()

    def _target_indices(self) -> list[int]:
        if self.selected:
            return sorted(i for i in self.selected if 0 <= i < len(self.app.photos))
        if 0 <= self.app.idx < len(self.app.photos):
            return [self.app.idx]
        return []

    def select_visible(self):
        for i in self._visible_indices():
            self.selected.add(i)
        self.refresh()

    def clear_selection(self):
        self.selected.clear()
        self.refresh()

    def _visible_indices(self) -> list[int]:
        out = []
        ser = self.app.series
        for i, p in enumerate(self.app.photos):
            if self.filter_mode.startswith("Wybrane"):
                pr = ser.params_for(p) if ser else None
                if not pr or pr.rating < 1:
                    continue
            elif self.filter_mode.startswith("Edytowane"):
                if not (ser and ser.has_override(p)):
                    continue
            out.append(i)
        return out

    def toggle_select(self, idx: int):
        if idx in self.selected:
            self.selected.discard(idx)
        else:
            self.selected.add(idx)
        self.refresh()

    def open_in_develop(self, idx: int):
        self.app.hide_library()
        if idx != self.app.idx:
            self.app.select(idx)

    def set_photo_rating(self, idx: int, stars: int):
        if not self.app.series or not (0 <= idx < len(self.app.photos)):
            return
        ph = self.app.photos[idx]
        p = self.app.series.params_for(ph)
        p.rating = stars
        self.app.series.set_override(ph, p)
        if idx == self.app.idx:
            self.app.params.rating = stars
        self.refresh()

    def set_rating_selected(self, stars: int):
        if not self.app.series:
            return
        for idx in self._target_indices():
            ph = self.app.photos[idx]
            p = self.app.series.params_for(ph)
            p.rating = stars
            self.app.series.set_override(ph, p)
            if idx == self.app.idx:
                self.app.params.rating = stars
        self.refresh()

    def _assign_ref_to_targets(self, ref_path: str | None, cc: bool = False):
        if not self.app.series:
            return
        for idx in self._target_indices():
            ph = self.app.photos[idx]
            p = self.app.series.params_for(ph)
            p.reference_path = ref_path
            p.ref_colorchecker = cc
            self.app.series.set_override(ph, p)
            if idx == self.app.idx:
                self.app.params.reference_path = ref_path
                self.app.params.ref_colorchecker = cc
        self.app.series.save()
        if 0 <= self.app.idx < len(self.app.photos):
            self.app._sync_photo_reference()
        self.refresh()

    def assign_ref_file(self):
        from .app import PHOTO_TYPES
        p = filedialog.askopenfilename(title="Wybierz wzorzec dla zaznaczonych zdjęć",
                                       initialdir=self.app.settings.last_folder or None, filetypes=PHOTO_TYPES)
        if p:
            self._assign_ref_to_targets(str(Path(p)), cc=False)

    def assign_cc_file(self):
        from .app import PHOTO_TYPES
        p = filedialog.askopenfilename(title="Wybierz tablicę ColorChecker 24 dla zaznaczonych zdjęć",
                                       initialdir=self.app.settings.last_folder or None, filetypes=PHOTO_TYPES)
        if p:
            self._assign_ref_to_targets(str(Path(p)), cc=True)

    def assign_series_ref(self):
        self._assign_ref_to_targets(None, cc=False)

    def assign_no_ref(self):
        self._assign_ref_to_targets("__none__", cc=False)

    def apply_preset_selected(self, name: str):
        if not self.app.series:
            return
        pr = next((x for x in self.app.presets if x.name == name), None)
        if pr is None:
            return
        for idx in self._target_indices():
            ph = self.app.photos[idx]
            cur = self.app.series.params_for(ph)
            upd = pr.apply_to(cur)
            upd.rating = cur.rating
            upd.reference_path = cur.reference_path
            upd.ref_colorchecker = cur.ref_colorchecker
            self.app.series.set_override(ph, upd)
            if idx == self.app.idx:
                self.app.params = upd
                self.app._load_params_into_ui(upd)
                self.app.request_render()
        self.refresh()

    def refresh(self):
        for w in self.card_widgets:
            w.destroy()
        self.card_widgets.clear()
        self.mod_switch.set("Biblioteka (G)")

        vis = self._visible_indices()
        targets = self._target_indices()
        self.sel_lbl.configure(text=f"Zaznaczone zdjęcia: {len(targets)}  (widoczne: {len(vis)}/{len(self.app.photos)})")

        ser = self.app.series
        for pos, i in enumerate(vis):
            ph = self.app.photos[i]
            p = ser.params_for(ph) if ser else self.app.params
            is_sel = (i in self.selected) or (not self.selected and i == self.app.idx)
            border_c = T.ACCENT if is_sel else T.BORDER

            card = ctk.CTkFrame(self.grid_scroll, fg_color=T.CARD, corner_radius=8, border_width=2, border_color=border_c)
            card.grid(row=pos // 4, column=pos % 4, padx=6, pady=6, sticky="nsew")
            self.card_widgets.append(card)

            # Thumbnail button
            im = self.app.thumbs.get(ph)
            tbtn = ctk.CTkButton(
                card, text="", image=im, width=150, height=104,
                fg_color=T.PANEL, hover_color=T.BTN_HOVER, corner_radius=6,
                command=lambda idx=i: self.toggle_select(idx)
            )
            tbtn.pack(padx=8, pady=(8, 4), fill="x")

            # Filename + Edited dot
            edited = ser.has_override(ph) if ser else False
            name_txt = ("● " if edited else "") + (ph.name if len(ph.name) <= 22 else ph.name[:19] + "…")
            T.label(card, name_txt, 12, T.ACCENT if edited else T.TEXT, weight="bold").pack(anchor="w", padx=8)

            # Reference badge
            if p.reference_path == "__none__":
                ref_txt = "Wzorzec: Samodzielne (brak)"
            elif p.reference_path:
                prefix = "CC24: " if p.ref_colorchecker else "Wzorzec: "
                ref_txt = prefix + Path(p.reference_path).name[:18]
            elif ser and ser.reference:
                ref_txt = "Seria: " + ser.reference.name[:18]
            else:
                ref_txt = "Tryb: Samodzielny edytor"
            T.label(card, ref_txt, 11, T.MUTED).pack(anchor="w", padx=8, pady=(1, 3))

            # Bottom row: Stars + Edit button
            bot = ctk.CTkFrame(card, fg_color=T.CARD)
            bot.pack(fill="x", padx=8, pady=(0, 8))
            stars_txt = "★" * p.rating + "☆" * (5 - p.rating)
            s_btn = T.button(bot, stars_txt, lambda idx=i, cur=p.rating: self.set_photo_rating(idx, (cur + 1) % 6), width=78)
            s_btn.configure(height=24, font=T.font(11), text_color=T.ACCENT if p.rating > 0 else T.MUTED)
            s_btn.pack(side="left")

            e_btn = T.button(bot, "Edytuj (D)", lambda idx=i: self.open_in_develop(idx), accent=(i == self.app.idx), width=76)
            e_btn.configure(height=24, font=T.font(11))
            e_btn.pack(side="right")
