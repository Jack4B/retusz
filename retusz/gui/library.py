"""Moduł Biblioteka (G) w stylu Adobe Lightroom Classic / Capture One:
- Lewy panel: Drzewo Folderów Roboczych (automatyczne wykrywanie folderów i podfolderów ze zdjęciami + licznik zdjęć)
  oraz Kolekcje Smart (Wszystkie, Wybrane ★, Własny wzorzec 🎯, Edytowane ●)
- Środkowa siatka kart (Contact Sheet): każde zdjęcie posiada bezpośredni przycisk „🎯 Wzorzec…” do przypisania
  indywidualnego wzorca tylko temu zdjęciu, gwiazdki oceny 1..5 ★ oraz przycisk „Edytuj (D)”
- Prawy dok: Inspektor Wzorców i Serii z Pulą Wzorców („+ Dodaj wzorce do puli…”) oraz inteligentnym parowaniem
  „✨ Auto-dopasuj wzorce z puli (AI)” (dobiera dla każdego zdjęcia z serii najbliższy wzorzec po sygnaturze OKLab).

Author: Jack4B.
"""
from __future__ import annotations
from pathlib import Path
from tkinter import filedialog
import customtkinter as ctk
from . import theme as T
from ..sidecar import Series


class LibraryView(ctk.CTkFrame):
    def __init__(self, master, app):
        super().__init__(master, fg_color=T.BG)
        self.app = app
        self.filter_mode = "Wszystkie"
        self.selected: set[int] = set()
        self.card_widgets: list[ctk.CTkFrame] = []
        self.folder_widgets: list[ctk.CTkButton] = []

        # Top toolbar of Library module
        bar = ctk.CTkFrame(self, fg_color=T.BG)
        bar.pack(fill="x", padx=12, pady=(10, 6))

        self.mod_switch = T.segmented(bar, ["Biblioteka (G)", "Edycja (D)"], command=self._on_mod)
        self.mod_switch.set("Biblioteka (G)")
        self.mod_switch.pack(side="left")

        self.filter_seg = T.segmented(
            bar, ["Wszystkie", "Flaga ⚑", "Wybrane ★", "Własny wzorzec ♦", "Odrzucone ✕"], command=self._on_filter
        )
        self.filter_seg.set("Wszystkie")
        self.filter_seg.pack(side="left", padx=10)

        T.button(bar, "+ Importuj wybrane…", self.app.add_photos, width=136).pack(side="left", padx=(0, 4))
        T.button(bar, "Zaznacz widoczne", self.select_visible, width=122).pack(side="left", padx=(0, 4))
        T.button(bar, "Odznacz", self.clear_selection, width=74).pack(side="left")

        T.button(bar, "Eksportuj jako…", self.app.open_export_as, accent=True, width=128).pack(side="right")
        T.button(bar, "Przejdź do Edycji (D)", self.app.hide_library, accent=True, width=152).pack(side="right", padx=6)

        # Body: 3-pane Lightroom Library layout:
        # Left = Folder Tree & Collections | Center = Contact-Sheet Grid | Right = Reference Pool & Batch Inspector
        body = ctk.CTkFrame(self, fg_color=T.BG)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        # ---- Left Pane: Lightroom Folder Tree & Workspace Navigator ----
        left = ctk.CTkFrame(body, fg_color=T.CARD, width=228, corner_radius=10, border_width=1, border_color=T.BORDER)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        T.label(left, "▸ Foldery Robocze", 13, weight="bold").pack(anchor="w", padx=12, pady=(12, 4))
        f_act = ctk.CTkFrame(left, fg_color=T.CARD)
        f_act.pack(fill="x", padx=10, pady=(0, 6))
        b_of = T.button(f_act, "Otwórz folder…", self.app.open_folder, width=100)
        b_of.configure(height=25, font=T.font(11)); b_of.pack(side="left")
        b_os = T.button(f_act, "Wybierz pliki…", self.app.open_photo, width=100)
        b_os.configure(height=25, font=T.font(11)); b_os.pack(side="right")

        self.folder_scroll = ctk.CTkScrollableFrame(left, fg_color=T.PANEL, corner_radius=8, height=250)
        self.folder_scroll.pack(fill="x", padx=10, pady=(0, 10))

        T.label(left, "▸ Kolekcje Smart", 13, weight="bold").pack(anchor="w", padx=12, pady=(4, 4))
        for col_name in ("Wszystkie", "Flaga ⚑", "Wybrane ★", "Własny wzorzec ♦", "Edytowane ●", "Odrzucone ✕"):
            cb = T.button(left, col_name, lambda c=col_name: self._set_collection(c), width=204)
            cb.configure(height=25, font=T.font(11))
            cb.pack(padx=10, pady=2)

        # ---- Center Pane: Scrollable Contact-Sheet Grid (3 columns) ----
        self.grid_scroll = ctk.CTkScrollableFrame(body, fg_color=T.PANEL, corner_radius=10)
        self.grid_scroll.pack(side="left", fill="both", expand=True, padx=(0, 10))
        for c in range(3):
            self.grid_scroll.grid_columnconfigure(c, weight=1, uniform="lib_col")

        # ---- Right Pane: Reference Pool & Per-Photo Pairing Inspector ----
        side = ctk.CTkFrame(body, fg_color=T.CARD, width=294, corner_radius=10, border_width=1, border_color=T.BORDER)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)

        T.label(side, "Inspektor Wzorców i Serii", 14, weight="bold").pack(anchor="w", padx=14, pady=(12, 2))
        self.sel_lbl = T.label(side, "Zaznaczone zdjęcia: 0", 11, T.MUTED)
        self.sel_lbl.pack(anchor="w", padx=14, pady=(0, 8))

        T.label(side, "Indywidualny wzorzec dla zaznaczonych:", 12, weight="bold", anchor="w").pack(anchor="w", padx=14, pady=(2, 3))
        T.button(side, "♦ Przypisz własny wzorzec z pliku…", self.assign_ref_file, accent=True, width=264).pack(padx=14, pady=2)
        T.button(side, "Przypisz tablicę ColorChecker 24…", self.assign_cc_file, width=264).pack(padx=14, pady=2)
        T.button(side, "Używaj domyślnego wzorca serii", self.assign_series_ref, width=264).pack(padx=14, pady=2)
        T.button(side, "Brak wzorca (samodzielny edytor)", self.assign_no_ref, width=264).pack(padx=14, pady=(2, 8))

        T.label(side, "Pula Wzorców Serii (Auto-Parowanie AI):", 12, weight="bold", anchor="w").pack(anchor="w", padx=14, pady=(4, 2))
        self.pool_lbl = T.label(side, "Wzorce w puli: 0", 11, T.MUTED, anchor="w")
        self.pool_lbl.pack(anchor="w", padx=14, pady=(0, 3))
        T.button(side, "+ Dodaj wzorce do puli (wiele)…", self.add_references_to_pool, width=264).pack(padx=14, pady=2)
        T.button(side, "★ Auto-dopasuj wzorce z puli (AI)", self.auto_match_pool_references, accent=True, width=264).pack(padx=14, pady=(2, 8))

        self.pool_menu = ctk.CTkOptionMenu(
            side, values=["(Wybierz wzorzec z puli)"],
            command=self.assign_pool_reference_by_name, width=264, height=26,
            fg_color=T.BTN, button_color=T.BTN, button_hover_color=T.BTN_HOVER, text_color=T.TEXT,
            font=T.font(11), dropdown_font=T.font(11)
        )
        self.pool_menu.pack(padx=14, pady=(0, 10))

        T.label(side, "Ocena zaznaczonych (★):", 12, anchor="w").pack(anchor="w", padx=14, pady=(2, 3))
        r_row = ctk.CTkFrame(side, fg_color=T.CARD)
        r_row.pack(fill="x", padx=14, pady=(0, 8))
        for stars in range(6):
            txt = "0" if stars == 0 else f"{stars}★"
            b = T.button(r_row, txt, lambda s=stars: self.set_rating_selected(s), width=38)
            b.configure(height=24, font=T.font(11))
            b.pack(side="left", padx=(0, 3))

        T.label(side, "Zastosuj preset do zaznaczonych:", 12, anchor="w").pack(anchor="w", padx=14, pady=(2, 3))
        self.preset_menu = ctk.CTkOptionMenu(
            side, values=[p.name for p in self.app.presets],
            command=self.apply_preset_selected, width=264, height=26,
            fg_color=T.BTN, button_color=T.BTN, button_hover_color=T.BTN_HOVER, text_color=T.TEXT,
            font=T.font(11), dropdown_font=T.font(11)
        )
        self.preset_menu.pack(padx=14, pady=(0, 8))

        self.info_lbl = T.label(
            side,
            "▸ Każde zdjęcie ma na karcie przycisk „♦ Wzorzec…”, aby dograć mu własny wzorzec.\n"
            "Możesz też wgrać kilka wzorców do puli i kliknąć „★ Auto-dopasuj wzorce z puli (AI)”.",
            10, T.MUTED, justify="left", wraplength=264
        )
        self.info_lbl.pack(anchor="w", padx=14, pady=(4, 8))

    def _set_collection(self, col_name: str):
        self.filter_mode = col_name
        self.filter_seg.set(col_name)
        self.refresh()

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
            pr = ser.params_for(p) if ser else None
            if self.filter_mode.startswith("Flaga"):
                if not pr or getattr(pr, "flag", 0) < 1:
                    continue
            elif self.filter_mode.startswith("Odrzucone"):
                if not pr or getattr(pr, "flag", 0) >= 0:
                    continue
            elif self.filter_mode.startswith("Wybrane"):
                if not pr or pr.rating < 1:
                    continue
            elif self.filter_mode.startswith("Własny wzorzec"):
                if not pr or not pr.reference_path or pr.reference_path == "__none__":
                    continue
            elif self.filter_mode.startswith("Edytowane"):
                if not (ser and ser.has_override(p)):
                    continue
            out.append(i)
        return out

    def toggle_photo_flag(self, idx: int):
        if not self.app.series or not (0 <= idx < len(self.app.photos)):
            return
        ph = self.app.photos[idx]
        p = self.app.series.params_for(ph)
        cur = getattr(p, "flag", 0)
        nxt = 1 if cur == 0 else (-1 if cur == 1 else 0)
        p.flag = nxt
        self.app.series.set_override(ph, p)
        if idx == self.app.idx:
            self.app.params.flag = nxt
            self.app._update_flag_btns(nxt)
        self.refresh()

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

    def _assign_ref_to_indices(self, indices: list[int], ref_path: str | None, cc: bool = False):
        if not self.app.series:
            return
        for idx in indices:
            if not (0 <= idx < len(self.app.photos)):
                continue
            ph = self.app.photos[idx]
            p = self.app.series.params_for(ph)
            p.reference_path = ref_path
            p.ref_colorchecker = cc
            self.app.series.set_override(ph, p)
            if idx == self.app.idx:
                self.app.params.reference_path = ref_path
                self.app.params.ref_colorchecker = cc
        self.app.series.save()
        self.app._refresh_strip_marks()
        if 0 <= self.app.idx < len(self.app.photos):
            self.app._sync_photo_reference()
        self.refresh()

    def _assign_ref_to_targets(self, ref_path: str | None, cc: bool = False):
        self._assign_ref_to_indices(self._target_indices(), ref_path, cc=cc)

    def assign_ref_for_single_photo(self, idx: int):
        """1-click reference picker directly on a photo card in Biblioteka (G)."""
        from .app import PHOTO_TYPES
        if not (0 <= idx < len(self.app.photos)):
            return
        ph = self.app.photos[idx]
        p = filedialog.askopenfilename(
            title=f"Wybierz własny wzorzec kolorystyczny dla zdjęcia {ph.name}",
            initialdir=self.app.settings.last_folder or None,
            filetypes=PHOTO_TYPES,
        )
        if p:
            self._assign_ref_to_indices([idx], str(Path(p)), cc=False)

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

    def add_references_to_pool(self):
        from .app import PHOTO_TYPES
        paths = filedialog.askopenfilenames(
            title="Dodaj jeden lub więcej wzorców do Puli Wzorców Serii",
            initialdir=self.app.settings.last_folder or None,
            filetypes=PHOTO_TYPES,
        )
        if not paths:
            return
        if self.app.series is None:
            self.app.series = Series(Path(paths[0]).parent)
        for p in paths:
            self.app.series.add_to_reference_bank(Path(p))
        self.app.series.save()
        self.app._refresh_photo_ref_menu()
        self.refresh()

    def auto_match_pool_references(self):
        if not self.app.series or not self.app.photos:
            return
        if not self.app.series.reference_bank and not self.app.series.reference:
            self.add_references_to_pool()
            if not self.app.series or not self.app.series.reference_bank:
                return
        mapping = self.app.series.auto_match_references(self.app.photos)
        if 0 <= self.app.idx < len(self.app.photos):
            cur_ph = self.app.photos[self.app.idx]
            self.app.params = self.app.series.params_for(cur_ph)
            self.app._refresh_photo_ref_menu()
            self.app._sync_photo_reference()
        self.app._refresh_strip_marks()
        self.app._set_status(f"Auto-dopasowano wzorce z puli ({len(self.app.series.reference_bank)}) do {len(mapping)} zdjęć w serii.")
        self.refresh()

    def assign_pool_reference_by_name(self, name: str):
        if not self.app.series or name.startswith("("):
            return
        match = next((b for b in self.app.series.reference_bank if b.name == name), None)
        if match is not None:
            self._assign_ref_to_targets(str(match), cc=False)

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

    def _refresh_folder_tree(self):
        for w in self.folder_widgets:
            w.destroy()
        self.folder_widgets.clear()
        ser = self.app.series
        if ser is None:
            return
        cur_dir = ser.folder.resolve()
        folders = Series.discover_workspace_folders(cur_dir, self.app.settings.output_subfolder)
        for fpath, count in folders:
            is_cur = fpath.resolve() == cur_dir
            label = f"▸ {fpath.name} ({count})"
            fb = ctk.CTkButton(
                self.folder_scroll, text=label, anchor="w", width=184, height=26, corner_radius=6,
                fg_color="#5A4120" if is_cur else T.CARD,
                hover_color=T.BTN_HOVER,
                text_color=T.ACCENT if is_cur else T.TEXT,
                font=T.font(11, "bold" if is_cur else "normal"),
                command=lambda p=fpath: self.app.open_path(p)
            )
            fb.pack(fill="x", padx=4, pady=2)
            self.folder_widgets.append(fb)

    def refresh(self):
        for w in self.card_widgets:
            w.destroy()
        self.card_widgets.clear()
        self.mod_switch.set("Biblioteka (G)")
        self._refresh_folder_tree()

        ser = self.app.series
        bank = ser.reference_bank if ser else []
        self.pool_lbl.configure(text=f"Wzorce w puli: {len(bank)}")
        pool_vals = [b.name for b in bank] if bank else ["(Wybierz wzorzec z puli)"]
        self.pool_menu.configure(values=pool_vals)
        self.pool_menu.set(pool_vals[0])

        vis = self._visible_indices()
        targets = self._target_indices()
        self.sel_lbl.configure(text=f"Zaznaczone zdjęcia: {len(targets)}  (widoczne: {len(vis)}/{len(self.app.photos)})")

        for pos, i in enumerate(vis):
            ph = self.app.photos[i]
            p = ser.params_for(ph) if ser else self.app.params
            is_sel = (i in self.selected) or (not self.selected and i == self.app.idx)
            border_c = T.ACCENT if is_sel else T.BORDER

            card = ctk.CTkFrame(self.grid_scroll, fg_color=T.CARD, corner_radius=8, border_width=2, border_color=border_c)
            card.grid(row=pos // 3, column=pos % 3, padx=6, pady=6, sticky="nsew")
            self.card_widgets.append(card)

            # Thumbnail button
            im = self.app.thumbs.get(ph)
            tbtn = ctk.CTkButton(
                card, text="", image=im, width=156, height=106,
                fg_color=T.PANEL, hover_color=T.BTN_HOVER, corner_radius=6,
                command=lambda idx=i: self.toggle_select(idx)
            )
            tbtn.pack(padx=8, pady=(8, 4), fill="x")

            # Filename + Flag + Edited dot
            edited = ser.has_override(ph) if ser else False
            fl = getattr(p, "flag", 0)
            fl_prefix = "⚑ " if fl > 0 else ("✕ " if fl < 0 else "")
            name_txt = fl_prefix + ("● " if edited else "") + (ph.name if len(ph.name) <= 22 else ph.name[:19] + "…")
            name_col = "#72CF88" if fl > 0 else ("#E06868" if fl < 0 else (T.ACCENT if edited else T.TEXT))
            T.label(card, name_txt, 12, name_col, weight="bold").pack(anchor="w", padx=8)

            # Reference badge (highlighted in warm amber when photo has its own reference)
            has_own_ref = bool(p.reference_path and p.reference_path != "__none__")
            if p.reference_path == "__none__":
                ref_txt = "Tryb: Samodzielne (bez wzorca)"
            elif has_own_ref:
                prefix = "♦ CC24: " if p.ref_colorchecker else "♦ Własny: "
                ref_txt = prefix + Path(p.reference_path).name[:20]
            elif ser and ser.reference:
                ref_txt = "Seria: " + ser.reference.name[:20]
            else:
                ref_txt = "Tryb: Samodzielny edytor"
            T.label(card, ref_txt, 11, T.ACCENT if has_own_ref else T.MUTED).pack(anchor="w", padx=8, pady=(1, 4))

            # Bottom row: Flag + Stars + 1-Click Per-Photo Reference button ('♦ Wzorzec…') + Edit button
            bot = ctk.CTkFrame(card, fg_color=T.CARD)
            bot.pack(fill="x", padx=8, pady=(0, 8))
            fl_btn = T.button(bot, "⚑" if fl >= 0 else "✕", lambda idx=i: self.toggle_photo_flag(idx), width=26)
            fl_btn.configure(height=24, font=T.font(11),
                             border_color="#72CF88" if fl > 0 else ("#E06868" if fl < 0 else T.BORDER),
                             text_color="#72CF88" if fl > 0 else ("#E06868" if fl < 0 else T.MUTED))
            fl_btn.pack(side="left", padx=(0, 3))

            stars_txt = "★" * p.rating + "☆" * (5 - p.rating)
            s_btn = T.button(bot, stars_txt, lambda idx=i, cur=p.rating: self.set_photo_rating(idx, (cur + 1) % 6), width=58)
            s_btn.configure(height=24, font=T.font(10), text_color=T.ACCENT if p.rating > 0 else T.MUTED)
            s_btn.pack(side="left")

            e_btn = T.button(bot, "Edytuj (D)", lambda idx=i: self.open_in_develop(idx), accent=(i == self.app.idx), width=64)
            e_btn.configure(height=24, font=T.font(10))
            e_btn.pack(side="right")

            r_btn = T.button(bot, "♦ Wzorzec…", lambda idx=i: self.assign_ref_for_single_photo(idx), width=68)
            r_btn.configure(height=24, font=T.font(10))
            r_btn.pack(side="right", padx=2)
