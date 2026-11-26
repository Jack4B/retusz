"""Retusz main window (mockup A) + Porównaj view (mockup B)."""
from __future__ import annotations
import queue, threading, traceback, sys
from pathlib import Path
from tkinter import filedialog, messagebox
import numpy as np, cv2
import customtkinter as ctk
from PIL import Image
from . import theme as T
from .imageview import ImageView
from .compare import CompareView
from .library import LibraryView
from .histview import HistView
from .dialogs import SettingsDialog, ModelDialog, ProgressDialog, ExportAsDialog
from .. import imageio, models, batch, export, colorman as CM, presets as PR, __version__
from ..reference import Reference
from ..settings import Settings
from ..sidecar import Series
from ..engine.pipeline import Engine, Params
from ..engine import render as R, color as C, device as DEV, modflows as MF, relight as RL
from ..engine import histo as HI, wb as WB, geometry as G, colorchecker as CC, detail as DET

ENGINE_LABEL = {"ai": "AI (B6)", "ai_light": "AI lekki (B0)", "classic": "bez AI (IDT)"}
CANVAS_MODES = ["Wynik", "Przed | Po", "Podział", "Z wzorcem"]
SLIDER_TABS = ["Światło i Kolor", "Portret i 3D", "Detal i Ostrość", "Wzorzec (AI)"]
SLIDER_GROUPS = {
    SLIDER_TABS[0]: [  # (key, label, lo, hi, kind, row, col) — 2-column vertical Lightroom stack
        ("temp", "Temperatura (WB)", -100, 100, "num", 0, 0),
        ("tint", "Odcień (Tint WB)", -100, 100, "num", 0, 1),
        ("exposure", "Ekspozycja", -200, 200, "ev", 1, 0),
        ("contrast", "Kontrast", -100, 100, "num", 1, 1),
        ("highlights", "Światła", -100, 100, "num", 2, 0),
        ("shadows", "Cienie", -100, 100, "num", 2, 1),
        ("whites", "Biele", -100, 100, "num", 3, 0),
        ("blacks", "Czernie", -100, 100, "num", 3, 1),
        ("vibrance", "Jaskrawość", -100, 100, "num", 4, 0),
        ("saturation", "Nasycenie", -100, 100, "num", 4, 1),
    ],
    SLIDER_TABS[1]: [
        ("db", "Dodge & Burn", 0, 100, "pct", 0, 0),
        ("plastic", "Plastyka 3D", 0, 100, "pct", 0, 1),
        ("light_x", "Światło 3D: Lewo↔Prawo", -100, 100, "num", 1, 0),
        ("light_y", "Światło 3D: Dół↔Góra", -100, 100, "num", 1, 1),
        ("bg_exposure", "Separacja tła 3D", -100, 100, "num", 2, 0),
        ("skin_hue", "Odcień skóry", -100, 100, "num", 2, 1),
        ("skin_cast", "Czysta skóra", 0, 100, "pct", 3, 0),
        ("skin_even", "Równanie skóry", 0, 100, "pct", 3, 1),
        ("deshine", "Redukcja połysku", 0, 100, "pct", 4, 0),
        ("blemish", "Niedoskonałości", 0, 100, "pct", 4, 1),
        ("eyes_teeth", "Oczy i zęby", 0, 100, "pct", 5, 0),
    ],
    SLIDER_TABS[2]: [
        ("sharpen_amount", "Ostrość", 0, 100, "pct", 0, 0),
        ("texture", "Struktura", -100, 100, "num", 0, 1),
        ("clarity", "Przejrzystość (LLF)", -100, 100, "num", 1, 0),
        ("denoise", "Odszumianie OKLab", 0, 100, "pct", 1, 1),
        ("brightness", "Jasność (Półtony)", -100, 100, "num", 2, 0),
        ("dehaze", "Zamglenie (Dehaze)", -100, 100, "num", 2, 1),
        ("wide", "Pełne barwy", 0, 100, "pct", 3, 0),
        ("vignette", "Winieta", 0, 100, "pct", 3, 1),
    ],
    SLIDER_TABS[3]: [
        ("strength", "Siła kolorów wzorca", 0, 100, "pct", 0, 0),
        ("skin", "Ochrona skóry", 0, 100, "pct", 0, 1),
        ("mask_hue_tol", "Maska AI: Odcień skóry", -100, 100, "num", 1, 0),
        ("mask_protect", "Maska AI: Oczy i usta", 0, 100, "pct", 1, 1),
    ],
}

# Wizualne podpowiedzi kierunkowe pod suwakami: (lewy_opis, prawy_opis, lewy_kolor_hex, prawy_kolor_hex)
SLIDER_HINTS: dict[str, tuple[str, str, str, str]] = {
    "temp": ("◀ Chłodny (Błękit)", "Ciepły (Bursztyn) ▶", "#64B5F6", "#FFB74D"),
    "tint": ("◀ Zieleń", "Magenta (Róż) ▶", "#81C784", "#F06292"),
    "exposure": ("◀ Ciemniej (-EV)", "Jaśniej (+EV) ▶", "#9A9A9A", "#E0E0E0"),
    "contrast": ("◀ Miękki / Płaski", "Głęboki kontrast ▶", "#9A9A9A", "#E0E0E0"),
    "highlights": ("◀ Odzyskaj światła", "Rozjaśnij światła ▶", "#90CAF9", "#FFE082"),
    "shadows": ("◀ Pogłęb cienie", "Otwórz cienie ▶", "#9A9A9A", "#FFE082"),
    "whites": ("◀ Punkt bieli −", "Punkt bieli + ▶", "#9A9A9A", "#E0E0E0"),
    "blacks": ("◀ Głęboka czerń", "Matowa czerń ▶", "#9A9A9A", "#E0E0E0"),
    "vibrance": ("◀ Stonowane", "Żywe (chroni skórę) ▶", "#9A9A9A", "#F5A33B"),
    "saturation": ("◀ Monochrom", "Pełne nasycenie ▶", "#9A9A9A", "#F5A33B"),
    "db": ("0% (wył.)", "Rzeźba i gładkość ▶", "#9A9A9A", "#F5A33B"),
    "plastic": ("0% (płasko)", "Bryła 3D ▶", "#9A9A9A", "#F5A33B"),
    "light_x": ("◀ z Lewej (-100)", "z Prawej (+100) ▶", "#90CAF9", "#FFE082"),
    "light_y": ("◀ z Dołu (blenda)", "z Góry (kluczowe) ▶", "#B0BEC5", "#FFE082"),
    "bg_exposure": ("◀ Przyciemnij tło", "Rozjaśnij tło ▶", "#78909C", "#FFF59D"),
    "skin_hue": ("◀ Róż / Czerwień", "Złoty / Żółty ▶", "#E57373", "#FFD54F"),
    "skin_cast": ("0%", "Usuń rumień i szarość ▶", "#F48FB1", "#FFD54F"),
    "skin_even": ("0%", "Wyrównaj plamy skóry ▶", "#9A9A9A", "#FFD54F"),
    "deshine": ("0%", "Usuń tłusty połysk ▶", "#9A9A9A", "#90CAF9"),
    "blemish": ("0%", "Usuń wypryski (pory OK) ▶", "#9A9A9A", "#81C784"),
    "eyes_teeth": ("0%", "Tęczówka i biel zębów ▶", "#9A9A9A", "#E0E0E0"),
    "sharpen_amount": ("0%", "Ostrość (0% halo) ▶", "#9A9A9A", "#F5A33B"),
    "texture": ("◀ Wygładź detal", "Podkreśl mikrodetal ▶", "#90CAF9", "#FFB74D"),
    "clarity": ("◀ Miękki kontur", "Kontrast lokalny LLF ▶", "#9A9A9A", "#E0E0E0"),
    "denoise": ("0%", "Usuń szum koloru/cieni ▶", "#9A9A9A", "#81C784"),
    "brightness": ("◀ Ciemniej półtony", "Jaśniej półtony ▶", "#9A9A9A", "#FFE082"),
    "dehaze": ("◀ Mglisty blask", "Usuń mgłę / Głębia ▶", "#B0BEC5", "#64B5F6"),
    "wide": ("0% (sRGB)", "Szeroki gamut ▶", "#9A9A9A", "#F5A33B"),
    "vignette": ("0%", "Przyciemnij brzegi ▶", "#9A9A9A", "#E0E0E0"),
    "strength": ("0% (oryginał)", "100% (pełny wzorzec) ▶", "#9A9A9A", "#F5A33B"),
    "skin": ("0%", "100% (chroń karnację) ▶", "#9A9A9A", "#FFD54F"),
    "mask_hue_tol": ("◀ Wąska (twarz)", "Szeroka (szyja/cień) ▶", "#90CAF9", "#FFB74D"),
    "mask_protect": ("0%", "Chroń oczy i usta ▶", "#9A9A9A", "#81C784"),
}

PRO_TIPS: dict[str, str] = {
    "temp": "Temperatura (Bradford CAT): w lewo (−) chłodzi zdjęcie w stronę błękitu, w prawo (+) ociepla w stronę złotego bursztynu.",
    "tint": "Odcień (Tint): w lewo (−) koryguje magentę dodając zieleni, w prawo (+) usuwa zafarb zielony/fluorescencyjny dodając magenty.",
    "exposure": "Ekspozycja (EV w świetle liniowym): zmienia jasność jak czas naświetlania w aparacie z filmową ochroną świateł.",
    "contrast": "Kontrast (OKLab): krzywa S wokół średniej szarości bez przesuwania odcienia skóry.",
    "highlights": "Światła: w lewo (−) odzyskuje detale w jasnych partiach (suknia, chmury, czoło), w prawo (+) rozświetla jasne tony.",
    "shadows": "Cienie: w prawo (+) wyciąga detale z cieni bez spłaszczania czerni, w lewo (−) pogłębia cienie.",
    "whites": "Biele: reguluje górny punkt bieli i blask najjaśniejszych partii.",
    "blacks": "Czernie: w lewo (−) pogłębia punkt czerni, w prawo (+) daje miękki, filmowy mat w cieniach.",
    "vibrance": "Jaskrawość (OKLab): inteligentnie podbija nienasycone barwy tła i ubrań, chroniąc skórę przed przesyceniem.",
    "saturation": "Nasycenie (OKLab): równomiernie zmienia intensywność wszystkich barw (0% przesunięcia odcienia).",
    "db": "Dodge & Burn v2: automatycznie wyrównuje mikro-cienie pod oczami i na policzkach przy 100% zachowaniu tekstury porów.",
    "plastic": "Plastyka 3D: modeluje trójwymiarową bryłę twarzy na mapie głębi Depth Anything V2 (rozświetla stronę światła, pogłębia cień).",
    "light_x": "Światło 3D (Lewo ↔ Prawo): przesuwa wirtualny softbox 3D w poziomie — w lewo (−) światło z lewej, w prawo (+) światło z prawej.",
    "light_y": "Światło 3D (Dół ↔ Góra): przesuwa wirtualny softbox 3D w pionie — w dół (−) odbicie od blendy, w górę (+) światło górne/motylkowe.",
    "bg_exposure": "Separacja tła 3D: w lewo (−) przyciemnia tło za postacią (efekt studyjny), w prawo (+) rozjaśnia tło. Twarz pozostaje nietknięta.",
    "skin_hue": "Odcień skóry (OKLCh): w lewo (−) przesuwa karnację w stronę różu/czerwieni, w prawo (+) w stronę złotego/brzoskwiniowego odcienia.",
    "skin_cast": "Czysta skóra: redukuje czerwony rumień (hemoglobinę) oraz ziemiste/sine zabarwienia cieni, sprowadzając skórę do osi 52° OKLCh.",
    "skin_even": "Równanie skóry: wyrównuje niskoczęstotliwościowe plamy barwne między twarzą, szyją i dłońmi bez rozmywania porów.",
    "deshine": "Redukcja połysku (De-Shine SUV): fizyczna separacja odbicia lustrzanego usuwająca tłusty błysk z czoła i nosa z odzyskaniem porów.",
    "blemish": "Niedoskonałości: dwupasmowy retusz punktowych wyprysków i przebarwień z pełnym zachowaniem naturalnej struktury porów.",
    "eyes_teeth": "Oczy i zęby: wzmacnia mikrokontrast tęczówki oraz naturalnie oczyszcza białka oczu i zęby na podstawie punktów anatomicznych.",
    "sharpen_amount": "Ostrość: wyostrzanie na żywo (100% podgląd i eksport) w wybranym trybie (Portret / Dekonwolucja RL / Krawędzie / Soft Focus).",
    "texture": "Struktura: w prawo (+) wydobywa drobny mikrodetal (rzęsy, tkaniny), w lewo (−) delikatnie zmiękcza wysokie częstotliwości.",
    "clarity": "Przejrzystość (Fast LLF): wieloskalowa piramida Laplace'a wzmacniająca kontrast lokalny bez efektu halo na krawędziach.",
    "denoise": "Odszumianie OKLab: usuwa kolorowy szum matrycy (chroma) i ziarnistość w cieniach z zachowaniem ostrych krawędzi.",
    "brightness": "Jasność (Półtony w stylu Capture One): rozjaśnia lub przyciemnia półtony bez ruszania czystej czerni i czystej bieli.",
    "dehaze": "Zamglenie (Dehaze): w prawo (+) usuwa mgiełkę i flarę pod światło, w lewo (−) dodaje miękką, powietrzną poświatę.",
    "wide": "Pełne barwy: symuluje głębię barw monitora szeroko-gamutowego (Adobe RGB) w przestrzeni docelowej.",
    "vignette": "Winieta: przyciemnia brzegi kadru poza postacią, skupiając wzrok na portrecie.",
    "strength": "Siła kolorów wzorca: określa w ilu procentach zastosować dopasowanie barwne do wybranego wzorca lub tablicy ColorChecker.",
    "skin": "Ochrona skóry: chroni naturalną jasność i nasycenie karnacji przed zbyt agresywnym gradingiem wzorca.",
    "mask_hue_tol": "Maska AI — Odcień skóry (Mahalanobis): w lewo (−) zawęża maskę do czystego odcienia policzków, w prawo (+) rozszerza na szyję i cienie.",
    "mask_protect": "Maska AI — Oczy i usta: wyklucza z maski skóry oczy, rzęsy oraz czerwień ust na podstawie 5 punktów anatomicznych YuNet.",
}

_TAB_ALIASES = {
    "Wzorzec i Portret": SLIDER_TABS[1],
    "Portret i Plastyka": SLIDER_TABS[1],
    "Portret i 3D": SLIDER_TABS[1],
    "Światło i Kolor": SLIDER_TABS[0],
    "Detal i Ostrość": SLIDER_TABS[2],
    "Wzorzec (AI)": SLIDER_TABS[3],
    "▸ Wzorzec (AI)": SLIDER_TABS[3],
}
_DEFAULTS = Params()
_COPY_KEYS = (
    "strength", "skin", "exposure", "contrast", "brightness", "dehaze", "shadows", "highlights", "whites", "blacks",
    "vignette", "db", "plastic", "clarity", "texture", "sharpen_amount", "sharpen_mode", "denoise",
    "light_angle", "light_x", "light_y", "bg_exposure", "saturation", "vibrance",
    "skin_even", "skin_cast", "skin_hue", "deshine", "blemish", "eyes_teeth", "mask_hue_tol", "mask_protect",
    "wide", "temp", "tint",
)
SLIDER_AREA_H = 414
TOOLS = ["Pędzel D&B", "Kadr"]
TOOL_W, TOOL_H = 382, 134
PHOTO_TYPES = [(
    "Zdjęcia i RAW",
    "*.jpg *.jpeg *.png *.tif *.tiff *.webp "
    "*.nef *.NEF *.nrw *.NRW *.cr2 *.CR2 *.cr3 *.CR3 *.arw *.ARW *.dng *.DNG *.raf *.RAF *.orf *.ORF *.rw2 *.RW2 *.pef *.PEF"
)]
BRUSH_MIN, BRUSH_MAX = 5, 600   # brush radius in original-image pixels
BRUSH_MODES = ["Wył.", "Rozjaśnij", "Przyciemnij"]

class RetuszApp(ctk.CTk):
    def __init__(self, path=None, reference=None):
        T.setup(); super().__init__(fg_color=T.BG)
        self.title(f"Retusz {__version__} – Edytor Portretowy"); self.geometry("1440x880"); self.minsize(1120, 700)
        self.settings = Settings.load(); DEV.MODE = self.settings.device
        if sys.platform == "win32":
            try:
                from ..paths import resource_dir; self.iconbitmap(str(resource_dir()/"retusz.ico"))
            except Exception: pass
        self.engine: Engine | None = None; self.engine_name = ""; self.device = ""
        self.series: Series | None = None; self.photos: list[Path] = []; self.idx = -1
        self.ref_analysis = None; self.ref_prev = None; self.reference: Reference | None = None
        self.ref_enabled = bool(reference)
        self.src_icc = None; self.full_cache = None; self._detail_after = None; self._space = False; self._pan = None
        self.src_prev = None; self.src_size = None; self.fit = None; self.params = Params()
        self.result = None; self.mask_prev = None; self.thumbs = {}; self.thumb_btns = []
        self._gen = {}; self._ui = queue.Queue(); self._jobs = queue.Queue(); self._save_after = None
        self._stroke = None
        self.frame = None; self.detail_raw = None; self.show_clip = False; self.picking = False
        self.show_before = False; self._clipboard = None
        self.presets: list[PR.Preset] = PR.list_presets()
        self._undo: list[dict] = []; self._redo: list[dict] = []; self._hist_anchor = self.params.to_dict(); self._hist_after = None
        self.tool = "brush"; self.hline_on = False; self._hline = None; self._crop_drag = None; self._crop_live = None
        self._geo_after = None
        threading.Thread(target=self._worker, daemon=True).start()
        self._build(); self._poll()
        self.bind("<Left>", lambda e: self.step(-1)); self.bind("<Right>", lambda e: self.step(1))
        self.bind("<Control-s>", lambda e: self.save_current())
        self.bind("<Control-z>", lambda e: self._key(self.undo))
        self.bind("<Control-y>", lambda e: self._key(self.redo))
        self.bind("<Control-Shift-Z>", lambda e: self._key(self.redo))
        self.bind("<Control-Shift-C>", lambda e: self._key(self.copy_params))
        self.bind("<Control-Shift-V>", lambda e: self._key(self.paste_params))
        self.bind("<backslash>", lambda e: self._key(self.toggle_before))
        self.bind("<KeyPress-space>", lambda e: self._space_key(True)); self.bind("<KeyRelease-space>", lambda e: self._space_key(False))
        self.bind("<bracketleft>", lambda e: self._brush_step(-1)); self.bind("<bracketright>", lambda e: self._brush_step(1))
        self.bind("<Control-0>", lambda e: self.zoom_fit()); self.bind("<Control-1>", lambda e: self.zoom_100())
        self.bind("<Control-Shift-E>", lambda e: self.open_export_as()); self.bind("<Control-Shift-e>", lambda e: self.open_export_as())
        for k, fn in (("w", self.toggle_pick), ("c", self.toggle_crop_tool), ("j", self.toggle_clip), ("m", self.toggle_mask),
                      ("g", self.toggle_library), ("d", self.hide_library)):
            self.bind(f"<KeyPress-{k}>", lambda e, fn=fn: self._key(fn)); self.bind(f"<KeyPress-{k.upper()}>", lambda e, fn=fn: self._key(fn))
        self.bind("<Return>", lambda e: self._key(self._crop_done)); self.bind("<Escape>", lambda e: self._key(self._escape))
        self.submit(lambda: (R.warmup(), None)[1], None)
        self.setup_engine()
        if path: self.after(100, lambda: self.open_path(Path(path), Path(reference) if reference else None))
        self.after(700, self._first_run)

    # ================= layout =================
    def _build(self):
        # Clean Lightroom-style top bar divided into 3 uncluttered groups: Files (left), Modules & View (center), Export (right)
        top = ctk.CTkFrame(self, fg_color=T.BG); top.pack(fill="x", padx=12, pady=(8, 6))
        def tbtn(parent, text, cmd, w, accent=False, side="left", padx=0):
            b = T.button(parent, text, cmd, accent=accent, width=w)
            b.configure(font=T.font(12, "bold" if accent else "normal"), height=30)
            b.pack(side=side, padx=padx)
            return b
        tbtn(top, "Otwórz zdjęcie", self.open_photo, 96)
        tbtn(top, "Otwórz folder", self.open_folder, 90, padx=4)
        tbtn(top, "Ustawienia", self.open_settings, 76)

        tbtn(top, "Eksportuj serię", self.export_series, 108, accent=True, side="right")
        tbtn(top, "Eksportuj jako…", self.open_export_as, 112, accent=True, side="right", padx=4)
        tbtn(top, "Zapisz", self.save_current, 58, accent=True, side="right")

        mod_bar = ctk.CTkFrame(top, fg_color=T.BG); mod_bar.pack(side="left", padx=(10, 0))
        tbtn(mod_bar, "Biblioteka (G)", self.show_library, 92, padx=(0, 3))
        tbtn(mod_bar, "Porównaj", self.show_compare, 68, padx=(0, 3))
        tbtn(mod_bar, "Grading z AI", self.open_transfer, 86)

        vb = ctk.CTkFrame(top, fg_color=T.BG); vb.pack(side="left", padx=(8, 0))
        self.canvas_mode = T.segmented(vb, CANVAS_MODES, command=self._set_canvas_mode)
        self.canvas_mode.configure(font=T.font(12), height=30)
        self.canvas_mode.set("Przed | Po" if not self.ref_enabled else "Z wzorcem")
        self.canvas_mode.pack(side="left")

        zb = ctk.CTkFrame(top, fg_color=T.BG); zb.pack(side="left", padx=(6, 0))
        tbtn(zb, "Dopasuj", self.zoom_fit, 56)
        tbtn(zb, "100%", self.zoom_100, 42, padx=3)
        self.before_btn = tbtn(zb, "Przed/Po", self.toggle_before, 64, padx=(0, 3))
        zbox = ctk.CTkFrame(zb, fg_color=T.BG, width=38, height=28); zbox.pack(side="left"); zbox.pack_propagate(False)
        self.zoom_lbl = T.label(zbox, "", 12, T.MUTED, anchor="w"); self.zoom_lbl.pack(fill="both", expand=True)

        self.main = ctk.CTkFrame(self, fg_color=T.BG); self.main.pack(fill="both", expand=True)
        self.compare = CompareView(self, self.hide_compare, self.save_current, self._compare_full)
        self.library = LibraryView(self, self)

        # Left Filmstrip (compact)
        side = ctk.CTkFrame(self.main, fg_color=T.BG, width=154); side.pack(side="left", fill="y", padx=(12, 0))
        self.series_lbl = T.label(side, "Seria", 12, weight="bold"); self.series_lbl.pack(anchor="w", pady=(0, 4))
        self.strip = ctk.CTkScrollableFrame(side, fg_color=T.PANEL, width=136, corner_radius=8)
        self.strip.pack(fill="y", expand=True)

        # Right-Side Lightroom / Capture One Develop Inspector Dock
        dock = ctk.CTkFrame(self.main, fg_color=T.CARD, width=414, corner_radius=10, border_width=1, border_color=T.BORDER)
        dock.pack(side="right", fill="y", padx=(0, 12), pady=(0, 2))
        dock.pack_propagate(False)

        # Central Viewport (Full vertical height for portraits!)
        body = ctk.CTkFrame(self.main, fg_color=T.BG); body.pack(side="left", fill="both", expand=True, padx=10)
        self.imgs = ctk.CTkFrame(body, fg_color=T.BG); self.imgs.pack(fill="both", expand=True)
        self.imgs.grid_rowconfigure(1, weight=1)
        self.src_hdr = T.label(self.imgs, "Przed (Źródło)", 13)
        self.res_hdr = T.label(self.imgs, "Wynik", 13)
        self.ref_hdr = T.label(self.imgs, "Wzorzec (opcjonalny)", 13)
        self.view_src = ImageView(self.imgs, "Otwórz zdjęcie lub folder")
        self.view_res = ImageView(self.imgs, "Otwórz zdjęcie lub folder — pełny edytor OKLab / Plastyka 3D / D&B v2.\nSuwaki i narzędzia znajdują się w prawym doku inspektora.")
        self.view_ref = ImageView(self.imgs, "Opcjonalny wzorzec (AI Color Match)\nWybierz wzorzec w zakładce „Wzorzec (AI)” po prawej")
        self._apply_canvas_layout()

        self.view_res.bind("<ButtonPress-1>", self._brush_press); self.view_res.bind("<B1-Motion>", self._brush_move)
        self.view_res.bind("<ButtonRelease-1>", self._brush_release); self.view_res.bind("<Motion>", self._brush_cursor)
        self.view_res.bind("<Leave>", lambda e: self.view_res.delete("cursor"))
        self.view_res.on_redraw = self._draw_crop_overlay
        for v in (self.view_src, self.view_res):
            v.on_view_change = self._view_changed
            v.bind("<MouseWheel>", lambda e, v=v: self._wheel(v, e, 1 if e.delta > 0 else -1))
            v.bind("<Button-4>", lambda e, v=v: self._wheel(v, e, 1)); v.bind("<Button-5>", lambda e, v=v: self._wheel(v, e, -1))
            v.bind("<ButtonPress-2>", lambda e, v=v: self._pan_start(v, e)); v.bind("<B2-Motion>", lambda e, v=v: self._pan_move(v, e))
            v.bind("<ButtonRelease-2>", lambda e: self._pan_end())
        self.view_src.bind("<ButtonPress-1>", lambda e: self._pan_start(self.view_src, e))
        self.view_src.bind("<B1-Motion>", lambda e: self._pan_move(self.view_src, e)); self.view_src.bind("<ButtonRelease-1>", lambda e: self._pan_end())

        # ---- Right Dock Section 1: Histogram + OKLCh Skin Vectorscope (52°) + Quick History & Series status ----
        top_dock = ctk.CTkFrame(dock, fg_color=T.CARD)
        top_dock.pack(fill="x", padx=8, pady=(6, 2))
        h_left = ctk.CTkFrame(top_dock, fg_color=T.CARD)
        h_left.pack(side="left")
        self.hist = HistView(h_left, on_toggle=self.toggle_clip, width=204, height=68); self.hist.pack(anchor="w")
        hb = ctk.CTkFrame(h_left, fg_color=T.CARD, width=204, height=16); hb.pack(anchor="w", pady=(1, 0)); hb.pack_propagate(False)
        self.clip_lbl = T.label(hb, "Cienie – · Światła –", 10, T.MUTED, anchor="w"); self.clip_lbl.pack(fill="both", expand=True)
        sw_row = ctk.CTkFrame(h_left, fg_color=T.CARD); sw_row.pack(anchor="w", fill="x", pady=(1, 0))
        self.clip_sw = ctk.CTkSwitch(sw_row, text="Przepalenia (J)", progress_color=T.ACCENT, text_color=T.TEXT, font=T.font(10),
                                     width=102, switch_width=24, switch_height=12, command=lambda: self.toggle_clip(self.clip_sw.get() == 1))
        self.clip_sw.pack(side="left")
        self.mask_sw = ctk.CTkSwitch(sw_row, text="Maska AI (M)", progress_color=T.ACCENT, text_color=T.TEXT, font=T.font(10),
                                     width=96, switch_width=24, switch_height=12, command=lambda: self.toggle_mask(self.mask_sw.get() == 1))
        if self.settings.show_mask:
            self.mask_sw.select()
        self.mask_sw.pack(side="left", padx=(4, 0))

        h_right = ctk.CTkFrame(top_dock, fg_color=T.CARD)
        h_right.pack(side="right", fill="both", expand=True, padx=(6, 0))
        ob = ctk.CTkFrame(h_right, fg_color=T.CARD); ob.pack(anchor="w", fill="x")
        T.label(ob, "Zdjęcie:", 11).pack(side="left")
        self.override_lbl = T.label(ob, "serii", 11, T.MUTED); self.override_lbl.pack(side="left", padx=(4, 0))
        s_row = ctk.CTkFrame(h_right, fg_color=T.CARD); s_row.pack(fill="x", pady=(3, 2))
        b1 = T.button(s_row, "Przywróć z serii", self.reset_to_series, width=90); b1.configure(height=23, font=T.font(10)); b1.pack(side="left")
        b2 = T.button(s_row, "Ustaw jako serię", self.make_series_default, width=92); b2.configure(height=23, font=T.font(10)); b2.pack(side="right")

        act = ctk.CTkFrame(h_right, fg_color=T.CARD); act.pack(fill="x", pady=(3, 0))
        def sbtn(text, cmd, w):
            b_ = T.button(act, text, cmd, width=w); b_.configure(height=23, font=T.font(10), corner_radius=5); b_.pack(side="left", padx=(0, 2)); return b_
        self.undo_btn = sbtn("Cofnij", self.undo, 36)
        self.redo_btn = sbtn("Ponów", self.redo, 36)
        sbtn("Kopiuj", self.copy_params, 36)
        self.paste_btn = sbtn("Wklej", self.paste_params, 34)
        sbtn("Zeruj", self.reset_adjustments, 34)

        # ---- Right Dock Section 2: Develop Tabs & Sliders with Colored Directional Hint Bars ----
        sl = ctk.CTkFrame(dock, fg_color=T.CARD)
        sl.pack(fill="x", padx=6, pady=(2, 2))
        self.slider_tabs = T.segmented(sl, SLIDER_TABS, command=self._slider_tab, dynamic_resizing=False, width=398)
        self.slider_tabs.configure(font=T.font(11), height=28)
        self.slider_tabs.set(SLIDER_TABS[0]); self.slider_tabs.pack(fill="x", pady=(0, 2))

        sl_area = ctk.CTkFrame(sl, fg_color=T.CARD, height=SLIDER_AREA_H)
        sl_area.pack(fill="x", expand=True); sl_area.pack_propagate(False)
        self.slider_frames = {}
        self.sliders = {}
        def value_box(parent, width):
            box = ctk.CTkFrame(parent, fg_color=T.CARD, width=width, height=17); box.pack_propagate(False)
            lab = T.label(box, "", 11, T.MUTED, anchor="e"); lab.pack(fill="both", expand=True); return box, lab
        for tname in SLIDER_TABS:
            f = ctk.CTkFrame(sl_area, fg_color=T.CARD)
            f.place(x=0, y=0, relwidth=1, relheight=1)
            for c in range(2): f.grid_columnconfigure(c, weight=1, uniform=f"sl_{tname}")
            self.slider_frames[tname] = f
            for key, text, lo, hi, kind, r_, c_ in SLIDER_GROUPS[tname]:
                cell = ctk.CTkFrame(f, fg_color=T.CARD); cell.grid(row=r_, column=c_, sticky="ew", padx=4, pady=1)
                cell.grid_columnconfigure(0, weight=1)
                lbl = T.label(cell, text, 11, anchor="w", weight="bold"); lbl.grid(row=0, column=0, sticky="w")
                box, val = value_box(cell, 54 if kind == "ev" else 38); box.grid(row=0, column=1, sticky="e")
                s = T.slider(cell, lo, hi, number_of_steps=hi-lo, width=170, command=lambda v, k=key: self._slider(k, v))
                s.configure(height=12)
                s.grid(row=1, column=0, columnspan=2, sticky="ew")
                if key in SLIDER_HINTS:
                    l_txt, r_txt, l_col, r_col = SLIDER_HINTS[key]
                    hrow = ctk.CTkFrame(cell, fg_color=T.CARD, height=12)
                    hrow.grid(row=2, column=0, columnspan=2, sticky="ew")
                    T.label(hrow, l_txt, 9, l_col, anchor="w").pack(side="left")
                    T.label(hrow, r_txt, 9, r_col, anchor="e").pack(side="right")
                for w_ in (cell, lbl, val, s):
                    w_.bind("<Double-Button-1>", lambda e, k=key: self._reset_slider(k))
                    w_.bind("<Enter>", lambda e, k=key: self._set_pro_tip(k))
                self.sliders[key] = (s, val, kind)
        for alias, target in _TAB_ALIASES.items():
            self.slider_frames[alias] = self.slider_frames[target]

        # Tab 0: White Balance bar in row 5, cols 0-1
        f0 = self.slider_frames[SLIDER_TABS[0]]
        wbc = ctk.CTkFrame(f0, fg_color=T.CARD); wbc.grid(row=5, column=0, columnspan=2, sticky="ew", padx=4, pady=(4, 1))
        T.label(wbc, "Balans bieli (Bradford):", 11, T.MUTED, anchor="w").pack(side="left", padx=(0, 4))
        self.pick_btn = T.button(wbc, "Pipeta (W)", self.toggle_pick, width=82); self.pick_btn.configure(height=23, font=T.font(10))
        self.pick_btn.pack(side="left", padx=(0, 3))
        b = T.button(wbc, "Balans → seria", self.sync_wb, width=96); b.configure(height=23, font=T.font(10)); b.pack(side="left", padx=(0, 3))
        b = T.button(wbc, "Zeruj WB", self.reset_wb, width=62); b.configure(height=23, font=T.font(10)); b.pack(side="left")

        # Tab 1: 3D Light Auto switch + Presets selector in row 5, col 1 (plus hidden legacy self.light angle slider for gui_check)
        f1 = self.slider_frames[SLIDER_TABS[1]]
        pcell = ctk.CTkFrame(f1, fg_color=T.CARD); pcell.grid(row=5, column=1, sticky="ew", padx=4, pady=1)
        pcell.grid_columnconfigure(0, weight=1)
        l_hdr = ctk.CTkFrame(pcell, fg_color=T.CARD); l_hdr.grid(row=0, column=0, columnspan=3, sticky="ew")
        self.light_auto = ctk.CTkSwitch(l_hdr, text="Auto 3D", progress_color=T.ACCENT, text_color=T.TEXT, width=68,
                                        switch_width=24, switch_height=12, font=T.font(10), command=self._light_auto)
        self.light_auto.pack(side="left")
        _, self.light_val = value_box(l_hdr, 42)
        pb_save = T.button(l_hdr, "+Preset", self.save_custom_preset_dialog, width=48)
        pb_save.configure(height=18, font=T.font(9)); pb_save.pack(side="right")
        pb_ser = T.button(l_hdr, "→seria", self.preset_to_series, width=42)
        pb_ser.configure(height=18, font=T.font(9)); pb_ser.pack(side="right", padx=(0, 2))
        self.preset_menu = ctk.CTkOptionMenu(
            pcell, values=[p.name for p in self.presets], command=self.apply_preset_by_name,
            width=170, height=22, fg_color=T.BTN, button_color=T.BTN, button_hover_color=T.BTN_HOVER,
            text_color=T.TEXT, font=T.font(10), dropdown_font=T.font(11), corner_radius=6
        )
        self.preset_menu.set("Portret naturalny"); self.preset_menu.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(1, 0))
        # Alias self.light to light_x slider so legacy callers checking app.light geometry work seamlessly
        self.light = self.sliders["light_x"][0]

        # Tab 2: Detail & Sharpening ('Detal i Ostrość'): Sharpening mode selector at (4, 0) + Star Rating at (4, 1)
        f_det = self.slider_frames[SLIDER_TABS[2]]
        sh_cell = ctk.CTkFrame(f_det, fg_color=T.CARD); sh_cell.grid(row=4, column=0, sticky="ew", padx=4, pady=2)
        sh_cell.grid_columnconfigure(0, weight=1)
        T.label(sh_cell, "Tryb wyostrzania", 11, anchor="w", weight="bold").grid(row=0, column=0, sticky="w")
        self.sharpen_mode_menu = ctk.CTkOptionMenu(
            sh_cell, values=list(DET.SHARPEN_MODES.values()), command=self._set_sharpen_mode,
            width=170, height=23, fg_color=T.BTN, button_color=T.BTN, button_hover_color=T.BTN_HOVER,
            text_color=T.TEXT, font=T.font(10), dropdown_font=T.font(11), corner_radius=6
        )
        self.sharpen_mode_menu.set(DET.SHARPEN_MODES["portrait"])
        self.sharpen_mode_menu.grid(row=1, column=0, sticky="ew", pady=(2, 0))

        st_cell = ctk.CTkFrame(f_det, fg_color=T.CARD); st_cell.grid(row=4, column=1, sticky="ew", padx=4, pady=2)
        T.label(st_cell, "Ocena zdjęcia (★)", 11, anchor="w", weight="bold").grid(row=0, column=0, columnspan=6, sticky="w")
        self.star_btns = []
        for st_i in range(6):
            txt = "0" if st_i == 0 else f"{st_i}★"
            sb = T.button(st_cell, txt, lambda s=st_i: self._set_current_rating(s), width=24)
            sb.configure(height=23, font=T.font(10), corner_radius=5)
            sb.grid(row=1, column=st_i, padx=(0, 2), pady=(2, 0))
            self.star_btns.append(sb)
        T.label(f_det, "Wyostrzanie RL / Clamped USM (0% halo) i odszumianie OKLab działa na żywo.",
                10, T.MUTED, anchor="w").grid(row=5, column=0, columnspan=2, sticky="w", padx=4, pady=(3, 1))

        # Tab 3: Reference Match, Per-Photo Reference, AI Mask Refinement & ColorChecker section
        f2 = self.slider_frames[SLIDER_TABS[3]]
        rc = ctk.CTkFrame(f2, fg_color=T.CARD); rc.grid(row=2, column=0, sticky="ew", padx=4, pady=3)
        T.label(rc, "Wzorzec serii (AI)", 11, anchor="w", weight="bold").grid(row=0, column=0, columnspan=2, sticky="w")
        b_ref = T.button(rc, "Wybierz…", self.choose_reference, accent=True, width=86); b_ref.configure(height=24, font=T.font(11, "bold"))
        b_ref.grid(row=1, column=0, sticky="w", pady=(2, 0))
        self.clear_ref_btn = T.button(rc, "Wyłącz", self.clear_reference, width=68); self.clear_ref_btn.configure(height=24, font=T.font(11))
        self.clear_ref_btn.grid(row=1, column=1, sticky="w", padx=(4, 0), pady=(2, 0))

        cc_cell = ctk.CTkFrame(f2, fg_color=T.CARD); cc_cell.grid(row=2, column=1, sticky="ew", padx=4, pady=3)
        T.label(cc_cell, "ColorChecker (24)", 11, anchor="w", weight="bold").grid(row=0, column=0, columnspan=2, sticky="w")
        b_cc = T.button(cc_cell, "Z pliku…", self.choose_colorchecker, width=82); b_cc.configure(height=24, font=T.font(11))
        b_cc.grid(row=1, column=0, sticky="w", pady=(2, 0))
        b_cc_cur = T.button(cc_cell, "Z kadru", self.calibrate_current_colorchecker, width=74); b_cc_cur.configure(height=24, font=T.font(11))
        b_cc_cur.grid(row=1, column=1, sticky="w", padx=(4, 0), pady=(2, 0))

        pr_cell = ctk.CTkFrame(f2, fg_color=T.CARD); pr_cell.grid(row=3, column=0, columnspan=2, sticky="ew", padx=4, pady=3)
        pr_cell.grid_columnconfigure(0, weight=1)
        T.label(pr_cell, "Wzorzec TEGO zdjęcia (indywidualny / Pula Wzorców)", 11, anchor="w", weight="bold").grid(row=0, column=0, columnspan=3, sticky="w")
        self.photo_ref_menu = ctk.CTkOptionMenu(
            pr_cell, values=["Domyślny serii", "Samodzielne (brak)"], command=self._on_photo_ref_menu,
            width=182, height=24, fg_color=T.BTN, button_color=T.BTN, button_hover_color=T.BTN_HOVER,
            text_color=T.TEXT, font=T.font(10), dropdown_font=T.font(11), corner_radius=6
        )
        self.photo_ref_menu.set("Domyślny serii")
        self.photo_ref_menu.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        b_pref = T.button(pr_cell, "▸ Własny…", self.choose_photo_reference, accent=True, width=84); b_pref.configure(height=24, font=T.font(10, "bold"))
        b_pref.grid(row=1, column=1, sticky="e", padx=(4, 0), pady=(2, 0))
        b_auto_pool = T.button(pr_cell, "★ Auto z puli", lambda: self.library.auto_match_pool_references(), width=92)
        b_auto_pool.configure(height=24, font=T.font(10))
        b_auto_pool.grid(row=1, column=2, sticky="e", padx=(4, 0), pady=(2, 0))

        self.ref_info_lbl = T.label(
            f2,
            "Edytor działa samodzielnie bez wzorca. Możesz przypisać osobny wzorzec każdemu zdjęciu lub dopasować wzorce z puli (AI).",
            10, T.MUTED, anchor="w", justify="left", wraplength=380
        )
        self.ref_info_lbl.grid(row=4, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 2))
        f0.lift()

        # ---- Right Dock Section 3: Local Tools Drawer (Pędzel D&B | Kadr) ----
        tools = ctk.CTkFrame(dock, fg_color=T.CARD)
        tools.pack(fill="x", padx=8, pady=(2, 2))
        self.tool_tabs = T.segmented(tools, TOOLS, command=self._tool_tab); self.tool_tabs.configure(height=26, font=T.font(11))
        self.tool_tabs.set(TOOLS[0]); self.tool_tabs.pack(fill="x")
        area = ctk.CTkFrame(tools, fg_color=T.CARD, width=TOOL_W, height=TOOL_H); area.pack(fill="x", pady=(4, 0)); area.pack_propagate(False)
        br = ctk.CTkFrame(area, fg_color=T.CARD); cr = ctk.CTkFrame(area, fg_color=T.CARD)
        for f in (br, cr): f.place(x=0, y=0, relwidth=1, relheight=1)
        self.tool_frames = {TOOLS[0]: br, TOOLS[1]: cr}
        br.grid_columnconfigure(1, weight=1)
        self.brush_mode = T.segmented(br, BRUSH_MODES, command=lambda v: self.view_res.delete("cursor"))
        self.brush_mode.configure(height=24, font=T.font(11)); self.brush_mode.set(BRUSH_MODES[0])
        self.brush_mode.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 3))
        bb = ctk.CTkFrame(br, fg_color=T.CARD); bb.grid(row=0, column=2, sticky="e", padx=(4, 0), pady=(0, 3))
        b_bu = T.button(bb, "Cofnij", self.undo_stroke, width=52); b_bu.configure(height=24, font=T.font(10)); b_bu.pack(side="left")
        b_bc = T.button(bb, "Wyczyść", self.clear_strokes, width=56); b_bc.configure(height=24, font=T.font(10)); b_bc.pack(side="left", padx=(3, 0))
        self.brush_vals = {}
        def brow(r, text, lo, hi, val, key):
            T.label(br, text, 11, T.MUTED).grid(row=r, column=0, sticky="w")
            box = ctk.CTkFrame(br, fg_color=T.CARD, width=50, height=18); box.grid(row=r, column=2, sticky="e", padx=(4, 0)); box.pack_propagate(False)
            lab = T.label(box, "", 11, T.MUTED, anchor="e"); lab.pack(fill="both", expand=True); self.brush_vals[key] = lab
            sl_ = T.slider(br, lo, hi, width=180, command=lambda v: self._brush_label(key)); sl_.configure(height=13); sl_.set(val)
            sl_.grid(row=r, column=1, sticky="ew", padx=(6, 0), pady=1)
            return sl_
        self.brush_size = brow(1, "Rozmiar", 0, 1, 0.45, "size")
        self.brush_soft = brow(2, "Miękkość", 0, 100, 50, "soft")
        self.brush_str = brow(3, "Przepływ", 5, 100, 40, "flow")
        for k in self.brush_vals: self._brush_label(k)
        T.label(br, "[ ] lub Alt+kółko: rozmiar · Ctrl+kółko: zoom", 10, T.MUTED).grid(row=4, column=0, columnspan=3, sticky="w", pady=(2, 0))

        # crop & straighten
        cr.grid_columnconfigure(1, weight=1)
        T.label(cr, "Proporcje", 11, T.MUTED).grid(row=0, column=0, sticky="w")
        self.aspect_menu = ctk.CTkOptionMenu(cr, values=list(G.ASPECTS.values()), command=self._aspect, width=110, height=24,
                                             fg_color=T.BTN, button_color=T.BTN, button_hover_color=T.BTN_HOVER, text_color=T.TEXT,
                                             font=T.font(11), dropdown_font=T.font(11), corner_radius=6)
        self.aspect_menu.grid(row=0, column=1, sticky="w", padx=(6, 0))
        b = T.button(cr, "Pion/Poziom", self._flip_aspect, width=86); b.configure(height=24, font=T.font(10)); b.grid(row=0, column=2, sticky="e")
        T.label(cr, "Prostowanie", 11, T.MUTED).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.angle = T.slider(cr, -45, 45, number_of_steps=900, width=140, command=self._angle)
        self.angle.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=(6, 0))
        box = ctk.CTkFrame(cr, fg_color=T.CARD, width=54, height=18); box.grid(row=1, column=2, sticky="e", pady=(6, 0)); box.pack_propagate(False)
        self.angle_val = T.label(box, "", 11, T.MUTED, anchor="e"); self.angle_val.pack(fill="both", expand=True)
        cb = ctk.CTkFrame(cr, fg_color=T.CARD); cb.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self.hline_btn = T.button(cb, "Linia horyzontu", self.toggle_hline, width=116); self.hline_btn.configure(height=24, font=T.font(10)); self.hline_btn.pack(side="left")
        b_rc = T.button(cb, "Resetuj kadr", self.reset_crop, width=96); b_rc.configure(height=24, font=T.font(10)); b_rc.pack(side="left", padx=6)
        T.label(cr, "Przeciągnij na wyniku: kadr · Enter: gotowe · Esc: wyjście", 10, T.MUTED,
                justify="left").grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))
        br.lift()

        # ---- Right Dock Section 4: Context Help Box ('Podpowiedź PRO') ----
        tip_box = ctk.CTkFrame(dock, fg_color=T.PANEL, corner_radius=8, height=48)
        tip_box.pack(fill="x", padx=8, pady=(2, 6))
        tip_box.pack_propagate(False)
        self.pro_tip_lbl = T.label(
            tip_box,
            "▸ Podpowiedź PRO: Najedź na dowolny suwak, aby zobaczyć wskazówkę kierunkową (np. Odcień skóry: ◀ Róż/Czerwień ↔ Złoty/Żółty ▶). Dwuklik zeruje suwak.",
            10, T.MUTED, anchor="w", justify="left", wraplength=384
        )
        self.pro_tip_lbl.pack(fill="both", expand=True, padx=8, pady=4)

        self.status = T.label(self, "", 12, T.MUTED); self.status.pack(fill="x", pady=4)
        self._load_params_into_ui(self.params); self._update_status()

    def _set_pro_tip(self, key: str):
        if hasattr(self, "pro_tip_lbl") and key in PRO_TIPS:
            self.pro_tip_lbl.configure(text=f"▸ {PRO_TIPS[key]}")

    # ================= threading =================
    def _worker(self):
        while True:
            fn, done, kind, gen = self._jobs.get()
            if kind and self._gen.get(kind) != gen: continue  # superseded
            try:
                res = fn(); self._ui.put((done, res))
            except Exception as e:
                tb = traceback.format_exc(); self._ui.put((self._error, (e, tb)))

    def submit(self, fn, done, kind=None):
        g = None
        if kind: g = self._gen[kind] = self._gen.get(kind, 0)+1
        self._jobs.put((fn, done, kind, g))

    def _poll(self):
        try:
            while True:
                done, res = self._ui.get_nowait()
                if done: done(res)
        except queue.Empty: pass
        self.after(30, self._poll)

    def _error(self, info):
        e, tb = info; print(tb, file=sys.stderr); self._set_status(f"Błąd: {e}")

    # ================= engine / model =================
    def setup_engine(self):
        path, eff = models.resolve(self.settings.method)
        def work():
            e = Engine(path, eff); return e, e.device()
        def done(r):
            self.engine, self.device = r; self.engine_name = ENGINE_LABEL[self.engine.method]
            self.fit = None; self._update_status(); self.refit()
        self.submit(work, done)

    def _first_run(self):
        if self.settings.method == "ai" and not models.have_ai() and not self.settings.asked_download:
            self.settings.asked_download = True; self.settings.save()
            ModelDialog(self, lambda ok: ok and self.setup_engine())

    def open_settings(self):
        old = (self.settings.method, self.settings.preview_side, self.settings.device, models.have_ai())
        def saved():
            new = (self.settings.method, self.settings.preview_side, self.settings.device, models.have_ai())
            if new[2] != old[2]:
                DEV.MODE = self.settings.device; DEV.reset(); MF.reset_sessions(); RL.reset_session()
            if new != old:
                self.setup_engine()
                if self.idx >= 0: self.select(self.idx)
            self.redraw_result()
        SettingsDialog(self, self.settings, saved)

    # ================= opening =================
    def open_transfer(self):
        from .transfer_view import TransferWindow
        cur = self.photos[self.idx] if 0 <= self.idx < len(self.photos) else None
        self.transfer_win = TransferWindow(self, orig=cur, on_open=self.open_path)

    def open_photo(self):
        paths = filedialog.askopenfilenames(
            title="Wybierz jedno lub kilka zdjęć z folderu (Ctrl/Shift + klik)",
            initialdir=self.settings.last_folder or None,
            filetypes=PHOTO_TYPES,
        )
        if not paths:
            return
        if len(paths) == 1:
            self.open_path(Path(paths[0]))
        else:
            self.open_photos([Path(p) for p in paths])

    def add_photos(self):
        paths = filedialog.askopenfilenames(
            title="Dodaj wybrane zdjęcia z folderu do bieżącej serii",
            initialdir=self.settings.last_folder or None,
            filetypes=PHOTO_TYPES,
        )
        if paths:
            self.open_photos([Path(p) for p in paths], append=True)

    def open_photos(self, paths: list[Path], append: bool = False):
        """Import a user-selected subset of photos from a folder (or append them to the current series)."""
        if not paths:
            return
        if append and self.series is not None and self.photos:
            existing = {p.resolve() for p in self.photos}
            ref = self.series.reference.resolve() if (self.series.reference and self.ref_enabled) else None
            added = 0
            for p in paths:
                rp = Path(p).resolve()
                if Path(p).is_file() and rp not in existing and rp != ref:
                    self.photos.append(Path(p))
                    existing.add(rp)
                    added += 1
            self.series_lbl.configure(text=f"Seria · {len(self.photos)}")
            self._build_strip()
            if self.library.winfo_ismapped():
                self.library.refresh()
            self._set_status(f"Dodano {added} zdjęć do serii (łącznie: {len(self.photos)}).")
            return

        self.series, self.photos = Series.from_selection(paths)
        self.settings.last_folder = str(self.series.folder); self.settings.save()
        self.series_lbl.configure(text=f"Seria · {len(self.photos)}")
        self._build_strip()
        if self.series.reference and self.series.reference.exists():
            self.ref_enabled = True
            self.load_reference(self.series.reference)
        else:
            self._sync_ref_ui()
        if self.photos:
            self.select(0)
            self._set_status(f"Zaimportowano {len(self.photos)} wybranych zdjęć z folderu {self.series.folder.name}.")
        if self.library.winfo_ismapped():
            self.library.refresh()

    def open_folder(self):
        p = filedialog.askdirectory(title="Otwórz cały folder", initialdir=self.settings.last_folder or None)
        if p: self.open_path(Path(p))

    def _set_canvas_mode(self, mode):
        self._apply_canvas_layout()
        self.redraw_result()

    def _apply_canvas_layout(self):
        m = self.canvas_mode.get() if hasattr(self, "canvas_mode") else "Przed | Po"
        for w_ in (self.src_hdr, self.res_hdr, self.ref_hdr, self.view_src, self.view_res, self.view_ref):
            w_.grid_remove()
        for c in range(3):
            self.imgs.grid_columnconfigure(c, weight=0, uniform="")
        if m in ("Wynik", "Podział"):
            self.imgs.grid_columnconfigure(1, weight=1, uniform="")
            self.res_hdr.configure(text="Podział: Przed | Po (przeciągnij kurtynę)" if m == "Podział" else ("Wynik (Oryginał)" if self.show_before else "Wynik"))
            self.res_hdr.grid(row=0, column=1, sticky="w", padx=4, pady=(0, 4))
            self.view_res.grid(row=1, column=1, sticky="nsew", padx=4)
        elif m == "Przed | Po":
            for c in (0, 1):
                self.imgs.grid_columnconfigure(c, weight=1, uniform="v2")
            self.res_hdr.configure(text="Wynik (Oryginał)" if self.show_before else "Wynik")
            self.src_hdr.grid(row=0, column=0, sticky="w", padx=4, pady=(0, 4))
            self.res_hdr.grid(row=0, column=1, sticky="w", padx=4, pady=(0, 4))
            self.view_src.grid(row=1, column=0, sticky="nsew", padx=4)
            self.view_res.grid(row=1, column=1, sticky="nsew", padx=4)
        else:  # "Z wzorcem"
            for c in range(3):
                self.imgs.grid_columnconfigure(c, weight=1, uniform="v3")
            self.res_hdr.configure(text="Wynik (Oryginał)" if self.show_before else "Wynik")
            self.src_hdr.grid(row=0, column=0, sticky="w", padx=4, pady=(0, 4))
            self.res_hdr.grid(row=0, column=1, sticky="w", padx=4, pady=(0, 4))
            self.ref_hdr.grid(row=0, column=2, sticky="w", padx=4, pady=(0, 4))
            self.view_src.grid(row=1, column=0, sticky="nsew", padx=4)
            self.view_res.grid(row=1, column=1, sticky="nsew", padx=4)
            self.view_ref.grid(row=1, column=2, sticky="nsew", padx=4)

    def open_path(self, path: Path, reference: Path | None = None):
        folder = path if path.is_dir() else path.parent
        self.series = Series(folder); self.settings.last_folder = str(folder); self.settings.save()
        if reference:
            self.series.reference = reference; self.series.save()
            self.ref_enabled = True
        ref = self.series.reference.resolve() if (self.series.reference and self.ref_enabled) else None
        if path.is_dir():
            out_dir = (folder / self.settings.output_subfolder).resolve()
            self.photos = [p for p in imageio.list_images(folder) if p.resolve() != ref and p.parent.resolve() != out_dir]
        else:
            self.photos = [path]
        self.series_lbl.configure(text=f"Seria · {len(self.photos)}")
        self._build_strip()
        if self.series.reference and self.series.reference.exists():
            self.ref_enabled = True
            self.load_reference(self.series.reference)
        else:
            self._sync_ref_ui()
        if self.photos: self.select(0)
        else: self._set_status("W folderze nie ma zdjęć JPG/PNG/TIFF/NEF.")
        tw = imageio.raw_twins(folder) if path.is_dir() else 0
        if tw: self.after(1500, lambda: self._set_status(f"Pominięto {tw} JPG z parą NEF o tej samej nazwie – edytujesz pliki NEF."))

    def _sync_ref_ui(self):
        if not hasattr(self, "ref_info_lbl"): return
        if self.reference is not None:
            rep = getattr(self.fit, "cc_report", None)
            if rep is not None:
                txt = f"{self.reference.path.name}  ·  {rep.summary_pl().replace('.', ',')}"
            elif getattr(self.reference, "colorchecker", False):
                txt = f"Tablica ColorChecker: {self.reference.path.name}  ·  Trwa analiza 24 pól…"
            else:
                txt = f"Aktywny wzorzec: {self.reference.path.name}  ·  Widok „Z wzorcem” pokazuje 3 kolumny."
            self.ref_info_lbl.configure(text=txt, text_color=T.ACCENT)
            self.clear_ref_btn.configure(state="normal")
        else:
            self.ref_info_lbl.configure(
                text="Edytor działa samodzielnie bez wzorca. Wybierz wzorzec lub tablicę ColorChecker (24 pola), aby skalibrować serię.",
                text_color=T.MUTED
            )
            self.clear_ref_btn.configure(state="disabled")

    def clear_reference(self):
        self.reference = None; self.ref_analysis = None; self.ref_prev = None; self.ref_enabled = False
        if self.series is not None:
            self.series.reference = None; self.series.save()
        self.view_ref.set_image(None, keep_view=False)
        if self.canvas_mode.get() == "Z wzorcem":
            self.canvas_mode.set("Przed | Po"); self._apply_canvas_layout()
        self._sync_ref_ui(); self.fit = None; self.refit()

    def choose_reference(self):
        p = filedialog.askopenfilename(title="Wybierz wzorzec", initialdir=self.settings.last_folder or None,
                                       filetypes=PHOTO_TYPES)
        if not p: return
        p = Path(p)
        if self.series is None: self.series = Series(p.parent)
        self.series.reference = p; self.series.save(); self.ref_enabled = True
        if p in self.photos:
            i = self.photos.index(p); self.photos.remove(p); self._build_strip()
            if self.photos: self.select(min(i, len(self.photos)-1))
        self.load_reference(p, colorchecker=False)

    def choose_colorchecker(self, path: Path | None = None):
        if path is None:
            p = filedialog.askopenfilename(title="Wybierz zdjęcie z tablicą ColorChecker (24 pola)",
                                           initialdir=self.settings.last_folder or None, filetypes=PHOTO_TYPES)
            if not p: return
            path = Path(p)
        else:
            path = Path(path)
        if self.series is None: self.series = Series(path.parent)
        self.series.reference = path; self.series.save(); self.ref_enabled = True
        self.load_reference(path, colorchecker=True)

    def calibrate_current_colorchecker(self):
        if self.idx < 0 or not self.photos:
            self._set_status("Najpierw otwórz zdjęcie z tablicą ColorChecker."); return
        self.choose_colorchecker(self.photos[self.idx])

    def _refresh_photo_ref_menu(self):
        if not hasattr(self, "photo_ref_menu"): return
        opts = ["Domyślny serii", "Samodzielne (brak)"]
        if self.series is not None:
            for bp in self.series.reference_bank:
                label = f"Wzorzec: {bp.name}"
                if label not in opts:
                    opts.append(label)
        rp = getattr(self.params, "reference_path", None)
        if rp == "__none__":
            cur = "Samodzielne (brak)"
        elif rp:
            cur = f"Wzorzec: {Path(rp).name}"
            if cur not in opts:
                opts.append(cur)
        else:
            cur = "Domyślny serii"
        self.photo_ref_menu.configure(values=opts)
        self.photo_ref_menu.set(cur)

    def _on_photo_ref_menu(self, choice: str):
        if self.idx < 0 or not self.photos or self.series is None: return
        self._commit_history()
        if choice == "Domyślny serii":
            self.params.reference_path = None
            self.params.ref_colorchecker = False
        elif choice == "Samodzielne (brak)":
            self.params.reference_path = "__none__"
            self.params.ref_colorchecker = False
        elif choice.startswith("Wzorzec: "):
            fname = choice[len("Wzorzec: "):]
            match = next((b for b in self.series.reference_bank if b.name == fname), None)
            self.params.reference_path = str(match) if match is not None else fname
            self.params.ref_colorchecker = False
        self.series.set_override(self.photos[self.idx], self.params)
        self._refresh_strip_marks(); self._update_override_label()
        self._sync_photo_reference()

    def choose_photo_reference(self):
        if self.idx < 0 or not self.photos:
            self._set_status("Najpierw otwórz zdjęcie."); return
        p = filedialog.askopenfilename(title="Wybierz indywidualny wzorzec dla tego zdjęcia",
                                       initialdir=self.settings.last_folder or None, filetypes=PHOTO_TYPES)
        if not p: return
        p = Path(p)
        if self.series is None: self.series = Series(p.parent)
        self._commit_history()
        self.series.add_to_reference_bank(p)
        self.params.reference_path = str(p)
        self.params.ref_colorchecker = False
        self.series.set_override(self.photos[self.idx], self.params)
        self.series.save()
        self._refresh_photo_ref_menu(); self._refresh_strip_marks(); self._update_override_label()
        self._sync_photo_reference()

    def _sync_photo_reference(self):
        """Ensure self.reference matches the current photo's effective reference (per-photo override or series default)."""
        if self.series is None or self.idx < 0 or not self.photos:
            self.refit(); return
        eff_path, eff_cc = self.series.effective_reference_for(self.photos[self.idx], self.params)
        cur_path = self.reference.path if self.reference is not None else None
        cur_cc = bool(getattr(self.reference, "colorchecker", False)) if self.reference is not None else False
        if eff_path is None:
            if self.reference is not None:
                self.reference = None; self.ref_analysis = None; self.ref_prev = None
                self.view_ref.set_image(None, keep_view=False)
                self._sync_ref_ui()
            self.fit = None; self.refit()
        elif cur_path is None or cur_path.resolve() != eff_path.resolve() or cur_cc != eff_cc:
            self.load_reference(eff_path, colorchecker=eff_cc)
        else:
            self.fit = None; self.refit()

    def load_reference(self, p: Path, colorchecker: bool = False):
        self._set_status("Kalibruję tablicę ColorChecker (24 pola)…" if colorchecker else "Wczytuję wzorzec…")
        side = self.settings.preview_side
        if self.series is not None:
            self.series.add_to_reference_bank(p)
        def work():
            ref = Reference(p, colorchecker=colorchecker)
            prev, icc = imageio.load_preview(p, side, with_icc=True)
            cal_prev = CC.calibrate_chart(prev) if colorchecker else None
            disp_prev = CC.draw_chart_overlay(prev, cal_prev) if cal_prev is not None else C.crop_border(prev, margin=2)
            return ref, (disp_prev, icc), cal_prev
        def done(r):
            self.reference, (self.ref_prev, ref_icc), cal_prev = r
            self.ref_analysis = None; self.ref_enabled = True
            self.view_ref.set_image(self.ref_prev, icc=ref_icc, keep_view=False)
            if self.canvas_mode.get() != "Z wzorcem":
                self.canvas_mode.set("Z wzorcem"); self._apply_canvas_layout()
            if colorchecker and cal_prev is None:
                self._set_status("Nie wykryto tablicy ColorChecker 24 na zdjęciu – używam dopasowania wzorca.")
            self._refresh_photo_ref_menu(); self._sync_ref_ui(); self.fit = None; self.refit()
        self.submit(work, done)

    # ================= series strip =================
    def _build_strip(self):
        for b in self.thumb_btns: b.destroy()
        self.thumb_btns = []
        for i, p in enumerate(self.photos):
            b = ctk.CTkButton(self.strip, text=self._thumb_text(p), width=130, height=110, compound="top", corner_radius=6,
                              fg_color=T.PANEL, hover_color=T.BTN_HOVER, text_color=T.TEXT, border_width=2, border_color=T.PANEL,
                              font=T.font(11), command=lambda i=i: self.select(i))
            b.pack(pady=4, padx=4); self.thumb_btns.append(b)
        photos = list(self.photos)
        def load_all():
            for i, p in enumerate(photos):
                if p in self.thumbs: continue
                try: im = Image.fromarray(imageio.load_preview(p, 240))
                except Exception: continue
                self._ui.put((lambda r: self._set_thumb(*r), (i, p, im)))
        threading.Thread(target=load_all, daemon=True).start()

    def _set_thumb(self, i, p, im):
        im.thumbnail((124, 84)); self.thumbs[p] = ctk.CTkImage(im, size=im.size)
        if i < len(self.thumb_btns) and self.photos[i] == p: self.thumb_btns[i].configure(image=self.thumbs[p])

    def _thumb_text(self, p):
        name = p.name if len(p.name) <= 18 else p.name[:15] + "…"
        return ("● " if self.series and self.series.has_override(p) else "") + name

    def _refresh_strip_marks(self):
        for i, b in enumerate(self.thumb_btns):
            b.configure(text=self._thumb_text(self.photos[i]), border_color=T.ACCENT if i == self.idx else T.PANEL)

    # ================= selection / processing =================
    def step(self, d):
        if self.photos and not isinstance(self.focus_get(), ctk.CTkEntry):
            self.select(max(0, min(len(self.photos)-1, self.idx+d)))

    def select(self, i):
        self.idx = i; p = self.photos[i]; self.params = self.series.params_for(p)
        self._reset_history(); self.show_before = False; self._style_before()
        self._load_params_into_ui(self.params); self._refresh_strip_marks(); self.fit = None
        self.result = None; self.frame = None; self.detail_raw = None; self.full_cache = None
        self.view_res.set_image(None, keep_view=False); self._set_status(f"Wczytuję {p.name}…")
        side = self.settings.preview_side
        def work():
            return imageio.load_preview(p, side, with_icc=True), imageio.image_size(p)
        def done(r):
            if self.idx != i: return
            (self.src_prev, self.src_icc), self.src_size = r
            self.view_src.set_image(self.src_prev, icc=self.src_icc, full_wh=self.src_size, keep_view=False)
            self._sync_zoom_label(); self._sync_photo_reference()
        self.submit(work, done, kind="load")

    def refit(self):
        if self.src_prev is None or self.engine is None:
            self._update_status(); return
        i, src, ref, eng, icc = self.idx, self.src_prev, self.reference, self.engine, self.src_icc
        cc_flag = bool(ref is not None and getattr(ref, "colorchecker", False))
        self._set_status("Kalibruję ColorChecker…" if cc_flag else ("Dopasowuję kolory…" if ref is not None else "Analizuję portret i mapę głębi 3D…"))
        def work():
            ra = ref.analysis(eng, icc) if ref is not None else None
            return ra, eng.fit(src, ra, colorchecker=cc_flag)
        def done(r):
            if self.idx != i: return
            self.ref_analysis, self.fit = r; self._sync_ref_ui(); self._show_light(); self.request_render()
        self.submit(work, done, kind="fit")

    def request_render(self):
        if self.fit is None or self.src_prev is None: return
        fit, src, p, eng, i = self.fit, self.src_prev, Params.from_dict(self.params.to_dict()), self.engine, self.idx
        def done(r):
            if self.idx != i: return
            self.result, self.mask_prev = r; self.redraw_result(); self._update_status()
            if self.compare.winfo_ismapped(): self.compare.set_images(self._disp(self.src_prev), self._disp(self.result))
            self._schedule_detail()
        self.submit(lambda: eng.render(src, fit, p, return_mask=True), done, kind="render")

    def redraw_result(self):
        """Result view = the straightened frame; outside the crop is dimmed (or shaded by the crop tool)."""
        if self.result is None: return
        a = self.params.angle
        self.frame = G.apply(self.result, a, None, interp=cv2.INTER_CUBIC) if a else self.result
        if self.show_before and self.src_prev is not None:
            img = G.apply(self.src_prev, a, None, interp=cv2.INTER_CUBIC) if a else self.src_prev
        else:
            img = self.frame
            if self.settings.show_mask and self.mask_prev is not None:
                faces_n = self.fit.faces_n if self.fit is not None else ()
                lms_n = getattr(self.fit, "lms_n", None) if self.fit is not None else None
                subj_fg = (self.fit.relight.fg if (self.fit is not None and self.fit.relight is not None) else None)
                ov_base = HI.ai_mask_overlay(self.result, self.mask_prev, faces_n=faces_n, lms_n=lms_n, subject_fg=subj_fg)
                img = G.apply(ov_base, a, None, interp=cv2.INTER_CUBIC) if a else ov_base
        self.view_res.set_image(self._decorate(img), icc=self.src_icc, full_wh=self.src_size)
        if self.show_before and self.full_cache is not None and self.idx >= 0 and self.full_cache[0] == self.photos[self.idx]:
            det = G.apply(self.full_cache[1], a, None, interp=cv2.INTER_LINEAR) if a else self.full_cache[1]
            self.view_res.detail = self._decorate(det); self.view_res.redraw()
        elif self.detail_raw is not None:
            self.view_res.detail = self._decorate(self.detail_raw); self.view_res.redraw()
        if hasattr(self, "canvas_mode") and self.canvas_mode.get() == "Podział" and self.src_prev is not None and not self.show_before:
            bf = G.apply(self.src_prev, a, None, interp=cv2.INTER_CUBIC) if a else self.src_prev
            bdet = None
            if self.full_cache is not None and self.idx >= 0 and self.full_cache[0] == self.photos[self.idx]:
                bdet = self._decorate(G.apply(self.full_cache[1], a, None, interp=cv2.INTER_LINEAR) if a else self.full_cache[1])
            self.view_res.set_split(self._decorate(bf), getattr(self, "split_u", 0.5), bdet)
        else:
            self.view_res.set_split(None, None, None)
        if self.view_res.zoom != self.view_src.zoom or self.view_res.center != self.view_src.center:
            self.view_res.set_view(self.view_src.zoom, self.view_src.center)
        if not self._brush_on():
            self.view_res.delete("cursor")
        self._sync_zoom_label(); self._update_hist()

    def _style_before(self):
        if hasattr(self, "before_btn"):
            self.before_btn.configure(border_color=T.ACCENT if self.show_before else T.BORDER,
                                      text_color=T.ACCENT if self.show_before else T.TEXT)
        if getattr(self, "res_hdr", None) is not None:
            m = self.canvas_mode.get() if hasattr(self, "canvas_mode") else "Przed | Po"
            self.res_hdr.configure(text="Wynik (Oryginał)" if self.show_before else ("Podział: Przed | Po (przeciągnij kurtynę)" if m == "Podział" else "Wynik"),
                                   text_color=T.ACCENT if self.show_before else T.TEXT)

    def toggle_before(self, on=None):
        if self.result is None: return
        self.show_before = (not self.show_before) if on is None else bool(on)
        self._style_before(); self.redraw_result()

    def toggle_mask(self, on=None):
        self.settings.show_mask = (not self.settings.show_mask) if on is None else bool(on)
        if hasattr(self, "mask_sw"):
            (self.mask_sw.select() if self.settings.show_mask else self.mask_sw.deselect())
        self.redraw_result()
        n_faces = len(self.fit.faces_n) if (self.fit is not None and self.fit.faces_n) else 0
        self._set_status(
            f"Maska AI (M): włączona — wykryte twarze: {n_faces} (bursztyn = maska skóry, turkus = granica 3D, punkty = oczy/nos/usta)."
            if self.settings.show_mask else "Maska AI (M): wyłączona."
        )

    def _decorate(self, img):
        """Clipping overlay + crop dimming for display (the export never sees this)."""
        if self.show_clip: img = HI.overlay(img)
        c = self.params.crop
        if c is not None and self.tool != "crop":
            h, w = img.shape[:2]; x0, y0, x1, y1 = G.crop_px(c, w, h)
            out = (img*0.28).astype(np.uint8); out[y0:y1, x0:x1] = img[y0:y1, x0:x1]; img = out
        return img

    def _update_hist(self):
        if self.frame is None: self.hist.draw(None, active=self.show_clip); return
        h, w = self.frame.shape[:2]; x0, y0, x1, y1 = G.crop_px(self.params.crop, w, h)
        reg = self.frame[y0:y1, x0:x1]; lo, hi = HI.clipping(reg)[:2]
        vec = HI.skin_vectorscope(reg, self.mask_prev)
        self.hist.draw(HI.histogram(reg), lo, hi, self.show_clip, vec=vec)
        self.clip_lbl.configure(text=f"Cienie {lo*100:.1f}% · Światła {hi*100:.1f}%".replace(".", ","),
                                text_color=T.TEXT if max(lo, hi) > 0.005 else T.MUTED)

    def toggle_clip(self, on=None):
        self.show_clip = (not self.show_clip) if on is None else bool(on)
        (self.clip_sw.select() if self.show_clip else self.clip_sw.deselect())
        self.redraw_result()

    # ================= white balance =================
    def toggle_pick(self, on=None):
        self.picking = (not self.picking) if on is None else bool(on)
        self.pick_btn.configure(border_color=T.ACCENT if self.picking else T.BORDER, text_color=T.ACCENT if self.picking else T.TEXT)
        self.view_res.configure(cursor="crosshair" if self.picking else "")
        if self.picking:
            self.view_res.delete("cursor"); self._set_status("Pipeta: kliknij na wyniku punkt, który ma być neutralnie szary lub biały (Esc: anuluj).")

    def _pick(self, e):
        if self.frame is None: return
        u, v = self.view_res.to_norm(e.x, e.y); h, w = self.frame.shape[:2]
        x, y = int(u*w), int(v*h)
        if not (0 <= x < w and 0 <= y < h): return
        self._last_pick = (x, y); d = WB.solve(self.frame[max(0, y-3):y+4, max(0, x-3):x+4])
        if d is None:
            self._set_status("Ten punkt jest przepalony albo za ciemny – kliknij jasnoszary punkt bez przepaleń."); return
        t, ti = self.params.temp + d[0], self.params.tint + d[1]
        cut = abs(t) > 1 or abs(ti) > 1
        self._set_wb(float(np.clip(t, -1, 1)), float(np.clip(ti, -1, 1))); self.toggle_pick(False)
        self._set_status("Balans bieli ustawiony z pipety" + (" (poza zakresem suwaków – przycięty)." if cut else "."))

    def _set_wb(self, t, ti):
        self._commit_history()
        self.params.temp, self.params.tint = t, ti
        for k in ("temp", "tint"):
            s_, val, kind = self.sliders[k]; pv = getattr(self.params, k); v = pv*100; s_.set(v)
            val.configure(text=self._fmt(v, kind), text_color=self._val_color(k, pv))
        self._changed(discrete=True)

    def reset_wb(self):
        self._set_wb(0.0, 0.0)

    def sync_wb(self):
        if not self.series or not self.photos: return
        if self._save_after: self.after_cancel(self._save_after); self._save_after = None
        self.series.set_override(self.photos[self.idx], Params.from_dict(self.params.to_dict())) if self.series.has_override(self.photos[self.idx]) else None
        self.series.sync_fields(self.photos, {"temp": self.params.temp, "tint": self.params.tint})
        self._refresh_strip_marks(); self._update_override_label()
        self._set_status(f"Balans bieli skopiowany do całej serii ({len(self.photos)} zdjęć).")

    # ================= crop & straighten =================
    def _frame_wh(self):
        if self.src_size: return self.src_size
        return (self.result.shape[1], self.result.shape[0]) if self.result is not None else (3, 2)

    def _crop_portrait(self):
        W, H = self._frame_wh(); c = self.params.crop
        return (c[2]-c[0])*W < (c[3]-c[1])*H if c else H > W

    def _tool_tab(self, name):
        self.tool = "crop" if name == TOOLS[1] else "brush"
        self.tool_frames[name].lift(); self.hline_on = False; self._hline = None; self._style_hline()
        if self.tool == "crop": self.toggle_pick(False); self.view_res.delete("cursor")
        self.redraw_result(); self._draw_crop_overlay()

    def toggle_crop_tool(self):
        name = TOOLS[0] if self.tool == "crop" else TOOLS[1]; self.tool_tabs.set(name); self._tool_tab(name)

    def _crop_done(self):
        if self.tool == "crop": self.toggle_crop_tool()

    def _escape(self):
        if self.picking: self.toggle_pick(False); self._update_status()
        elif self.tool == "crop": self.toggle_crop_tool()

    def _key(self, fn):
        if isinstance(self.focus_get(), ctk.CTkEntry): return
        fn()

    def toggle_hline(self):
        self.hline_on = not self.hline_on; self._style_hline()
        if self.hline_on: self._set_status("Linia horyzontu: przeciągnij wzdłuż horyzontu lub pionowej krawędzi.")

    def _style_hline(self):
        self.hline_btn.configure(border_color=T.ACCENT if self.hline_on else T.BORDER, text_color=T.ACCENT if self.hline_on else T.TEXT)

    def _crop_ui_sync(self):
        a = self.params.angle; self.angle.set(a)
        self.angle_val.configure(text=f"{a:+.1f}°".replace(".", ",") if a else "0°")
        self.aspect_menu.set(G.ASPECTS.get(self.params.aspect, G.ASPECTS["orig"]))

    def _geometry_changed(self):
        self._crop_ui_sync(); self.redraw_result(); self._changed(render=False)

    def _aspect(self, label):
        key = {v: k for k, v in G.ASPECTS.items()}[label]; self.params.aspect = key
        if key != "free":
            W, H = self._frame_wh(); self.params.crop = G.clamp_crop(G.aspect_rect(key, W, H, self._crop_portrait()))
        self._geometry_changed()

    def _flip_aspect(self):
        W, H = self._frame_wh(); c = self.params.crop
        if self.params.aspect == "free":
            if c is None: return
            cx, cy = (c[0]+c[2])/2, (c[1]+c[3])/2; hw, hh = (c[3]-c[1])*H/W/2, (c[2]-c[0])*W/H/2
            k = min(1.0, cx/hw, (1-cx)/hw, cy/hh, (1-cy)/hh); hw, hh = hw*k, hh*k
            self.params.crop = G.clamp_crop([cx-hw, cy-hh, cx+hw, cy+hh])
        else:
            self.params.crop = G.clamp_crop(G.aspect_rect(self.params.aspect, W, H, not self._crop_portrait()))
        self._geometry_changed()

    def _angle(self, v):
        self.params.angle = round(float(v), 1)
        self.angle_val.configure(text=f"{self.params.angle:+.1f}°".replace(".", ",") if self.params.angle else "0°")
        self.detail_raw = None; self.view_res.detail = None
        self.redraw_result()
        if self._geo_after: self.after_cancel(self._geo_after)
        self._geo_after = self.after(400, lambda: (self._changed(render=False), self._schedule_detail()))

    def reset_crop(self):
        self.params.angle = 0.0; self.params.crop = None; self.params.aspect = "orig"
        self.detail_raw = None; self.view_res.detail = None; self._geometry_changed()

    def _crop_press(self, e):
        v = self.view_res
        if self._space or self.result is None: self._pan_start(v, e); return
        if self.hline_on: self._hline = (e.x, e.y, e.x, e.y); self._crop_drag = ("line",); return
        u, w_ = (float(np.clip(t, 0, 1)) for t in v.to_norm(e.x, e.y))
        c = self.params.crop or [0.0, 0.0, 1.0, 1.0]; x0, y0, dw, dh = v.geom
        corners = [(c[0], c[1]), (c[2], c[1]), (c[0], c[3]), (c[2], c[3])]
        for i, (cx, cy) in enumerate(corners):
            if abs(x0+cx*dw-e.x) < 14 and abs(y0+cy*dh-e.y) < 14:
                self._crop_drag = ("resize", corners[3-i]); return
        if self.params.crop and c[0] < u < c[2] and c[1] < w_ < c[3]:
            self._crop_drag = ("move", (u, w_), list(c)); return
        self._crop_drag = ("new", (u, w_))

    def _crop_move(self, e):
        if self._pan is not None: self._pan_move(self.view_res, e); return
        d = self._crop_drag
        if not d: return
        if d[0] == "line":
            self._hline = (*self._hline[:2], e.x, e.y); self._draw_crop_overlay(); return
        u, w_ = (float(np.clip(t, 0, 1)) for t in self.view_res.to_norm(e.x, e.y)); W, H = self._frame_wh()
        if d[0] == "move":
            (pu, pv), c = d[1], d[2]
            dx = float(np.clip(u-pu, -c[0], 1-c[2])); dy = float(np.clip(w_-pv, -c[1], 1-c[3]))
            self._crop_live = [c[0]+dx, c[1]+dy, c[2]+dx, c[3]+dy]
        else:
            ax, ay = d[1]; rect = [min(ax, u), min(ay, w_), max(ax, u), max(ay, w_)]
            if self.params.aspect != "free":
                portrait = abs(u-ax)*W < abs(w_-ay)*H if d[0] == "new" else self._crop_portrait()
                r = G.ratio(self.params.aspect, W, H, portrait)
                if r: rect = G.fit_aspect(rect, r, W, H, anchor=(ax, ay))
            self._crop_live = rect
        self._draw_crop_overlay()

    def _crop_release(self, e):
        if self._pan is not None: self._pan_end(); return
        d, self._crop_drag = self._crop_drag, None
        if not d: return
        if d[0] == "line":
            x0_, y0_, x1_, y1_ = self._hline; self._hline = None
            if np.hypot(x1_-x0_, y1_-y0_) > 12:
                W, H = self._frame_wh(); v = self.view_res
                a = np.array(v.to_norm(x0_, y0_))*(W, H); b = np.array(v.to_norm(x1_, y1_))*(W, H)
                self.params.angle = float(np.clip(round(self.params.angle + G.horizon_angle(a, b), 1), -45, 45))
                self.detail_raw = None; self.view_res.detail = None
                self._set_status(f"Wyprostowano: kąt {self.params.angle:+.1f}°".replace(".", ","))
            else: self._update_status()
            self.hline_on = False; self._style_hline(); self._geometry_changed(); self._schedule_detail(); return
        live, self._crop_live = self._crop_live, None
        if live is None: self._draw_crop_overlay(); return
        if (live[2]-live[0]) < 0.01 or (live[3]-live[1]) < 0.01:
            self._draw_crop_overlay(); return
        self.params.crop = G.clamp_crop(live); self._geometry_changed()

    def _draw_crop_overlay(self):
        v = self.view_res; v.delete("crop")
        if self.result is None or self.tool != "crop": return
        x0, y0, dw, dh = v.geom
        c = self._crop_live or self.params.crop or [0.0, 0.0, 1.0, 1.0]
        X0, Y0, X1, Y1 = x0+c[0]*dw, y0+c[1]*dh, x0+c[2]*dw, y0+c[3]*dh
        for r in ((x0, y0, x0+dw, Y0), (x0, Y1, x0+dw, y0+dh), (x0, Y0, X0, Y1), (X1, Y0, x0+dw, Y1)):
            if r[2]-r[0] > 0.5 and r[3]-r[1] > 0.5: v.create_rectangle(*r, fill="#000000", stipple="gray50", outline="", tags="crop")
        for i in (1, 2):
            xx, yy = X0+(X1-X0)*i/3, Y0+(Y1-Y0)*i/3
            v.create_line(xx, Y0, xx, Y1, fill="#BDBDBD", tags="crop"); v.create_line(X0, yy, X1, yy, fill="#BDBDBD", tags="crop")
        v.create_rectangle(X0, Y0, X1, Y1, outline="#FFFFFF", width=1, tags="crop")
        L = min(16, (X1-X0)/3, (Y1-Y0)/3)
        for cx, cy, sx, sy in ((X0, Y0, 1, 1), (X1, Y0, -1, 1), (X0, Y1, 1, -1), (X1, Y1, -1, -1)):
            v.create_line(cx, cy+sy*L, cx, cy, cx+sx*L, cy, fill=T.ACCENT, width=3, tags="crop")
        if self._hline: v.create_line(*self._hline, fill=T.ACCENT, width=2, dash=(6, 3), tags="crop")

    # ================= controls =================
    def _slider_tab(self, name):
        self.slider_frames[name].lift()

    def _val_color(self, key, param_val):
        return T.TEXT if abs(param_val - getattr(_DEFAULTS, key)) > 1e-3 else T.MUTED

    def _load_params_into_ui(self, p: Params):
        for key, (s, val, kind) in self.sliders.items():
            pv = getattr(p, key); v = pv*100; s.set(v)
            val.configure(text=self._fmt(v, kind), text_color=self._val_color(key, pv))
        is_auto = (p.light_angle is None and abs(p.light_x) < 1e-4 and abs(p.light_y) < 1e-4)
        if is_auto:
            self.light_auto.select()
        else:
            self.light_auto.deselect()
        if hasattr(self, "sharpen_mode_menu"):
            self.sharpen_mode_menu.set(DET.SHARPEN_MODES.get(p.sharpen_mode, DET.SHARPEN_MODES["portrait"]))
        self._update_star_btns(p.rating)
        self._refresh_photo_ref_menu()
        self._show_light(); self._crop_ui_sync()
        self._update_override_label(); self._update_history_btns()

    def _set_sharpen_mode(self, label: str):
        inv = {v: k for k, v in DET.SHARPEN_MODES.items()}
        mode = inv.get(label, "portrait")
        if self.params.sharpen_mode == mode: return
        self._commit_history()
        self.params.sharpen_mode = mode
        self._changed(discrete=True)

    def _update_star_btns(self, rating: int):
        for i, sb in enumerate(getattr(self, "star_btns", [])):
            active = (i > 0 and i <= rating) or (i == 0 and rating == 0)
            sb.configure(border_color=T.ACCENT if active else T.BORDER,
                         text_color=T.ACCENT if (i > 0 and i <= rating) else T.TEXT)

    def _set_current_rating(self, stars: int):
        self.params.rating = int(np.clip(stars, 0, 5))
        self._update_star_btns(self.params.rating)
        if self.series is not None and 0 <= self.idx < len(self.photos):
            self.series.set_override(self.photos[self.idx], self.params)
            self._refresh_strip_marks(); self._update_override_label()
        self._set_status(f"Ocena zdjęcia: {'★' * self.params.rating if self.params.rating else 'brak (0★)'}")

    def _auto_angle(self):
        r = self.fit.relight if self.fit is not None and self.fit.relight is not None else None
        return r.auto_angle if r is not None and r.auto_angle is not None else 120.0

    @staticmethod
    def _dir_name(a):
        names = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]   # where the light comes from (short: fixed-width box)
        return names[int(((a % 360)+22.5)//45) % 8]

    def _show_light(self):
        auto = (self.params.light_angle is None and abs(self.params.light_x) < 1e-4 and abs(self.params.light_y) < 1e-4)
        if auto:
            a = self._auto_angle()
        elif abs(self.params.light_x) > 1e-4 or abs(self.params.light_y) > 1e-4:
            a = float(np.degrees(np.arctan2(self.params.light_y, self.params.light_x)) % 360)
        else:
            a = float(self.params.light_angle or 0.0)
        self.light_val.configure(text=f"{int(a)}° {self._dir_name(a)}", text_color=T.MUTED if auto else T.TEXT)

    def _light(self, v):
        if self.params.light_angle is None and self.light_auto.get(): return
        ang = float(v) % 360
        self.params.light_angle = ang
        lx, ly = RL.angle_to_xy(ang)
        self.params.light_x, self.params.light_y = lx, ly
        for k, pv in (("light_x", lx), ("light_y", ly)):
            if k in self.sliders:
                s_, val_, kind_ = self.sliders[k]
                s_.set(pv * 100.0)
                val_.configure(text=self._fmt(pv * 100.0, kind_), text_color=self._val_color(k, pv))
        self._show_light(); self._changed()

    def _light_auto(self):
        self._commit_history()
        if self.light_auto.get():
            self.params.light_angle = None
            self.params.light_x = 0.0
            self.params.light_y = 0.0
            for k in ("light_x", "light_y"):
                if k in self.sliders:
                    s_, val_, kind_ = self.sliders[k]
                    s_.set(0); val_.configure(text="0", text_color=T.MUTED)
        else:
            a = self._auto_angle()
            self.params.light_angle = float(a)
        self._show_light(); self._changed(discrete=True)

    @staticmethod
    def _fmt(v, kind):
        if kind == "ev":
            return f"{v/100:+.2f} EV".replace(".", ",") if round(v) else "0,00 EV"
        return f"{int(round(v))}%" if kind == "pct" else f"{int(round(v)):+d}" if round(v) else "0"

    def _slider(self, key, v):
        s, val, kind = self.sliders[key]; pv = float(v)/100
        val.configure(text=self._fmt(v, kind), text_color=self._val_color(key, pv))
        setattr(self.params, key, pv)
        if key in ("light_x", "light_y"):
            if abs(self.params.light_x) > 1e-4 or abs(self.params.light_y) > 1e-4:
                self.light_auto.deselect()
                self.params.light_angle = float(np.degrees(np.arctan2(self.params.light_y, self.params.light_x)) % 360)
            self._show_light()
        self._set_pro_tip(key)
        self._changed()

    def _reset_slider(self, key):
        dv = getattr(_DEFAULTS, key)
        if abs(getattr(self.params, key) - dv) <= 1e-4: return
        self._commit_history()
        setattr(self.params, key, dv)
        s, val, kind = self.sliders[key]; v = dv*100; s.set(v)
        val.configure(text=self._fmt(v, kind), text_color=T.MUTED)
        if key in ("light_x", "light_y") and abs(self.params.light_x) <= 1e-4 and abs(self.params.light_y) <= 1e-4:
            self.params.light_angle = None; self.light_auto.select(); self._show_light()
        self._changed(discrete=True)

    # ================= undo / redo / copy / paste =================
    def _reset_history(self):
        if self._hist_after: self.after_cancel(self._hist_after); self._hist_after = None
        self._undo = []; self._redo = []; self._hist_anchor = self.params.to_dict()
        self._update_history_btns()

    def _commit_history(self):
        if self._hist_after: self.after_cancel(self._hist_after); self._hist_after = None
        cur = self.params.to_dict()
        if self._hist_anchor is not None and cur != self._hist_anchor:
            self._undo.append(self._hist_anchor)
            if len(self._undo) > 50: self._undo.pop(0)
            self._redo.clear()
            self._hist_anchor = cur
        self._update_history_btns()

    def _update_history_btns(self):
        if not hasattr(self, "undo_btn"): return
        can_u = bool(self._undo) or (self._hist_anchor is not None and self.params.to_dict() != self._hist_anchor)
        self.undo_btn.configure(state="normal" if can_u else "disabled", text_color=T.TEXT if can_u else T.MUTED)
        self.redo_btn.configure(state="normal" if self._redo else "disabled", text_color=T.TEXT if self._redo else T.MUTED)
        self.paste_btn.configure(state="normal" if self._clipboard else "disabled", text_color=T.TEXT if self._clipboard else T.MUTED)

    def undo(self):
        if self._hist_after: self.after_cancel(self._hist_after); self._hist_after = None
        cur = self.params.to_dict()
        if self._hist_anchor is not None and cur != self._hist_anchor:
            prev = self._hist_anchor
        elif self._undo:
            prev = self._undo.pop()
        else:
            return
        self._redo.append(cur)
        self.params = Params.from_dict(prev); self._hist_anchor = self.params.to_dict()
        self.detail_raw = None; self.view_res.detail = None
        self._load_params_into_ui(self.params); self._persist_override(); self.request_render()
        self._set_status("Cofnięto zmianę.")

    def redo(self):
        if not self._redo: return
        if self._hist_after: self.after_cancel(self._hist_after); self._hist_after = None
        self._undo.append(self.params.to_dict())
        nxt = self._redo.pop()
        self.params = Params.from_dict(nxt); self._hist_anchor = self.params.to_dict()
        self.detail_raw = None; self.view_res.detail = None
        self._load_params_into_ui(self.params); self._persist_override(); self.request_render()
        self._set_status("Ponowiono zmianę.")

    def copy_params(self):
        d = self.params.to_dict()
        self._clipboard = {k: d[k] for k in _COPY_KEYS}
        self._update_history_btns()
        self._set_status("Skopiowano ustawienia zdjęcia (Ctrl+Shift+V, aby wkleić).")

    def paste_params(self):
        if not self._clipboard: return
        self._commit_history()
        merged = {**self.params.to_dict(), **self._clipboard}
        self.params = Params.from_dict(merged)
        self._load_params_into_ui(self.params); self._changed(discrete=True)
        self._set_status("Wklejono ustawienia do bieżącego zdjęcia.")

    def reset_adjustments(self):
        self._commit_history()
        d = _DEFAULTS.to_dict()
        keep = {k: getattr(self.params, k) for k in ("strokes", "angle", "crop", "aspect")}
        self.params = Params.from_dict({**d, **keep})
        self._load_params_into_ui(self.params); self._changed(discrete=True)
        self._set_status("Wyzerowano suwaki bieżącego zdjęcia (Ctrl+Z, aby cofnąć).")

    def _persist_override(self):
        if self.series and self.idx >= 0:
            if self._save_after: self.after_cancel(self._save_after)
            p, ph = Params.from_dict(self.params.to_dict()), self.photos[self.idx]
            self._save_after = self.after(300, lambda: (self.series.set_override(ph, p), self._refresh_strip_marks(), self._update_override_label()))

    def _changed(self, render=True, discrete=False):
        if discrete:
            self._commit_history()
        else:
            if self._hist_after: self.after_cancel(self._hist_after)
            self._hist_after = self.after(450, self._commit_history)
            self._update_history_btns()
        self._persist_override()
        if render: self.request_render()

    def _update_override_label(self):
        own = bool(self.series and self.idx >= 0 and self.series.has_override(self.photos[self.idx]))
        self.override_lbl.configure(text="własne (.retusz.json)" if own else "ustawienia serii",
                                    text_color=T.ACCENT if own else T.MUTED)

    def reset_to_series(self):
        if not self.series or self.idx < 0: return
        self._commit_history()
        if self._save_after: self.after_cancel(self._save_after); self._save_after = None
        self.series.clear_override(self.photos[self.idx]); self.params = self.series.params_for(self.photos[self.idx])
        self._commit_history()
        self._load_params_into_ui(self.params); self._refresh_strip_marks(); self.request_render()

    def make_series_default(self):
        if not self.series: return
        self.series.params = Params.from_dict({**self.params.to_dict(), "strokes": []}); self.series.save()
        if self.idx >= 0 and not self.params.strokes and self.params.crop is None and not self.params.angle:
            if self._save_after: self.after_cancel(self._save_after); self._save_after = None
            self.series.clear_override(self.photos[self.idx])
        self._refresh_strip_marks(); self._update_override_label(); self._set_status("Zapisano jako domyślne dla całej serii.")

    # ================= brush =================
    def _brush_px(self):
        """Brush radius in original-image pixels (log slider 5..600 px)."""
        return BRUSH_MIN*(BRUSH_MAX/BRUSH_MIN)**float(self.brush_size.get())

    def _brush_label(self, key):
        lab = self.brush_vals[key]
        if key == "size": lab.configure(text=f"{int(round(self._brush_px()))} px")
        elif key == "soft": lab.configure(text=f"{int(round(self.brush_soft.get()))}%")
        else: lab.configure(text=f"{int(round(self.brush_str.get()))}%")

    def _brush_step(self, d):
        if isinstance(self.focus_get(), ctk.CTkEntry): return
        self.brush_size.set(float(np.clip(self.brush_size.get() + d*0.04, 0, 1))); self._brush_label("size")
        self._redraw_cursor()

    def _brush_on(self):
        return self.brush_mode.get() != BRUSH_MODES[0] and self.result is not None

    def _brush_color(self):
        return T.ACCENT if self.brush_mode.get() == "Rozjaśnij" else "#6FA8FF"

    def _brush_cursor(self, e):
        self._mouse = (e.x, e.y); v = self.view_res; v.delete("cursor")
        if not self._brush_on() or self._space or self.picking or self.tool == "crop": return
        r = max(2.0, self._brush_px()*v.cur_zoom())
        v.create_oval(e.x-r, e.y-r, e.x+r, e.y+r, outline=self._brush_color(), width=1.5, tags="cursor")
        if r > 6: v.create_oval(e.x-1.5, e.y-1.5, e.x+1.5, e.y+1.5, outline=self._brush_color(), tags="cursor")

    def _redraw_cursor(self):
        if getattr(self, "_mouse", None):
            class E: pass
            e = E(); e.x, e.y = self._mouse; self._brush_cursor(e)

    def _brush_press(self, e):
        self.view_res.focus_set()
        if self.picking: self._pick(e); return
        if self.tool == "crop": self._crop_press(e); return
        if hasattr(self, "canvas_mode") and self.canvas_mode.get() == "Podział" and not self._space and not self._brush_on():
            self._split_drag = True
            self.split_u = float(np.clip(self.view_res.to_norm(e.x, e.y)[0], 0.02, 0.98))
            self.view_res.split_u = self.split_u; self.view_res.redraw()
            return
        if self._space or not self._brush_on():
            self._pan_start(self.view_res, e); return
        self._stroke = [self.view_res.to_norm(e.x, e.y)]; self._last = (e.x, e.y)

    def _brush_move(self, e):
        if getattr(self, "_split_drag", False):
            self.split_u = float(np.clip(self.view_res.to_norm(e.x, e.y)[0], 0.02, 0.98))
            self.view_res.split_u = self.split_u; self.view_res.redraw()
            return
        if self.tool == "crop": self._crop_move(e); return
        if self._pan is not None: self._pan_move(self.view_res, e); return
        self._brush_cursor(e)
        if self._stroke is None: return
        self._stroke.append(self.view_res.to_norm(e.x, e.y))
        r = self._brush_px()*self.view_res.cur_zoom()
        self.view_res.create_line(*self._last, e.x, e.y, fill=self._brush_color(), width=max(1, r*1.2), capstyle="round",
                                  stipple="gray25", tags="stroke")
        self._last = (e.x, e.y)

    def _brush_release(self, e):
        if getattr(self, "_split_drag", False):
            self._split_drag = False; return
        if self.tool == "crop": self._crop_release(e); return
        if self._pan is not None: self._pan_end(); return
        if self._stroke is None: return
        W, H = self._frame_wh(); a = self.params.angle
        pts = [[float(np.clip(x, 0, 1)), float(np.clip(y, 0, 1))] for x, y in (G.frame_to_src(q, a, W, H) for q in self._stroke)]
        self._stroke = None
        self.view_res.delete("stroke")
        W = W*G.straighten_scale(W, H, a)                        # brush size in source pixels
        self._commit_history()
        self.params.strokes.append({"mode": "dodge" if self.brush_mode.get() == "Rozjaśnij" else "burn",
                                    "size": round(self._brush_px()/W, 5), "strength": round(self.brush_str.get()/100, 3),
                                    "softness": round(self.brush_soft.get()/100, 3),
                                    "points": pts[::max(1, len(pts)//300)]})
        self._changed(discrete=True)

    def undo_stroke(self):
        if self.params.strokes: self._commit_history(); self.params.strokes.pop(); self._changed(discrete=True)

    def clear_strokes(self):
        if self.params.strokes: self._commit_history(); self.params.strokes = []; self._changed(discrete=True)

    # ================= zoom / pan =================
    def _space_key(self, down):
        if self.compare.winfo_ismapped(): self.compare.key_space(down); return
        if isinstance(self.focus_get(), ctk.CTkEntry): return
        self._space = down
        for v in (self.view_src, self.view_res): v.configure(cursor="fleur" if down else "")
        if down: self.view_res.delete("cursor")

    def _wheel(self, v, e, d):
        st = e.state
        alt = bool(st & 0x20000) if sys.platform == "win32" else bool(st & 0x8)
        if alt: self._brush_step(d); return "break"
        if st & 0x4:                                   # Ctrl + wheel: zoom around the cursor
            v.zoom_at(1.25 if d > 0 else 0.8, e.x, e.y); self._redraw_cursor(); return "break"
        if v.zoom is not None:                         # wheel alone pans a zoomed image (Shift: sideways)
            (v.pan(d*80, 0) if st & 0x1 else v.pan(0, d*80)); return "break"

    def _pan_start(self, v, e):
        if v.zoom is None and not self._space: self._pan = None; return
        self._pan = (e.x, e.y)

    def _pan_move(self, v, e):
        if self._pan is None: return
        v.pan(e.x-self._pan[0], e.y-self._pan[1]); self._pan = (e.x, e.y)

    def _pan_end(self):
        self._pan = None

    def _view_changed(self, src_view):
        other = self.view_res if src_view is self.view_src else self.view_src
        other.set_view(src_view.zoom, src_view.center)
        self._sync_zoom_label(); self._schedule_detail()

    def zoom_fit(self):
        if isinstance(self.focus_get(), ctk.CTkEntry): return
        self.view_src.set_view(None, (0.5, 0.5), notify=True)

    def zoom_100(self):
        if isinstance(self.focus_get(), ctk.CTkEntry): return
        v = self.view_src; c = v.center if v.zoom else (0.5, 0.5)
        v.set_view(1.0, c, notify=True)

    def _sync_zoom_label(self):
        v = self.view_src
        self.zoom_lbl.configure(text="" if v.rgb is None else f"{int(round(v.cur_zoom()*100))}%")

    def _schedule_detail(self):
        """At zoom above the preview resolution, render the visible photo at full resolution (background)."""
        if self._detail_after: self.after_cancel(self._detail_after); self._detail_after = None
        if self.fit is None or not self.view_src.needs_detail(): return
        self._detail_after = self.after(250, self._render_detail)

    def _render_detail(self):
        self._detail_after = None
        if self.fit is None or self.idx < 0: return
        i, ph, fit, eng = self.idx, self.photos[self.idx], self.fit, self.engine
        params = Params.from_dict(self.params.to_dict())
        def work():
            fc = self.full_cache
            if fc is None or fc[0] != ph:
                rgb, _ = imageio.load(ph); fc = self.full_cache = (ph, rgb)
            res = eng.render(fc[1], fit, params)
            return fc[1], (G.apply(res, params.angle, None, interp=cv2.INTER_LINEAR) if params.angle else res)
        def done(r):
            if self.idx != i or params.angle != self.params.angle: return
            self.detail_raw = r[1]; self.view_src.set_detail(r[0]); self.view_res.set_detail(self._decorate(r[1]))
        self._set_status("Liczę pełną rozdzielczość…")
        self.submit(work, lambda r: (done(r), self._update_status()), kind="detail")

    # ================= presets =================
    def apply_preset_by_name(self, name: str):
        pr = next((p for p in self.presets if p.name == name), None)
        if pr is None: return
        if hasattr(self, "preset_menu"): self.preset_menu.set(pr.name)
        self._commit_history()
        self.params = pr.apply_to(self.params)
        self.detail_raw = None; self.view_res.detail = None
        self._load_params_into_ui(self.params); self._changed(discrete=True)
        self._set_status(f"Zastosowano preset: {pr.name}")

    def preset_to_series(self):
        name = self.preset_menu.get()
        pr = next((p for p in self.presets if p.name == name), None)
        if pr is None: return
        self._commit_history()
        self.params = pr.apply_to(self.params)
        if self.series is not None:
            self.series.params = pr.apply_to(self.series.params)
            self.series.save()
            for ph in self.photos:
                if self.series.has_override(ph):
                    self.series.set_override(ph, pr.apply_to(self.series.params_for(ph)))
        self.detail_raw = None; self.view_res.detail = None
        self._load_params_into_ui(self.params); self._refresh_strip_marks(); self._update_override_label()
        self._changed(discrete=True)
        self._set_status(f"Zastosowano preset dla całej serii: {pr.name}")

    def save_custom_preset_dialog(self):
        dlg = ctk.CTkInputDialog(text="Nazwa nowego presetu (zostanie zapisany w Twojej bibliotece):", title="Zapisz własny preset")
        name = dlg.get_input()
        if not name or not name.strip(): return
        pr = PR.save_preset(name.strip(), self.params)
        self.presets = PR.list_presets()
        self.preset_menu.configure(values=[p.name for p in self.presets])
        self.preset_menu.set(pr.name)
        self._set_status(f"Zapisano własny preset: {pr.name}")

    # ================= compare & library (Biblioteka G / Edycja D) =================
    def _disp(self, rgb):
        return CM.for_display(rgb, self.src_icc)

    def show_library(self):
        if self.compare.winfo_ismapped(): self.compare.pack_forget()
        if not self.library.winfo_ismapped():
            self.main.pack_forget()
            self.library.pack(fill="both", expand=True, before=self.status)
        self.library.refresh()

    def hide_library(self):
        if self.library.winfo_ismapped():
            self.library.pack_forget()
            self.main.pack(fill="both", expand=True, before=self.status)
            self._refresh_strip_marks()

    def toggle_library(self):
        if self.library.winfo_ismapped(): self.hide_library()
        else: self.show_library()

    def show_compare(self):
        if self.result is None: self._set_status("Najpierw otwórz zdjęcie."); return
        if self.library.winfo_ismapped(): self.library.pack_forget()
        self.main.pack_forget(); self.compare.pack(fill="both", expand=True, before=self.status)
        self.compare.set_images(self._disp(self.src_prev), self._disp(self.result))

    def hide_compare(self):
        self.compare.pack_forget(); self.main.pack(fill="both", expand=True, before=self.status)

    def _compare_full(self):
        if self.fit is None: return
        p, fit, eng, params = self.photos[self.idx], self.fit, self.engine, Params.from_dict(self.params.to_dict())
        def work():
            rgb, _ = imageio.load(p); return self._disp(rgb), self._disp(eng.render(rgb, fit, params))
        self.submit(work, lambda r: self.compare.set_full(*r), kind="full")

    # ================= export =================
    def _out_dir(self, photo):
        return Path(photo).parent / self.settings.output_subfolder

    def open_export_as(self):
        if not self.photos or self.engine is None:
            self._set_status("Najpierw otwórz zdjęcie lub folder."); return
        cur_ph = self.photos[max(0, self.idx)]
        def_dir = self._out_dir(cur_ph)
        def_scope = "series" if (hasattr(self, "library") and self.library.winfo_ismapped() and len(self.photos) > 1) else "current"
        ExportAsDialog(self, self.settings, def_dir, self._run_export_as, default_scope=def_scope)

    def _run_export_as(self, scope_key: str, opts: export.ExportOptions, out_dir_str: str):
        if not self.photos or self.engine is None: return
        cur_ph = self.photos[max(0, self.idx)]
        out_dir = Path(out_dir_str) if out_dir_str else self._out_dir(cur_ph)
        if scope_key == "current" and 0 <= self.idx < len(self.photos) and self.fit is not None:
            p, eng, ref, fit = self.photos[self.idx], self.engine, self.reference, self.fit
            params = Params.from_dict(self.params.to_dict())
            self._set_status(f"Eksportuję {p.name} ({export.FORMATS[opts.format]})…")
            self.submit(lambda: batch.export_one(eng, ref, p, params, out_dir, opts, fit=fit),
                        lambda dst: self._set_status(f"Zapisano: {dst}"))
            return
        if scope_key == "starred" and self.series is not None:
            photos = [ph for ph in self.photos if self.series.params_for(ph).rating >= 1]
            if not photos:
                self._set_status("Brak zdjęć z gwiazdką (★) — oceń zdjęcia w Bibliotece lub zakładce Detal."); return
        elif scope_key == "current":
            photos = [cur_ph]
        else:
            photos = list(self.photos)
        dlg = ProgressDialog(self, f"Eksport ({export.FORMATS[opts.format]}) · {len(photos)} zdjęć")
        eng, ref, series = self.engine, self.reference, self.series
        def prog(i, n, ph, sec): self._ui.put((lambda _: dlg.winfo_exists() and dlg.update_progress(i, n, f"{i}/{n} · {Path(ph).name} · {sec:.1f} s"), None))
        def work():
            res = batch.export_series(eng, ref, series, photos, out_dir, opts, prog, lambda: dlg.cancelled)
            ok = sum(1 for r in res if r[1]); err = [r for r in res if r[2]]
            msg = f"Gotowe: {ok} z {len(photos)} zapisanych w {out_dir}" + (f" · błędy: {len(err)}" if err else "")
            self._ui.put((lambda _: dlg.winfo_exists() and dlg.done(msg), None)); self._ui.put((lambda _: self._set_status(msg), None))
        threading.Thread(target=work, daemon=True).start()

    def save_current(self):
        if self.fit is None: self._set_status("Najpierw otwórz zdjęcie."); return
        p, eng, ref, fit = self.photos[self.idx], self.engine, self.reference, self.fit
        params, opts = Params.from_dict(self.params.to_dict()), self.settings.export_options()
        self._set_status(f"Zapisuję {p.name}…")
        def work():
            return batch.export_one(eng, ref, p, params, self._out_dir(p), opts, fit=fit)   # same fit as the preview
        self.submit(work, lambda dst: self._set_status(f"Zapisano: {dst}"))

    def export_series(self):
        if not self.photos or self.engine is None:
            self._set_status("Najpierw otwórz folder ze zdjęciami."); return
        dlg = ProgressDialog(self, f"Eksport serii · {len(self.photos)} zdjęć")
        photos, out_dir = list(self.photos), self._out_dir(self.photos[0])
        eng, ref, series, opts = self.engine, self.reference, self.series, self.settings.export_options()
        def prog(i, n, ph, sec): self._ui.put((lambda _: dlg.winfo_exists() and dlg.update_progress(i, n, f"{i}/{n} · {Path(ph).name} · {sec:.1f} s"), None))
        def work():
            res = batch.export_series(eng, ref, series, photos, out_dir, opts, prog, lambda: dlg.cancelled)
            ok = sum(1 for r in res if r[1]); err = [r for r in res if r[2]]
            msg = f"Gotowe: {ok} z {len(photos)} zapisanych w {out_dir}" + (f" · błędy: {len(err)}" if err else "")
            self._ui.put((lambda _: dlg.winfo_exists() and dlg.done(msg), None)); self._ui.put((lambda _: self._set_status(msg), None))
        threading.Thread(target=work, daemon=True).start()

    # ================= status =================
    def _set_status(self, text):
        self.status.configure(text=text)

    def _update_status(self):
        parts = []
        if self.fit is not None:
            parts.append(f"Twarze: {len(self.fit.faces_n)}")
            if self.fit.relight is not None: parts.append(f"3D: {self.fit.relight.method}")
        if self.src_size: parts.append(f"{self.src_size[0]}×{self.src_size[1]}")
        parts.append(f"podgląd {self.settings.preview_side} px")
        if self.engine_name: parts.append(f"silnik: {self.engine_name} · {DEV.summary() if DEV.ACTIVE else self.device}")
        if self.src_icc is not None: parts.append(f"profil: {CM.description(self.src_icc)}")
        parts.append(f"wzorzec: {self.reference.path.name}" if self.reference is not None else "tryb studyjny (bez wzorca)")
        self._set_status("  ·  ".join(parts))

def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="Retusz")
    ap.add_argument("path", nargs="?", help="zdjęcie lub folder")
    ap.add_argument("--reference", "-r", help="wzorzec")
    a = ap.parse_args(argv)
    RetuszApp(a.path, a.reference).mainloop()
