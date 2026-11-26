"""Retusz main window — Adobe Lightroom Classic 1-Column Studio UI & Effortless UX (v0.9.0)."""
from __future__ import annotations
import queue, threading, traceback, sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog
import numpy as np, cv2
import customtkinter as ctk
from PIL import Image
from . import theme as T
from .imageview import ImageView
from .compare import CompareView
from .library import LibraryView
from .histview import HistView
from .dialogs import (
    SettingsDialog, ModelDialog, ProgressDialog, ExportAsDialog,
    SelectiveCopyDialog, ShortcutsDialog,
)
from .. import imageio, models, batch, export, colorman as CM, presets as PR, __version__
from ..reference import Reference
from ..settings import Settings
from ..sidecar import Series
from ..engine.pipeline import Engine, Params
from ..engine import render as R, color as C, device as DEV, modflows as MF, relight as RL
from ..engine import histo as HI, wb as WB, geometry as G, colorchecker as CC, detail as DET, tone as TN

ENGINE_LABEL = {"ai": "AI (B6)", "ai_light": "AI lekki (B0)", "classic": "bez AI (IDT)"}
CANVAS_MODES = ["Wynik", "Przed | Po", "Podział", "Z wzorcem"]
SLIDER_TABS = ["Podstawowe", "Portret 3D", "Kolor HSL", "Detal", "Wzorzec"]
SLIDER_GROUPS = {
    SLIDER_TABS[0]: [  # (key, label, lo, hi, kind, row, col) — Authentic 1-Column Lightroom Classic Basic stack
        ("temp", "Temperatura", -100, 100, "num", 0, 0),
        ("tint", "Odcień", -100, 100, "num", 1, 0),
        ("exposure", "Ekspozycja", -200, 200, "ev", 2, 0),
        ("contrast", "Kontrast", -100, 100, "num", 3, 0),
        ("brightness", "Jasność", -100, 100, "num", 4, 0),
        ("highlights", "Światła", -100, 100, "num", 5, 0),
        ("shadows", "Cienie", -100, 100, "num", 6, 0),
        ("whites", "Biele", -100, 100, "num", 7, 0),
        ("blacks", "Czernie", -100, 100, "num", 8, 0),
        ("vibrance", "Jaskrawość", -100, 100, "num", 9, 0),
        ("saturation", "Nasycenie", -100, 100, "num", 10, 0),
    ],
    SLIDER_TABS[1]: [
        ("db", "Dodge & Burn", 0, 100, "pct", 0, 0),
        ("plastic", "Plastyka 3D", 0, 100, "pct", 1, 0),
        ("light_x", "Światło 3D: Poziom", -100, 100, "num", 2, 0),
        ("light_y", "Światło 3D: Pion", -100, 100, "num", 3, 0),
        ("bg_exposure", "Separacja tła 3D", -100, 100, "num", 4, 0),
        ("skin_hue", "Odcień skóry", -100, 100, "num", 5, 0),
        ("mixed_light", "Światło mieszane", 0, 100, "pct", 6, 0),
        ("skin_cast", "Czysta skóra", 0, 100, "pct", 7, 0),
        ("skin_even", "Równanie skóry", 0, 100, "pct", 8, 0),
        ("deshine", "Redukcja połysku", 0, 100, "pct", 9, 0),
        ("blemish", "Niedoskonałości", 0, 100, "pct", 10, 0),
        ("eyes_teeth", "Oczy i zęby", 0, 100, "pct", 11, 0),
    ],
    SLIDER_TABS[2]: [
        ("shadow_tone", "Ciepło cieni", -100, 100, "num", 0, 0),
        ("midtone_tone", "Ton półtonów", -100, 100, "num", 1, 0),
        ("highlight_tone", "Blask świateł", -100, 100, "num", 2, 0),
        ("grading_balance", "Balans gradingu", -100, 100, "num", 3, 0),
        ("hsl_green", "Zieleń (Plener)", -100, 100, "num", 4, 0),
        ("hsl_blue", "Błękit (Niebo/Biel)", -100, 100, "num", 5, 0),
    ],
    SLIDER_TABS[3]: [
        ("sharpen_amount", "Ostrość", 0, 100, "pct", 0, 0),
        ("texture", "Struktura", -100, 100, "num", 1, 0),
        ("clarity", "Przejrzystość (LLF)", -100, 100, "num", 2, 0),
        ("denoise", "Odszumianie OKLab", 0, 100, "pct", 3, 0),
        ("dehaze", "Zamglenie (Dehaze)", -100, 100, "num", 4, 0),
        ("vignette", "Winieta", 0, 100, "pct", 5, 0),
    ],
    SLIDER_TABS[4]: [
        ("strength", "Siła kolorów wzorca", 0, 100, "pct", 0, 0),
        ("skin", "Ochrona skóry", 0, 100, "pct", 1, 0),
        ("mask_hue_tol", "Maska AI: Odcień", -100, 100, "num", 2, 0),
        ("mask_protect", "Maska AI: Oczy i usta", 0, 100, "pct", 3, 0),
        ("wide", "Pełne barwy (Gamut)", 0, 100, "pct", 4, 0),
    ],
}

# Subtelny 2px pasek gradientu wyłącznie pod suwakami osi barwnych (styl makiety Lightroom Classic)
COLOR_BAR_KEYS = {
    "temp", "tint", "vibrance", "saturation",
    "skin_hue", "mixed_light",
    "shadow_tone", "midtone_tone", "highlight_tone", "hsl_green", "hsl_blue",
}

# Wizualne odstępy między sekcjami w zakładce Podstawowe (WB | Tonacja | Zakres | Obecność)
_SUBGROUP_BREAK_BEFORE = {
    ("Podstawowe", "exposure"),
    ("Podstawowe", "highlights"),
    ("Podstawowe", "vibrance"),
    ("Portret 3D", "skin_hue"),
    ("Portret 3D", "deshine"),
    ("Kolor HSL", "hsl_green"),
}

SLIDER_HINTS: dict[str, tuple[str, str, str, str]] = {
    "temp": ("◀ Chłodny (Błękit)", "Ciepły (Bursztyn) ▶", "#4C7FE0", "#E8A246"),
    "tint": ("◀ Zieleń", "Magenta (Róż) ▶", "#5CB868", "#D85E98"),
    "exposure": ("◀ Ciemniej (-EV)", "Jaśniej (+EV) ▶", "#9A9A9A", "#E0E0E0"),
    "contrast": ("◀ Miękki / Płaski", "Głęboki kontrast ▶", "#9A9A9A", "#E0E0E0"),
    "brightness": ("◀ Ciemniej półtony", "Jaśniej półtony ▶", "#9A9A9A", "#FFE082"),
    "highlights": ("◀ Odzyskaj światła", "Rozjaśnij światła ▶", "#90CAF9", "#FFE082"),
    "shadows": ("◀ Pogłęb cienie", "Otwórz cienie ▶", "#9A9A9A", "#FFE082"),
    "whites": ("◀ Punkt bieli −", "Punkt bieli + ▶", "#9A9A9A", "#E0E0E0"),
    "blacks": ("◀ Głęboka czerń", "Matowa czerń ▶", "#9A9A9A", "#E0E0E0"),
    "vibrance": ("◀ Stonowane", "Żywe (chroni skórę) ▶", "#6E6E78", "#D6A254"),
    "saturation": ("◀ Monochrom", "Pełne nasycenie ▶", "#6E6E78", "#C8629A"),
    "db": ("0% (wył.)", "Rzeźba i gładkość ▶", "#9A9A9A", "#F5A33B"),
    "plastic": ("0% (płasko)", "Bryła 3D ▶", "#9A9A9A", "#F5A33B"),
    "light_x": ("◀ z Lewej (-100)", "z Prawej (+100) ▶", "#90CAF9", "#FFE082"),
    "light_y": ("◀ z Dołu (blenda)", "z Góry (kluczowe) ▶", "#B0BEC5", "#FFE082"),
    "bg_exposure": ("◀ Przyciemnij tło", "Rozjaśnij tło ▶", "#78909C", "#FFF59D"),
    "skin_hue": ("◀ Róż / Czerwień", "Złoty / Żółty ▶", "#E57373", "#FFD54F"),
    "mixed_light": ("0% (Biel/Czerwień LED)", "Jednolita skóra 52° ▶", "#E05A6D", "#F0B878"),
    "skin_cast": ("0%", "Usuń rumień i szarość ▶", "#F48FB1", "#FFD54F"),
    "skin_even": ("0%", "Wyrównaj plamy skóry ▶", "#9A9A9A", "#FFD54F"),
    "deshine": ("0%", "Usuń tłusty połysk ▶", "#9A9A9A", "#90CAF9"),
    "blemish": ("0%", "Usuń wypryski (zachowaj pory) ▶", "#9A9A9A", "#81C784"),
    "eyes_teeth": ("0%", "Tęczówka i biel zębów ▶", "#9A9A9A", "#E0E0E0"),
    "shadow_tone": ("◀ Chłodny błękit cieni", "Ciepły bursztyn cieni ▶", "#5C8AE6", "#E09A48"),
    "midtone_tone": ("◀ Chłodne półtony", "Złociste półtony ▶", "#6898E8", "#E6AC58"),
    "highlight_tone": ("◀ Chłodne światła", "Złote kremowe światła ▶", "#90CAF9", "#FFD56B"),
    "grading_balance": ("◀ Więcej w cieniach", "Więcej w światłach ▶", "#78909C", "#FFE082"),
    "hsl_green": ("◀ Szlachetna zieleń", "Soczysta zieleń ▶", "#557A66", "#76C858"),
    "hsl_blue": ("◀ Czysta biel (bez błękitu)", "Głęboki błękit nieba ▶", "#B0BEC5", "#4294E8"),
    "sharpen_amount": ("0%", "Ostrość (Clamped USM) ▶", "#9A9A9A", "#F5A33B"),
    "texture": ("◀ Wygładź detal", "Podkreśl mikrodetal ▶", "#90CAF9", "#FFB74D"),
    "clarity": ("◀ Miękki kontur", "Kontrast lokalny LLF ▶", "#9A9A9A", "#E0E0E0"),
    "denoise": ("0%", "Usuń szum koloru/cieni ▶", "#9A9A9A", "#81C784"),
    "dehaze": ("◀ Mglisty blask", "Usuń mgłę / Głębia ▶", "#B0BEC5", "#64B5F6"),
    "vignette": ("0%", "Przyciemnij brzegi ▶", "#9A9A9A", "#E0E0E0"),
    "strength": ("0% (oryginał)", "100% (pełny wzorzec) ▶", "#9A9A9A", "#F5A33B"),
    "skin": ("0%", "100% (chroń karnację) ▶", "#9A9A9A", "#FFD54F"),
    "mask_hue_tol": ("◀ Wąska (twarz)", "Szeroka (szyja/cień) ▶", "#90CAF9", "#FFB74D"),
    "mask_protect": ("0%", "Chroń oczy i usta ▶", "#9A9A9A", "#81C784"),
    "wide": ("0% (sRGB)", "Szeroki gamut ▶", "#9A9A9A", "#F5A33B"),
}

PRO_TIPS: dict[str, str] = {
    "temp": "Temperatura (Bradford CAT): w lewo (−) chłodzi zdjęcie w stronę błękitu, w prawo (+) ociepla w stronę złotego bursztynu.",
    "tint": "Odcień (Tint): w lewo (−) koryguje magentę dodając zieleni, w prawo (+) usuwa zafarb zielony/fluorescencyjny dodając magenty.",
    "exposure": "Ekspozycja (EV w świetle liniowym): zmienia jasność jak czas naświetlania w aparacie z filmową ochroną świateł.",
    "contrast": "Kontrast (OKLab): krzywa S wokół średniej szarości bez przesuwania odcienia skóry.",
    "brightness": "Jasność (Półtony): rozjaśnia lub przyciemnia półtony bez ruszania punktu czerni i punktu bieli.",
    "highlights": "Światła: w lewo (−) odzyskuje detale w jasnych partiach (suknia, chmury, czoło), w prawo (+) rozświetla jasne tony.",
    "shadows": "Cienie: w prawo (+) wyciąga detale z cieni bez spłaszczania czerni, w lewo (−) pogłębia cienie.",
    "whites": "Biele: reguluje górny punkt bieli i blask najjaśniejszych partii.",
    "blacks": "Czernie: w lewo (−) pogłębia punkt czerni, w prawo (+) daje miękki, filmowy mat w cieniach.",
    "vibrance": "Jaskrawość (OKLab): inteligentnie podbija nienasycone barwy tła i ubrań, chroniąc skórę przed przesyceniem.",
    "saturation": "Nasycenie (OKLab): równomiernie zmienia intensywność wszystkich barw bez przesunięcia odcienia.",
    "db": "Dodge & Burn v2: automatycznie wyrównuje mikro-cienie pod oczami i na policzkach z zachowaniem pasma wysokich częstotliwości porów.",
    "plastic": "Plastyka 3D: modeluje trójwymiarową bryłę twarzy na mapie głębi Depth Anything V2 (rozświetla stronę światła, pogłębia cień).",
    "light_x": "Światło 3D (Lewo ↔ Prawo): przesuwa wirtualny softbox 3D w poziomie — w lewo (−) światło z lewej, w prawo (+) światło z prawej.",
    "light_y": "Światło 3D (Dół ↔ Góra): przesuwa wirtualny softbox 3D w pionie — w dół (−) odbicie od blendy, w górę (+) światło górne.",
    "bg_exposure": "Separacja tła 3D: w lewo (−) przyciemnia tło za postacią (efekt studyjny), w prawo (+) rozjaśnia tło. Twarz pozostaje chroniona.",
    "skin_hue": "Odcień skóry (OKLCh): w lewo (−) przesuwa karnację w stronę różu/czerwieni, w prawo (+) w stronę złotego/brzoskwiniowego odcienia.",
    "mixed_light": "Światło mieszane (LED / Błysk): ratuje twarz oświetloną dwoma światłami (pół białe od lampy, pół czerwone od LED) — scala do 52° OKLCh.",
    "skin_cast": "Czysta skóra: redukuje czerwony rumień (hemoglobinę) oraz ziemiste/sine zabarwienia cieni wokół osi 52° OKLCh.",
    "skin_even": "Równanie skóry: wyrównuje niskoczęstotliwościowe plamy barwne między twarzą, szyją i dłońmi bez rozmywania porów.",
    "deshine": "Redukcja połysku (De-Shine SUV): separacja odbicia lustrzanego usuwająca tłusty błysk z czoła i nosa z transferem mikro-tekstury.",
    "blemish": "Niedoskonałości: dwupasmowy retusz punktowych wyprysków i przebarwień chroniący mikro-strukturę porów.",
    "eyes_teeth": "Oczy i zęby: wzmacnia mikrokontrast tęczówki oraz naturalnie oczyszcza białka oczu i zęby na punktach anatomicznych.",
    "shadow_tone": "Ciepło cieni (3-Way Grading OKLab): w lewo (−) chłodny kinowy błękit w cieniach, w prawo (+) ciepły bursztyn.",
    "midtone_tone": "Ton półtonów (3-Way Grading OKLab): subtelne chłodzenie lub złocenie półtonów z automatyczną ochroną karnacji.",
    "highlight_tone": "Blask świateł (3-Way Grading OKLab): w lewo (−) chłodne światła, w prawo (+) kremowy blask złotej godziny.",
    "grading_balance": "Balans gradingu: przesuwa punkt podziału między tonowaniem cieni (−) a tonowaniem świateł (+).",
    "hsl_green": "Zieleń (Plener OKLCh): w lewo (−) zamienia jaskrawą trawę/liście w szlachetną, filmową zieleń, w prawo (+) ożywia roślinność.",
    "hsl_blue": "Błękit (Niebo/Biel OKLCh): w lewo (−) zdejmuje niebieski zafarb z białych koszul i sukni, w prawo (+) pogłębia błękit nieba.",
    "sharpen_amount": "Ostrość: wyostrzanie z morfologicznym ogranicznikiem obwódek (Clamped USM / Dekonwolucja RL / Krawędzie / Soft Focus).",
    "texture": "Struktura: w prawo (+) wydobywa drobny mikrodetal (rzęsy, tkaniny), w lewo (−) delikatnie zmiękcza wysokie częstotliwości.",
    "clarity": "Przejrzystość (Fast LLF): wieloskalowa piramida Laplace'a wzmacniająca kontrast lokalny z niskim poziomem obwódek.",
    "denoise": "Odszumianie OKLab: usuwa kolorowy szum matrycy (chroma) i ziarnistość w cieniach z zachowaniem krawędzi.",
    "dehaze": "Zamglenie (Dehaze): w prawo (+) usuwa mgiełkę i flarę pod światło, w lewo (−) dodaje miękką, powietrzną poświatę.",
    "vignette": "Winieta: przyciemnia brzegi kadru poza postacią, skupiając wzrok na portrecie.",
    "strength": "Siła kolorów wzorca: określa w ilu procentach zastosować dopasowanie barwne do wybranego wzorca lub tablicy ColorChecker.",
    "skin": "Ochrona skóry: chroni naturalną jasność i nasycenie karnacji przed zbyt agresywnym gradingiem wzorca.",
    "mask_hue_tol": "Maska AI — Odcień skóry (Mahalanobis): w lewo (−) zawęża maskę do policzków, w prawo (+) rozszerza na szyję i cienie.",
    "mask_protect": "Maska AI — Oczy i usta: wyklucza z maski skóry oczy, rzęsy oraz czerwień ust na podstawie 5 punktów anatomicznych YuNet.",
    "wide": "Pełne barwy: symuluje głębię barw monitora szeroko-gamutowego (Adobe RGB) w przestrzeni docelowej.",
}

_TAB_ALIASES = {
    "Światło i Kolor": SLIDER_TABS[0],
    "Wzorzec i Portret": SLIDER_TABS[1],
    "Portret i Plastyka": SLIDER_TABS[1],
    "Portret i 3D": SLIDER_TABS[1],
    "Detal i Ostrość": SLIDER_TABS[3],
    "Wzorzec (AI)": SLIDER_TABS[4],
    "▸ Wzorzec (AI)": SLIDER_TABS[4],
}

_HSL8_UI = (
    ("Czerwony", "#D85656"),
    ("Pomarańcz", "#E88E46"),
    ("Żółty", "#D4BE4A"),
    ("Zielony", "#5CB86E"),
    ("Turkus", "#48AEBE"),
    ("Niebieski", "#5882E4"),
    ("Fiolet", "#9A66DC"),
    ("Magenta", "#D65C98"),
)
_HSL_MODES = {"Nasycenie": "hsl_sat", "Odcień": "hsl_hue", "Jasność": "hsl_lum"}

_DEFAULTS = Params()
_COPY_KEYS = (
    "strength", "skin", "exposure", "contrast", "brightness", "dehaze", "shadows", "highlights", "whites", "blacks",
    "vignette", "db", "plastic", "clarity", "texture", "sharpen_amount", "sharpen_mode", "denoise",
    "light_angle", "light_x", "light_y", "bg_exposure", "saturation", "vibrance",
    "shadow_tone", "midtone_tone", "highlight_tone", "grading_balance",
    "hsl_green", "hsl_blue", "hsl_hue", "hsl_sat", "hsl_lum",
    "skin_even", "mixed_light", "skin_cast", "skin_hue", "deshine", "blemish", "eyes_teeth", "mask_hue_tol", "mask_protect",
    "wide", "temp", "tint",
)
SLIDER_AREA_H = 442
TOOLS = ["Kadr (C)", "Usuń (Q)", "Pędzel"]
TOOL_W, TOOL_H = 368, 144
PHOTO_TYPES = [(
    "Zdjęcia i RAW",
    "*.jpg *.jpeg *.png *.tif *.tiff *.webp "
    "*.nef *.NEF *.nrw *.NRW *.cr2 *.CR2 *.cr3 *.CR3 *.arw *.ARW *.dng *.DNG *.raf *.RAF *.orf *.ORF *.rw2 *.RW2 *.pef *.PEF"
)]
BRUSH_MIN, BRUSH_MAX = 5, 600   # brush radius in original-image pixels
HEAL_MIN, HEAL_MAX = 4, 140     # healing brush radius in original-image pixels
BRUSH_MODES = ["Wył.", "Rozjaśnij", "Przyciemnij"]


class RetuszApp(ctk.CTk):
    def __init__(self, path=None, reference=None):
        T.setup(); super().__init__(fg_color=T.BG)
        sw = max(1120, min(1440, self.winfo_screenwidth() - 16))
        sh = max(720, min(900, self.winfo_screenheight() - 40))
        self.title(f"Retusz {__version__} PRO"); self.geometry(f"{sw}x{sh}"); self.minsize(1120, 720)
        self.settings = Settings.load()
        DEV.MODE = self.settings.device
        DEV.ocl_configure(self.settings.device)
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
        self.filmstrip_open = False
        self._bypass_tab_saved: dict | None = None
        self._inline_entry: ctk.CTkEntry | None = None
        self.presets: list[PR.Preset] = PR.list_presets()
        self._undo: list[dict] = []; self._redo: list[dict] = []; self._hist_anchor = self.params.to_dict(); self._hist_after = None
        self.tool = "brush"; self.tool_open = False; self.hline_on = False; self._hline = None; self._crop_drag = None; self._crop_live = None
        self._geo_after = None
        threading.Thread(target=self._worker, daemon=True).start()
        self._build(); self._poll()

        # Keyboard & Lightroom Classic UX shortcuts
        self.bind("<Left>", lambda e: self.step(-1)); self.bind("<Right>", lambda e: self.step(1))
        self.bind("<Control-s>", lambda e: self.save_current())
        self.bind("<Control-z>", lambda e: self._key(self.undo))
        self.bind("<Control-y>", lambda e: self._key(self.redo))
        self.bind("<Control-Shift-Z>", lambda e: self._key(self.redo))
        self.bind("<Control-Shift-C>", lambda e: self._key(self.open_selective_copy))
        self.bind("<Control-Shift-c>", lambda e: self._key(self.open_selective_copy))
        self.bind("<Control-Shift-V>", lambda e: self._key(self.paste_params))
        self.bind("<Control-Shift-v>", lambda e: self._key(self.paste_params))
        self.bind("<backslash>", lambda e: self._key(self.toggle_before))
        self.bind("<F6>", lambda e: self._key(self.toggle_filmstrip))
        self.bind("<F1>", lambda e: self._key(self.open_shortcuts))
        self.bind("<question>", lambda e: self._key(self.open_shortcuts))
        self.bind("<KeyPress-space>", lambda e: self._space_key(True)); self.bind("<KeyRelease-space>", lambda e: self._space_key(False))
        self.bind("<bracketleft>", lambda e: self._brush_step(-1)); self.bind("<bracketright>", lambda e: self._brush_step(1))
        self.bind("<Control-0>", lambda e: self.zoom_fit()); self.bind("<Control-1>", lambda e: self.zoom_100())
        self.bind("<Control-Shift-E>", lambda e: self.open_export_as()); self.bind("<Control-Shift-e>", lambda e: self.open_export_as())
        for k, fn in (
            ("w", self.toggle_pick),
            ("c", self.toggle_crop_tool),
            ("r", self.toggle_crop_tool),
            ("q", self.toggle_heal_tool),
            ("k", self.toggle_brush_tool),
            ("j", self.toggle_clip),
            ("m", self.toggle_mask),
            ("g", self.toggle_library),
            ("d", self.hide_library),
            ("y", self.toggle_side_by_side),
            ("p", lambda: self._set_current_flag(1)),
            ("x", lambda: self._set_current_flag(-1)),
            ("u", lambda: self._set_current_flag(0)),
        ):
            self.bind(f"<KeyPress-{k}>", lambda e, fn=fn: self._key(fn))
            self.bind(f"<KeyPress-{k.upper()}>", lambda e, fn=fn: self._key(fn))
        for st_n in range(6):
            self.bind(f"<KeyPress-{st_n}>", lambda e, s=st_n: self._rating_key(e, s))
        self.bind("<Return>", lambda e: self._key(self._crop_done)); self.bind("<Escape>", lambda e: self._key(self._escape))
        self.submit(lambda: (R.warmup(), None)[1], None)
        self.setup_engine()
        if path: self.after(100, lambda: self.open_path(Path(path), Path(reference) if reference else None))
        self.after(700, self._first_run)

    # ================= layout =================
    def _build(self):
        # 1. Minimalist Mockup Top Bar: [Otwórz ▾] [Presety ▾]  |  [Biblioteka (G) | Edycja (D) | Grading z AI]  |  [Zapisz] [Eksportuj…]
        top = ctk.CTkFrame(self, fg_color=T.BG, height=38)
        top.pack(fill="x", padx=12, pady=(6, 4))
        top.pack_propagate(False)

        left_bar = ctk.CTkFrame(top, fg_color=T.BG)
        left_bar.pack(side="left", fill="y")

        self.open_btn = T.button(left_bar, "Otwórz ▾", self._show_open_menu, accent=True, width=88)
        self.open_btn.configure(height=28, font=T.font(12, "bold"))
        self.open_btn.pack(side="left", pady=4)

        self.preset_menu = ctk.CTkOptionMenu(
            left_bar, values=[p.name for p in self.presets], command=self.apply_preset_by_name,
            width=148, height=28, fg_color=T.BTN, button_color=T.BTN, button_hover_color=T.BTN_HOVER,
            text_color=T.TEXT, font=T.font(11), dropdown_font=T.font(11), corner_radius=7
        )
        self.preset_menu.set("Presety ▾")
        self.preset_menu.pack(side="left", padx=(6, 3), pady=4)

        pb_save = T.button(left_bar, "+", self.save_custom_preset_dialog, width=26)
        pb_save.configure(height=28, font=T.font(12, "bold")); pb_save.pack(side="left", padx=(0, 2), pady=4)
        pb_ser = T.button(left_bar, "→seria", self.preset_to_series, width=48)
        pb_ser.configure(height=28, font=T.font(10)); pb_ser.pack(side="left", padx=(0, 4), pady=4)
        pb_set = T.button(left_bar, "⚙", self.open_settings, width=28)
        pb_set.configure(height=28, font=T.font(12)); pb_set.pack(side="left", padx=(0, 2), pady=4)
        pb_help = T.button(left_bar, "?", self.open_shortcuts, width=26)
        pb_help.configure(height=28, font=T.font(11, "bold")); pb_help.pack(side="left", pady=4)

        right_bar = ctk.CTkFrame(top, fg_color=T.BG)
        right_bar.pack(side="right", fill="y")
        b_exp = T.button(right_bar, "Eksportuj…", self.open_export_as, accent=True, width=96)
        b_exp.configure(height=28, font=T.font(12, "bold")); b_exp.pack(side="right", pady=4)
        b_sav = T.button(right_bar, "Zapisz", self.save_current, accent=True, width=68)
        b_sav.configure(height=28, font=T.font(12, "bold")); b_sav.pack(side="right", padx=(0, 6), pady=4)

        center_wrap = ctk.CTkFrame(top, fg_color=T.BG)
        center_wrap.pack(side="left", fill="both", expand=True)
        mod_bar = ctk.CTkFrame(center_wrap, fg_color=T.PANEL, corner_radius=8, border_width=1, border_color=T.BORDER)
        mod_bar.pack(expand=True, pady=4)
        self.mod_lib_btn = ctk.CTkButton(
            mod_bar, text="Biblioteka (G)", command=self.show_library, width=102, height=26, corner_radius=6,
            fg_color="transparent", hover_color=T.BTN_HOVER, text_color=T.MUTED, font=T.font(11)
        )
        self.mod_lib_btn.pack(side="left", padx=(3, 1), pady=2)
        self.mod_dev_btn = ctk.CTkButton(
            mod_bar, text="Edycja (D)", command=self.hide_library, width=84, height=26, corner_radius=6,
            fg_color="#383842", hover_color="#444450", text_color=T.TEXT, font=T.font(11, "bold")
        )
        self.mod_dev_btn.pack(side="left", padx=1, pady=2)
        self.mod_cmp_btn = ctk.CTkButton(
            mod_bar, text="Porównaj", command=self.show_compare, width=74, height=26, corner_radius=6,
            fg_color="transparent", hover_color=T.BTN_HOVER, text_color=T.MUTED, font=T.font(11)
        )
        self.mod_cmp_btn.pack(side="left", padx=1, pady=2)
        self.mod_ai_btn = ctk.CTkButton(
            mod_bar, text="Grading z AI", command=self.open_transfer, width=92, height=26, corner_radius=6,
            fg_color="transparent", hover_color=T.BTN_HOVER, text_color=T.MUTED, font=T.font(11)
        )
        self.mod_ai_btn.pack(side="left", padx=(1, 3), pady=2)

        self.main = ctk.CTkFrame(self, fg_color=T.BG); self.main.pack(fill="both", expand=True)
        self.compare = CompareView(self, self.hide_compare, self.save_current, self._compare_full)
        self.library = LibraryView(self, self)

        # Collapsible Left Filmstrip (F6 / '▣' button on Loupe toolbar — starts collapsed so portrait is centered!)
        self.side_panel = ctk.CTkFrame(self.main, fg_color=T.BG, width=148)
        self.series_lbl = T.label(self.side_panel, "Seria", 12, weight="bold")
        self.series_lbl.pack(anchor="w", pady=(0, 4))
        self.strip = ctk.CTkScrollableFrame(self.side_panel, fg_color=T.PANEL, width=132, corner_radius=8)
        self.strip.pack(fill="y", expand=True)

        # Right-Side Authentic Single-Column Adobe Lightroom Classic Panel (384 px)
        dock = ctk.CTkFrame(self.main, fg_color=T.CARD, width=384, corner_radius=10, border_width=1, border_color=T.BORDER)
        dock.pack(side="right", fill="y", padx=(0, 12), pady=(0, 2))
        dock.pack_propagate(False)

        # Central Portrait Viewport + Bottom Lightroom Loupe Toolbar ('T' bar)
        self.body_panel = ctk.CTkFrame(self.main, fg_color=T.BG)
        self.body_panel.pack(side="left", fill="both", expand=True, padx=(12, 10))
        self.imgs = ctk.CTkFrame(self.body_panel, fg_color=T.CANVAS_BG, corner_radius=8)
        self.imgs.pack(fill="both", expand=True)
        self.imgs.grid_rowconfigure(1, weight=1)
        self.src_hdr = T.label(self.imgs, "Przed (Źródło)", 11, T.MUTED)
        self.res_hdr = T.label(self.imgs, "Wynik", 11, T.MUTED)
        self.ref_hdr = T.label(self.imgs, "Wzorzec (opcjonalny)", 11, T.MUTED)
        self.view_src = ImageView(self.imgs, "Otwórz zdjęcie lub folder")
        self.view_res = ImageView(self.imgs, "Otwórz zdjęcie lub folder — układ studyjny Lightroom Classic Pro.")
        self.view_ref = ImageView(self.imgs, "Opcjonalny wzorzec (AI Color Match)\nWybierz wzorzec w zakładce „Wzorzec” po prawej")

        # Bottom Lightroom Loupe Toolbar directly under the photo canvas (matches mockup 1:1)
        loupe_bar = ctk.CTkFrame(self.body_panel, fg_color=T.PANEL, height=34, corner_radius=7)
        loupe_bar.pack(fill="x", pady=(4, 0))
        loupe_bar.pack_propagate(False)

        self.film_btn = T.button(loupe_bar, "▣", self.toggle_filmstrip, width=28)
        self.film_btn.configure(height=24, font=T.font(12)); self.film_btn.pack(side="left", padx=(5, 2), pady=5)
        b_prev = T.button(loupe_bar, "◀", lambda: self.step(-1), width=24)
        b_prev.configure(height=24, font=T.font(10)); b_prev.pack(side="left", padx=1, pady=5)
        b_next = T.button(loupe_bar, "▶", lambda: self.step(1), width=24)
        b_next.configure(height=24, font=T.font(10)); b_next.pack(side="left", padx=(1, 4), pady=5)

        self.canvas_mode = T.segmented(loupe_bar, CANVAS_MODES, command=self._set_canvas_mode)
        self.canvas_mode.configure(font=T.font(11), height=25)
        self.canvas_mode.set("Wynik" if not self.ref_enabled else "Z wzorcem")
        self.canvas_mode.pack(side="left", padx=(0, 6), pady=4)

        zb = ctk.CTkFrame(loupe_bar, fg_color=T.PANEL); zb.pack(side="right", padx=(0, 6))
        self.flag_pick_btn = T.button(zb, "⚑", lambda: self._set_current_flag(1 if self.params.flag != 1 else 0), width=24)
        self.flag_pick_btn.configure(height=23, font=T.font(11), corner_radius=5); self.flag_pick_btn.pack(side="left", padx=(0, 2))
        self.flag_rej_btn = T.button(zb, "✕", lambda: self._set_current_flag(-1 if self.params.flag != -1 else 0), width=24)
        self.flag_rej_btn.configure(height=23, font=T.font(10), corner_radius=5); self.flag_rej_btn.pack(side="left", padx=(0, 5))

        self.star_btns = []
        for st_i in range(6):
            txt = "0" if st_i == 0 else "★"
            sb = T.button(zb, txt, lambda s=st_i: self._set_current_rating(s), width=21)
            sb.configure(height=23, font=T.font(11 if st_i else 9), corner_radius=4, border_width=0, fg_color=T.PANEL)
            sb.pack(side="left", padx=0)
            self.star_btns.append(sb)

        self.before_btn = T.button(zb, "Przed/Po", self.toggle_before, width=64)
        self.before_btn.configure(height=24, font=T.font(11)); self.before_btn.pack(side="left", padx=(6, 4))
        b_fit = T.button(zb, "Dopasuj", self.zoom_fit, width=58)
        b_fit.configure(height=24, font=T.font(11)); b_fit.pack(side="left", padx=(0, 3))
        b_100 = T.button(zb, "1:1", self.zoom_100, width=34)
        b_100.configure(height=24, font=T.font(11)); b_100.pack(side="left", padx=(0, 3))
        zbox = ctk.CTkFrame(zb, fg_color=T.PANEL, width=42, height=24); zbox.pack(side="left"); zbox.pack_propagate(False)
        self.zoom_lbl = T.label(zbox, "", 11, T.MUTED, anchor="e"); self.zoom_lbl.pack(fill="both", expand=True)

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

        # ---- Right Dock Section 1: Interactive Histogram + Circular OKLCh Skin Vectorscope (52°) ----
        top_dock = ctk.CTkFrame(dock, fg_color=T.CARD)
        top_dock.pack(fill="x", padx=8, pady=(6, 1))
        self.hist = HistView(
            top_dock,
            on_toggle=self.toggle_clip,
            on_zone_drag=self._hist_zone_drag,
            on_zone_end=self._commit_history,
            width=366,
            height=80,
        )
        self.hist.pack(fill="x")
        h_meta = ctk.CTkFrame(top_dock, fg_color=T.CARD, height=18)
        h_meta.pack(fill="x", pady=(2, 0)); h_meta.pack_propagate(False)
        hb = ctk.CTkFrame(h_meta, fg_color=T.CARD, width=188, height=16); hb.pack(side="left"); hb.pack_propagate(False)
        self.clip_lbl = T.label(hb, "Cienie – · Światła –", 10, T.MUTED, anchor="w"); self.clip_lbl.pack(fill="both", expand=True)
        self.clip_sw = ctk.CTkSwitch(
            h_meta, text="Przepalenia (J)", progress_color=T.ACCENT, text_color=T.MUTED, font=T.font(10),
            width=102, switch_width=24, switch_height=12, command=lambda: self.toggle_clip(self.clip_sw.get() == 1)
        )
        self.clip_sw.pack(side="right")

        # ---- Right Dock Section 2: Mockup Local Tool Strip under Histogram (Kadr | Usuń | Pędzel | Pipeta | Maska AI) ----
        tool_strip = ctk.CTkFrame(dock, fg_color=T.PANEL, corner_radius=7, height=32)
        tool_strip.pack(fill="x", padx=8, pady=(3, 4))
        tool_strip.pack_propagate(False)
        self.tool_tabs = T.segmented(tool_strip, TOOLS, command=self._tool_tab, dynamic_resizing=False, width=196)
        self.tool_tabs.configure(height=24, font=T.font(10))
        self.tool_tabs.set("")
        self.tool_tabs.pack(side="left", padx=(3, 3), pady=4)
        for t_val, t_btn in getattr(self.tool_tabs, "_buttons_dict", {}).items():
            t_btn.configure(command=lambda v=t_val: self._toggle_tool_segment(v))

        self.pick_btn = T.button(tool_strip, "Pipeta (W)", self.toggle_pick, width=76)
        self.pick_btn.configure(height=24, font=T.font(10), corner_radius=6); self.pick_btn.pack(side="left", padx=(0, 3), pady=4)
        self.mask_btn = T.button(tool_strip, "Maska AI (M)", self.toggle_mask, width=84)
        self.mask_btn.configure(height=24, font=T.font(10), corner_radius=6); self.mask_btn.pack(side="right", padx=(0, 3), pady=4)
        self._style_mask_btn()

        # ---- Right Dock Section 3: 5 Clean Develop Tabs & Calm Neutral Sliders ----
        sl = ctk.CTkFrame(dock, fg_color=T.CARD)
        sl.pack(fill="x", padx=8, pady=(0, 2))

        tab_row = ctk.CTkFrame(sl, fg_color=T.CARD)
        tab_row.pack(fill="x", pady=(0, 4))
        self.slider_tabs = T.segmented(tab_row, SLIDER_TABS, command=self._slider_tab, dynamic_resizing=False, width=316)
        self.slider_tabs.configure(font=T.font(10), height=26)
        self.slider_tabs.set(SLIDER_TABS[0]); self.slider_tabs.pack(side="left")

        self.tab_reset_btn = T.button(tab_row, "↺", self._reset_active_tab, width=23)
        self.tab_reset_btn.configure(height=26, font=T.font(11), corner_radius=6); self.tab_reset_btn.pack(side="right")
        self.tab_eye_btn = T.button(tab_row, "👁", self._toggle_tab_bypass, width=23)
        self.tab_eye_btn.configure(height=26, font=T.font(11), corner_radius=6); self.tab_eye_btn.pack(side="right", padx=(0, 2))

        sl_area = ctk.CTkFrame(sl, fg_color=T.CARD, height=SLIDER_AREA_H)
        sl_area.pack(fill="x", expand=True); sl_area.pack_propagate(False)
        self.sl_area = sl_area
        self.slider_frames = {}
        self.sliders = {}
        self.slider_labels: dict[str, ctk.CTkLabel] = {}
        self.slider_boxes: dict[str, ctk.CTkFrame] = {}

        def value_box(parent, width=52):
            box = ctk.CTkFrame(parent, fg_color=T.CARD, width=width, height=18); box.pack_propagate(False)
            lab = T.label(box, "", 11, T.MUTED, anchor="e"); lab.pack(fill="both", expand=True); return box, lab

        for tname in SLIDER_TABS:
            f = ctk.CTkFrame(sl_area, fg_color=T.CARD)
            f.place(x=0, y=0, relwidth=1, relheight=1)
            f.grid_columnconfigure(0, weight=1)
            self.slider_frames[tname] = f
            row_offset = 1 if tname == "Podstawowe" else 0
            for key, text, lo, hi, kind, r_, c_ in SLIDER_GROUPS[tname]:
                top_pad = 6 if (tname, key) in _SUBGROUP_BREAK_BEFORE else 1
                row_f = ctk.CTkFrame(f, fg_color=T.CARD)
                row_f.grid(row=r_ + row_offset, column=0, sticky="ew", padx=4, pady=(top_pad, 1))
                row_f.grid_columnconfigure(1, weight=1)
                lbox = ctk.CTkFrame(row_f, fg_color=T.CARD, width=124, height=18)
                lbox.grid(row=0, column=0, sticky="w"); lbox.pack_propagate(False)
                lbl = T.label(lbox, text, 11, anchor="w"); lbl.pack(fill="both", expand=True)
                s = T.lr_slider(row_f, lo, hi, number_of_steps=hi-lo, width=180, command=lambda v, k=key: self._slider(k, v))
                s.grid(row=0, column=1, sticky="ew", padx=(4, 4))
                box, val = value_box(row_f, 52); box.grid(row=0, column=2, sticky="e")
                if key in COLOR_BAR_KEYS and key in SLIDER_HINTS:
                    _, _, l_col, r_col = SLIDER_HINTS[key]
                    gbar = ctk.CTkFrame(row_f, fg_color=T.CARD, height=2)
                    gbar.grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(0, 1))
                    gbar.grid_columnconfigure(0, weight=1); gbar.grid_columnconfigure(1, weight=1)
                    ctk.CTkFrame(gbar, fg_color=l_col, height=2, corner_radius=0).grid(row=0, column=0, sticky="ew")
                    ctk.CTkFrame(gbar, fg_color=r_col, height=2, corner_radius=0).grid(row=0, column=1, sticky="ew")
                self.sliders[key] = (s, val, kind)
                self.slider_labels[key] = lbl
                self.slider_boxes[key] = box
                self._bind_slider_ux(key, row_f, lbox, lbl, box, val, s, lo, hi)

        for alias, target in _TAB_ALIASES.items():
            self.slider_frames[alias] = self.slider_frames[target]

        # ---- Tab 0 ('Podstawowe') Top Action Bar: Full-width Mockup '★ Auto Ton AI' + 'Auto WB' + 'WB→seria' ----
        f0 = self.slider_frames["Podstawowe"]
        at_bar = ctk.CTkFrame(f0, fg_color=T.CARD)
        at_bar.grid(row=0, column=0, sticky="ew", padx=4, pady=(1, 6))
        at_bar.grid_columnconfigure(0, weight=1)
        self.auto_tone_btn = ctk.CTkButton(
            at_bar, text="★ Auto Ton AI", command=self.auto_tone_ai, height=28, corner_radius=7,
            fg_color="#2F2F37", hover_color="#3C3C46", border_width=1, border_color="#4A4A56",
            text_color=T.TEXT, font=T.font(12, "bold")
        )
        self.auto_tone_btn.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        b_awb = T.button(at_bar, "Auto WB", self.auto_wb_ai, width=64)
        b_awb.configure(height=28, font=T.font(10)); b_awb.grid(row=0, column=1, padx=(0, 3))
        b_wbs = T.button(at_bar, "WB→seria", self.sync_wb, width=64)
        b_wbs.configure(height=28, font=T.font(10)); b_wbs.grid(row=0, column=2)

        # ---- Tab 1 ('Portret 3D') Footer: 3D Light Auto switch & angle readout ----
        f1 = self.slider_frames["Portret 3D"]
        pcell = ctk.CTkFrame(f1, fg_color=T.PANEL, corner_radius=6)
        pcell.grid(row=12, column=0, sticky="ew", padx=4, pady=(5, 2))
        T.label(pcell, "Światło 3D:", 11, T.MUTED).pack(side="left", padx=(8, 6), pady=4)
        self.light_auto = ctk.CTkSwitch(
            pcell, text="Auto z analizy 3D", progress_color=T.ACCENT, text_color=T.TEXT, width=136,
            switch_width=24, switch_height=12, font=T.font(10), command=self._light_auto
        )
        self.light_auto.pack(side="left", pady=4)
        lbox_v, self.light_val = value_box(pcell, 58)
        lbox_v.configure(fg_color=T.PANEL); lbox_v.pack(side="right", padx=(0, 8), pady=4)
        self.light = self.sliders["light_x"][0]

        # ---- Tab 2 ('Kolor HSL') Footer: Full 8-Channel OKLCh HSL Mixer (Nasycenie | Odcień | Jasność) ----
        f_hsl = self.slider_frames["Kolor HSL"]
        hsl_card = ctk.CTkFrame(f_hsl, fg_color=T.PANEL, corner_radius=7)
        hsl_card.grid(row=10, column=0, sticky="ew", padx=4, pady=(6, 2))
        hsl_card.grid_columnconfigure(0, weight=1)
        hsl_hdr = ctk.CTkFrame(hsl_card, fg_color=T.PANEL)
        hsl_hdr.grid(row=0, column=0, sticky="ew", padx=6, pady=(4, 3))
        self.hsl_mode_seg = T.segmented(
            hsl_hdr, list(_HSL_MODES.keys()), command=lambda _: self._sync_hsl8_ui(), width=236
        )
        self.hsl_mode_seg.configure(height=23, font=T.font(10))
        self.hsl_mode_seg.set("Nasycenie")
        self.hsl_mode_seg.pack(side="left")
        b_rhsl = T.button(hsl_hdr, "Zeruj HSL", self.reset_hsl8, width=68)
        b_rhsl.configure(height=22, font=T.font(10)); b_rhsl.pack(side="right")

        self.hsl8_sliders: list[tuple[ctk.CTkSlider, ctk.CTkLabel]] = []
        for idx_c, (cname, chex) in enumerate(_HSL8_UI):
            cr_f = ctk.CTkFrame(hsl_card, fg_color=T.PANEL)
            cr_f.grid(row=idx_c + 1, column=0, sticky="ew", padx=6, pady=1)
            cr_f.grid_columnconfigure(2, weight=1)
            dot = ctk.CTkFrame(cr_f, fg_color=chex, width=8, height=8, corner_radius=4)
            dot.grid(row=0, column=0, padx=(0, 5))
            clb_box = ctk.CTkFrame(cr_f, fg_color=T.PANEL, width=82, height=17); clb_box.pack_propagate(False)
            clb_box.grid(row=0, column=1, sticky="w")
            clb = T.label(clb_box, cname, 10, T.TEXT, anchor="w"); clb.pack(fill="both", expand=True)
            c_sl = T.lr_slider(cr_f, -100, 100, number_of_steps=200, width=186, command=lambda v, i=idx_c: self._on_hsl8(i, v))
            c_sl.configure(height=12)
            c_sl.grid(row=0, column=2, sticky="ew", padx=(4, 4))
            c_vbox = ctk.CTkFrame(cr_f, fg_color=T.PANEL, width=38, height=17); c_vbox.pack_propagate(False)
            c_vbox.grid(row=0, column=3, sticky="e")
            c_val = T.label(c_vbox, "0", 10, T.MUTED, anchor="e"); c_val.pack(fill="both", expand=True)
            for w_ in (cr_f, clb, c_val, c_sl):
                w_.bind("<Double-Button-1>", lambda e, i=idx_c: self._reset_hsl8_chan(i))
            c_sl.bind("<MouseWheel>", lambda e, i=idx_c: self._wheel_hsl8(i, 1 if e.delta > 0 else -1, e.state))
            c_sl.bind("<Button-4>", lambda e, i=idx_c: self._wheel_hsl8(i, 1, e.state))
            c_sl.bind("<Button-5>", lambda e, i=idx_c: self._wheel_hsl8(i, -1, e.state))
            self.hsl8_sliders.append((c_sl, c_val))

        # ---- Tab 3 ('Detal') Footer: Sharpening mode selector ----
        f_det = self.slider_frames["Detal"]
        sh_cell = ctk.CTkFrame(f_det, fg_color=T.PANEL, corner_radius=6)
        sh_cell.grid(row=10, column=0, sticky="ew", padx=4, pady=(8, 2))
        T.label(sh_cell, "Algorytm wyostrzania:", 11, T.MUTED, anchor="w").pack(side="left", padx=(8, 6), pady=5)
        self.sharpen_mode_menu = ctk.CTkOptionMenu(
            sh_cell, values=list(DET.SHARPEN_MODES.values()), command=self._set_sharpen_mode,
            width=206, height=24, fg_color=T.BTN, button_color=T.BTN, button_hover_color=T.BTN_HOVER,
            text_color=T.TEXT, font=T.font(10), dropdown_font=T.font(11), corner_radius=6
        )
        self.sharpen_mode_menu.set(DET.SHARPEN_MODES["portrait"])
        self.sharpen_mode_menu.pack(side="right", padx=(0, 6), pady=5)

        # ---- Tab 4 ('Wzorzec') Footer: Reference Match, ColorChecker 24 & Per-Photo Reference Pool ----
        f2 = self.slider_frames["Wzorzec"]
        rc = ctk.CTkFrame(f2, fg_color=T.PANEL, corner_radius=6)
        rc.grid(row=5, column=0, sticky="ew", padx=4, pady=(8, 3))
        T.label(rc, "Wzorzec serii (AI):", 11, anchor="w", weight="bold").pack(side="left", padx=(8, 6), pady=5)
        self.clear_ref_btn = T.button(rc, "Wyłącz", self.clear_reference, width=68); self.clear_ref_btn.configure(height=24, font=T.font(10))
        self.clear_ref_btn.pack(side="right", padx=(0, 6), pady=5)
        b_ref = T.button(rc, "Wybierz wzorzec…", self.choose_reference, accent=True, width=124); b_ref.configure(height=24, font=T.font(10, "bold"))
        b_ref.pack(side="right", padx=(0, 4), pady=5)

        cc_cell = ctk.CTkFrame(f2, fg_color=T.PANEL, corner_radius=6)
        cc_cell.grid(row=6, column=0, sticky="ew", padx=4, pady=3)
        T.label(cc_cell, "ColorChecker (24 pola):", 11, anchor="w", weight="bold").pack(side="left", padx=(8, 6), pady=5)
        b_cc_cur = T.button(cc_cell, "Z bieżącego kadru", self.calibrate_current_colorchecker, width=112); b_cc_cur.configure(height=24, font=T.font(10))
        b_cc_cur.pack(side="right", padx=(0, 6), pady=5)
        b_cc = T.button(cc_cell, "Z pliku…", self.choose_colorchecker, width=78); b_cc.configure(height=24, font=T.font(10))
        b_cc.pack(side="right", padx=(0, 4), pady=5)

        pr_cell = ctk.CTkFrame(f2, fg_color=T.PANEL, corner_radius=6)
        pr_cell.grid(row=7, column=0, sticky="ew", padx=4, pady=3)
        pr_cell.grid_columnconfigure(0, weight=1)
        T.label(pr_cell, "Wzorzec TEGO zdjęcia (indywidualny / Pula Wzorców AI)", 11, anchor="w", weight="bold").grid(
            row=0, column=0, columnspan=3, sticky="w", padx=8, pady=(4, 2)
        )
        self.photo_ref_menu = ctk.CTkOptionMenu(
            pr_cell, values=["Domyślny serii", "Samodzielne (brak)"], command=self._on_photo_ref_menu,
            width=166, height=24, fg_color=T.BTN, button_color=T.BTN, button_hover_color=T.BTN_HOVER,
            text_color=T.TEXT, font=T.font(10), dropdown_font=T.font(11), corner_radius=6
        )
        self.photo_ref_menu.set("Domyślny serii")
        self.photo_ref_menu.grid(row=1, column=0, sticky="ew", padx=(8, 4), pady=(0, 5))
        b_pref = T.button(pr_cell, "▸ Własny…", self.choose_photo_reference, accent=True, width=78); b_pref.configure(height=24, font=T.font(10, "bold"))
        b_pref.grid(row=1, column=1, sticky="e", padx=(0, 4), pady=(0, 5))
        b_auto_pool = T.button(pr_cell, "★ Auto z puli", lambda: self.library.auto_match_pool_references(), width=88)
        b_auto_pool.configure(height=24, font=T.font(10))
        b_auto_pool.grid(row=1, column=2, sticky="e", padx=(0, 8), pady=(0, 5))

        self.ref_info_lbl = T.label(
            f2,
            "Edytor działa samodzielnie bez wzorca. Możesz przypisać osobny wzorzec każdemu zdjęciu lub dopasować wzorce z puli (AI).",
            10, T.MUTED, anchor="w", justify="left", wraplength=352
        )
        self.ref_info_lbl.grid(row=8, column=0, sticky="w", padx=6, pady=(6, 2))
        f0.lift()

        # ---- Collapsible Local Tool Drawer (Kadr (C) | Usuń (Q) | Pędzel) ----
        self.tool_drawer = ctk.CTkFrame(
            sl_area, fg_color=T.PANEL, width=TOOL_W, height=TOOL_H, corner_radius=8,
            border_width=1, border_color=T.ACCENT
        )
        self.tool_drawer.pack_propagate(False)
        cr = ctk.CTkFrame(self.tool_drawer, fg_color=T.PANEL)
        hl = ctk.CTkFrame(self.tool_drawer, fg_color=T.PANEL)
        br = ctk.CTkFrame(self.tool_drawer, fg_color=T.PANEL)
        for tf in (cr, hl, br):
            tf.place(x=6, y=4, relwidth=0.965, relheight=0.94)
        self.tool_frames = {TOOLS[0]: cr, TOOLS[1]: hl, TOOLS[2]: br}

        # Drawer 0: Kadr i Prostowanie (C / R)
        cr.grid_columnconfigure(1, weight=1)
        T.label(cr, "Proporcje", 11, T.MUTED).grid(row=0, column=0, sticky="w")
        self.aspect_menu = ctk.CTkOptionMenu(
            cr, values=list(G.ASPECTS.values()), command=self._aspect, width=110, height=24,
            fg_color=T.BTN, button_color=T.BTN, button_hover_color=T.BTN_HOVER, text_color=T.TEXT,
            font=T.font(11), dropdown_font=T.font(11), corner_radius=6
        )
        self.aspect_menu.grid(row=0, column=1, sticky="w", padx=(6, 0))
        cr_top_r = ctk.CTkFrame(cr, fg_color=T.PANEL); cr_top_r.grid(row=0, column=2, sticky="e")
        b = T.button(cr_top_r, "Pion/Poziom", self._flip_aspect, width=78); b.configure(height=23, font=T.font(10)); b.pack(side="left", padx=(0, 2))
        b_cx = T.button(cr_top_r, "Gotowe ✕", self.close_tool_drawer, accent=True, width=64); b_cx.configure(height=23, font=T.font(10, "bold")); b_cx.pack(side="left")
        T.label(cr, "Prostowanie", 11, T.MUTED).grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.angle = T.lr_slider(cr, -45, 45, number_of_steps=900, width=140, command=self._angle)
        self.angle.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=(6, 0))
        box = ctk.CTkFrame(cr, fg_color=T.PANEL, width=54, height=18); box.grid(row=1, column=2, sticky="e", pady=(6, 0)); box.pack_propagate(False)
        self.angle_val = T.label(box, "", 11, T.MUTED, anchor="e"); self.angle_val.pack(fill="both", expand=True)
        cb = ctk.CTkFrame(cr, fg_color=T.PANEL); cb.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self.hline_btn = T.button(cb, "Linia horyzontu", self.toggle_hline, width=116); self.hline_btn.configure(height=24, font=T.font(10)); self.hline_btn.pack(side="left")
        b_rc = T.button(cb, "Resetuj kadr", self.reset_crop, width=96); b_rc.configure(height=24, font=T.font(10)); b_rc.pack(side="left", padx=6)
        T.label(cr, "Przeciągnij na wyniku: kadr · Enter/Esc: zamknij narzędzie", 10, T.MUTED,
                justify="left").grid(row=3, column=0, columnspan=3, sticky="w", pady=(4, 0))

        # Drawer 1: Pędzel Usuń / Popraw (Q) — Frequency-Separated Pore-Preserving Healing
        hl.grid_columnconfigure(1, weight=1)
        hl_top = ctk.CTkFrame(hl, fg_color=T.PANEL)
        hl_top.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 6))
        T.label(hl_top, "✦ Usuń / Popraw (Q)", 11, "#72CF88", weight="bold").pack(side="left")
        b_hx = T.button(hl_top, "Gotowe ✕", self.close_tool_drawer, accent=True, width=66)
        b_hx.configure(height=23, font=T.font(10, "bold")); b_hx.pack(side="right")
        b_hc = T.button(hl_top, "Wyczyść", self.clear_heals, width=56)
        b_hc.configure(height=23, font=T.font(10)); b_hc.pack(side="right", padx=3)
        b_hu = T.button(hl_top, "Cofnij", self.undo_heal, width=50)
        b_hu.configure(height=23, font=T.font(10)); b_hu.pack(side="right")

        T.label(hl, "Rozmiar plamki", 11, T.MUTED).grid(row=1, column=0, sticky="w", pady=4)
        self.heal_size = T.lr_slider(hl, 0, 1, width=170, command=lambda _: self._heal_label())
        self.heal_size.set(0.32)
        self.heal_size.grid(row=1, column=1, sticky="ew", padx=(6, 4), pady=4)
        hbox = ctk.CTkFrame(hl, fg_color=T.PANEL, width=54, height=18); hbox.grid(row=1, column=2, sticky="e"); hbox.pack_propagate(False)
        self.heal_size_lbl = T.label(hbox, "", 11, T.MUTED, anchor="e"); self.heal_size_lbl.pack(fill="both", expand=True)
        self._heal_label()
        self.heal_cnt_lbl = T.label(hl, "Punkty retuszu: 0", 11, T.TEXT, anchor="w")
        self.heal_cnt_lbl.grid(row=2, column=0, columnspan=3, sticky="w", pady=(2, 2))
        T.label(
            hl,
            "Kliknij lub zamaluj wyprysk/pyłek na zdjęciu — algorytm usuwa plamę i przenosi naturalną teksturę porów z otoczenia.",
            10, T.MUTED, justify="left", wraplength=344
        ).grid(row=3, column=0, columnspan=3, sticky="w")

        # Drawer 2: Pędzel lokalny Dodge & Burn (K)
        br.grid_columnconfigure(1, weight=1)
        self.brush_mode = T.segmented(br, BRUSH_MODES, command=lambda v: self.view_res.delete("cursor"))
        self.brush_mode.configure(height=24, font=T.font(11)); self.brush_mode.set(BRUSH_MODES[0])
        self.brush_mode.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 3))
        bb = ctk.CTkFrame(br, fg_color=T.PANEL); bb.grid(row=0, column=2, sticky="e", padx=(4, 0), pady=(0, 3))
        b_bu = T.button(bb, "Cofnij", self.undo_stroke, width=46); b_bu.configure(height=23, font=T.font(10)); b_bu.pack(side="left")
        b_bc = T.button(bb, "Wyczyść", self.clear_strokes, width=52); b_bc.configure(height=23, font=T.font(10)); b_bc.pack(side="left", padx=2)
        b_bx = T.button(bb, "✕", self.close_tool_drawer, width=24); b_bx.configure(height=23, font=T.font(10)); b_bx.pack(side="left")
        self.brush_vals = {}
        def brow(r, text, lo, hi, val, key):
            T.label(br, text, 11, T.MUTED).grid(row=r, column=0, sticky="w")
            box_ = ctk.CTkFrame(br, fg_color=T.PANEL, width=50, height=18); box_.grid(row=r, column=2, sticky="e", padx=(4, 0)); box_.pack_propagate(False)
            lab = T.label(box_, "", 11, T.MUTED, anchor="e"); lab.pack(fill="both", expand=True); self.brush_vals[key] = lab
            sl_ = T.lr_slider(br, lo, hi, width=170, command=lambda v: self._brush_label(key)); sl_.set(val)
            sl_.grid(row=r, column=1, sticky="ew", padx=(6, 0), pady=1)
            return sl_
        self.brush_size = brow(1, "Rozmiar", 0, 1, 0.45, "size")
        self.brush_soft = brow(2, "Miękkość", 0, 100, 50, "soft")
        self.brush_str = brow(3, "Przepływ", 5, 100, 40, "flow")
        for k in self.brush_vals: self._brush_label(k)
        T.label(br, "[ ] lub Alt+kółko: rozmiar · kliknij ✕ lub Esc, aby zwinąć", 10, T.MUTED).grid(row=4, column=0, columnspan=3, sticky="w", pady=(2, 0))

        # ---- Right Dock Section 4: Subtle Context Tip Box ('Podpowiedź UX') ----
        tip_box = ctk.CTkFrame(dock, fg_color=T.PANEL, corner_radius=7, height=40)
        tip_box.pack(fill="x", padx=8, pady=(2, 3))
        tip_box.pack_propagate(False)
        self.pro_tip_lbl = T.label(
            tip_box,
            "▸ Kółko nad suwakiem: ±2 · Przeciągnij nazwę: mikro-korekta · Klik w liczbę: wpisz wartość · Dwuklik: zeruj.",
            10, T.MUTED, anchor="w", justify="left", wraplength=354
        )
        self.pro_tip_lbl.pack(fill="both", expand=True, padx=7, pady=2)

        # ---- Right Dock Section 5: Mockup Bottom History & Sync Footer (Kopiuj | Kopiuj… | Wklej | →Seria | Zeruj) ----
        foot = ctk.CTkFrame(dock, fg_color=T.CARD)
        foot.pack(fill="x", padx=8, pady=(0, 4))
        s_row = ctk.CTkFrame(foot, fg_color=T.CARD); s_row.pack(fill="x", pady=(0, 2))
        T.label(s_row, "Zdjęcie:", 10, T.MUTED).pack(side="left")
        self.override_lbl = T.label(s_row, "ustawienia serii", 10, T.MUTED); self.override_lbl.pack(side="left", padx=(3, 6))
        b_rst_ser = T.button(s_row, "Przywróć z serii", self.reset_to_series, width=90)
        b_rst_ser.configure(height=20, font=T.font(10)); b_rst_ser.pack(side="right")

        act = ctk.CTkFrame(foot, fg_color=T.CARD); act.pack(fill="x")
        def sbtn(text, cmd, w):
            b_ = T.button(act, text, cmd, width=w)
            b_.configure(height=25, font=T.font(10), corner_radius=6)
            b_.pack(side="left", padx=(0, 3))
            return b_
        self.undo_btn = sbtn("↶", self.undo, 28)
        self.redo_btn = sbtn("↷", self.redo, 28)
        sbtn("Kopiuj", self.copy_params, 50)
        sbtn("Kopiuj…", self.open_selective_copy, 56)
        self.paste_btn = sbtn("Wklej", self.paste_params, 48)
        sbtn("→Seria", self.make_series_default, 54)
        b_z = T.button(act, "Zeruj", self.reset_adjustments, width=50)
        b_z.configure(height=25, font=T.font(10), corner_radius=6); b_z.pack(side="right")

        self.status = T.label(self, "", 11, T.MUTED); self.status.pack(fill="x", pady=2)
        self._load_params_into_ui(self.params); self._update_status()

    # ================= effortless UX helpers =================
    def _show_open_menu(self):
        m = tk.Menu(self, tearoff=0, bg=T.CARD, fg=T.TEXT, activebackground=T.ACCENT, activeforeground=T.ON_ACCENT,
                    font=(T.FAMILY, 10), bd=1, relief="flat")
        m.add_command(label="Otwórz zdjęcia…", command=self.open_photo)
        m.add_command(label="Otwórz cały folder…", command=self.open_folder)
        m.add_command(label="Dodaj zdjęcia do serii…", command=self.add_photos)
        m.add_separator()
        m.add_command(label="Eksportuj całą serię (szybko)", command=self.export_series)
        m.add_command(label="Eksportuj jako… (Ctrl+Shift+E)", command=self.open_export_as)
        m.add_separator()
        m.add_command(label="Ustawienia i akceleracja GPU / OpenCL…", command=self.open_settings)
        m.add_command(label="Skróty klawiszowe i ergonomia (?)", command=self.open_shortcuts)
        x = self.open_btn.winfo_rootx()
        y = self.open_btn.winfo_rooty() + self.open_btn.winfo_height() + 2
        m.tk_popup(x, y)

    def open_shortcuts(self):
        ShortcutsDialog(self)

    def open_selective_copy(self):
        def _do(keys: list[str]):
            d = self.params.to_dict()
            self._clipboard = {k: d[k] for k in keys if k in d}
            self._update_history_btns()
            self._set_status(f"Skopiowano wybrane parametry ({len(self._clipboard)}) — Ctrl+Shift+V wkleja.")
        SelectiveCopyDialog(self, _do)

    def toggle_filmstrip(self):
        self.filmstrip_open = not self.filmstrip_open
        if self.filmstrip_open:
            self.side_panel.pack(side="left", fill="y", padx=(12, 0), before=self.body_panel)
            self.film_btn.configure(border_color=T.ACCENT, text_color=T.ACCENT)
        else:
            self.side_panel.pack_forget()
            self.film_btn.configure(border_color=T.BORDER, text_color=T.TEXT)

    def toggle_side_by_side(self):
        cur = self.canvas_mode.get()
        nxt = "Wynik" if cur == "Przed | Po" else "Przed | Po"
        self.canvas_mode.set(nxt)
        self._set_canvas_mode(nxt)

    def _bind_slider_ux(self, key, row_f, lbox, lbl, box, val, s, lo, hi):
        for w_ in (row_f, lbox, lbl, s):
            w_.bind("<Double-Button-1>", lambda e, k=key: self._reset_slider(k))
            w_.bind("<Enter>", lambda e, k=key: self._set_pro_tip(k))
            w_.bind("<MouseWheel>", lambda e, k=key, lo=lo, hi=hi: self._wheel_slider(k, 1 if e.delta > 0 else -1, e.state, lo, hi))
            w_.bind("<Button-4>", lambda e, k=key, lo=lo, hi=hi: self._wheel_slider(k, 1, e.state, lo, hi))
            w_.bind("<Button-5>", lambda e, k=key, lo=lo, hi=hi: self._wheel_slider(k, -1, e.state, lo, hi))

        # Horizontal drag-scrubbing on slider label for ultra-precise micro-adjustments
        lbox.configure(cursor="sb_h_double_arrow")
        lbl.configure(cursor="sb_h_double_arrow")
        for lw in (lbox, lbl):
            lw.bind("<ButtonPress-1>", lambda e, k=key: self._scrub_start(k, e))
            lw.bind("<B1-Motion>", lambda e, k=key, lo=lo, hi=hi: self._scrub_move(k, e, lo, hi))
            lw.bind("<ButtonRelease-1>", lambda e: self._commit_history())

        # Click numeric readout for inline entry; double-click resets
        val.configure(cursor="xterm")
        val.bind("<Enter>", lambda e, k=key: self._set_pro_tip(k))
        val.bind("<Button-1>", lambda e, k=key, lo=lo, hi=hi: self._open_inline_entry(k, lo, hi))
        val.bind("<Double-Button-1>", lambda e, k=key: (self._close_inline_entry(), self._reset_slider(k)))

    def _wheel_slider(self, key: str, direction: int, state: int, lo: float, hi: float):
        step = 10.0 if (state & 0x1) else 2.0
        s = self.sliders[key][0]
        nv = float(np.clip(float(s.get()) + direction * step, lo, hi))
        s.set(nv); self._slider(key, nv)
        return "break"

    def _scrub_start(self, key: str, e):
        self._scrub_anchor = (key, e.x_root, float(self.sliders[key][0].get()))

    def _scrub_move(self, key: str, e, lo: float, hi: float):
        anc = getattr(self, "_scrub_anchor", None)
        if not anc or anc[0] != key:
            return
        dx = e.x_root - anc[1]
        nv = float(np.clip(anc[2] + dx * 0.30, lo, hi))
        s = self.sliders[key][0]
        s.set(nv); self._slider(key, nv)

    def _open_inline_entry(self, key: str, lo: float, hi: float):
        self._close_inline_entry()
        box = self.slider_boxes.get(key)
        if box is None:
            return
        s, _, kind = self.sliders[key]
        cur_v = float(s.get())
        init_txt = f"{cur_v / 100:.2f}" if kind == "ev" else f"{int(round(cur_v))}"
        ent = ctk.CTkEntry(box, fg_color=T.PANEL, border_color=T.ACCENT, border_width=1, text_color=T.TEXT, font=T.font(10), justify="right")
        ent.place(x=0, y=0, relwidth=1, relheight=1)
        ent.insert(0, init_txt)
        ent.select_range(0, "end")
        ent.focus_set()
        self._inline_entry = ent

        def _apply(_e=None):
            if self._inline_entry is not ent:
                return
            raw = ent.get().strip().replace(",", ".").replace("EV", "").replace("%", "").strip()
            self._close_inline_entry()
            try:
                num = float(raw)
                if kind == "ev" and abs(num) <= 4.0:
                    num *= 100.0
                nv = float(np.clip(num, lo, hi))
                self._commit_history()
                s.set(nv); self._slider(key, nv); self._commit_history()
            except ValueError:
                pass

        ent.bind("<Return>", _apply)
        ent.bind("<Escape>", lambda _e: self._close_inline_entry())
        ent.bind("<FocusOut>", _apply)

    def _close_inline_entry(self):
        if self._inline_entry is not None:
            try:
                self._inline_entry.destroy()
            except Exception:
                pass
            self._inline_entry = None

    def _hist_zone_drag(self, key: str, dx: int):
        if key not in self.sliders:
            return
        s, _, _ = self.sliders[key]
        lo = -200.0 if key == "exposure" else -100.0
        hi = 200.0 if key == "exposure" else 100.0
        scale = 0.85 if key == "exposure" else 0.70
        nv = float(np.clip(float(s.get()) + dx * scale, lo, hi))
        s.set(nv); self._slider(key, nv)

    def _active_tab_keys(self) -> list[str]:
        cur = self.slider_tabs.get()
        keys = [row[0] for row in SLIDER_GROUPS.get(cur, [])]
        if cur == "Kolor HSL":
            keys.extend(["hsl_hue", "hsl_sat", "hsl_lum"])
        return keys

    def _reset_active_tab(self):
        self._bypass_tab_saved = None
        self.tab_eye_btn.configure(border_color=T.BORDER, text_color=T.TEXT)
        self._commit_history()
        cur = self.slider_tabs.get()
        for k in self._active_tab_keys():
            setattr(self.params, k, getattr(Params(), k))
        if cur == "Portret 3D":
            self.params.light_angle = None
        self._load_params_into_ui(self.params)
        self._changed(discrete=True)
        self._set_status(f"Wyzerowano zakładkę: {cur}")

    def _toggle_tab_bypass(self):
        cur = self.slider_tabs.get()
        keys = self._active_tab_keys()
        if self._bypass_tab_saved is None:
            d = self.params.to_dict()
            self._bypass_tab_saved = {k: d[k] for k in keys if k in d}
            defs = Params().to_dict()
            for k in self._bypass_tab_saved:
                setattr(self.params, k, defs[k])
            self.tab_eye_btn.configure(border_color=T.ACCENT, text_color=T.ACCENT)
            self._load_params_into_ui(self.params)
            self.request_render()
            self._set_status(f"Podgląd bez zakładki „{cur}” — kliknij 👁 ponownie, aby przywrócić.")
        else:
            for k, v in self._bypass_tab_saved.items():
                setattr(self.params, k, v)
            self._bypass_tab_saved = None
            self.tab_eye_btn.configure(border_color=T.BORDER, text_color=T.TEXT)
            self._load_params_into_ui(self.params)
            self.request_render()
            self._set_status(f"Przywrócono zakładkę „{cur}”.")

    def _set_pro_tip(self, key: str):
        if hasattr(self, "pro_tip_lbl") and key in PRO_TIPS:
            hint_str = ""
            if key in SLIDER_HINTS:
                l_t, r_t, _, _ = SLIDER_HINTS[key]
                hint_str = f" [{l_t}  ↔  {r_t}]"
            self.pro_tip_lbl.configure(text=f"▸ {PRO_TIPS[key]}{hint_str}")

    # ================= 8-channel OKLCh HSL mixer =================
    def _cur_hsl_field(self) -> str:
        return _HSL_MODES.get(self.hsl_mode_seg.get(), "hsl_sat")

    def _sync_hsl8_ui(self):
        if not hasattr(self, "hsl8_sliders"):
            return
        arr = list(getattr(self.params, self._cur_hsl_field(), [0.0] * 8))
        while len(arr) < 8:
            arr.append(0.0)
        for i, (c_sl, c_val) in enumerate(self.hsl8_sliders):
            v = float(arr[i]) * 100.0
            c_sl.set(v)
            iv = int(round(v))
            c_val.configure(text=f"{iv:+d}" if iv else "0", text_color=T.TEXT if iv else T.MUTED)

    def _on_hsl8(self, idx: int, v: float):
        field = self._cur_hsl_field()
        arr = list(getattr(self.params, field, [0.0] * 8))
        while len(arr) < 8:
            arr.append(0.0)
        arr[idx] = round(float(v) / 100.0, 4)
        setattr(self.params, field, arr)
        iv = int(round(float(v)))
        self.hsl8_sliders[idx][1].configure(text=f"{iv:+d}" if iv else "0", text_color=T.TEXT if iv else T.MUTED)
        self._changed()

    def _wheel_hsl8(self, idx: int, direction: int, state: int):
        step = 10.0 if (state & 0x1) else 2.0
        c_sl = self.hsl8_sliders[idx][0]
        nv = float(np.clip(float(c_sl.get()) + direction * step, -100.0, 100.0))
        c_sl.set(nv); self._on_hsl8(idx, nv)
        return "break"

    def _reset_hsl8_chan(self, idx: int):
        field = self._cur_hsl_field()
        arr = list(getattr(self.params, field, [0.0] * 8))
        if idx < len(arr) and abs(arr[idx]) <= 1e-4:
            return
        self._commit_history()
        arr[idx] = 0.0
        setattr(self.params, field, arr)
        self._sync_hsl8_ui()
        self._changed(discrete=True)

    def reset_hsl8(self):
        self._commit_history()
        self.params.hsl_hue = [0.0] * 8
        self.params.hsl_sat = [0.0] * 8
        self.params.hsl_lum = [0.0] * 8
        self._sync_hsl8_ui()
        self._changed(discrete=True)
        self._set_status("Wyzerowano 8-kanałowy mikser OKLCh HSL.")

    # ================= threading =================
    def _worker(self):
        while True:
            fn, done, kind, gen = self._jobs.get()
            if kind and self._gen.get(kind) != gen: continue
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
            DEV.ocl_configure(self.settings.device)
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
                DEV.MODE = self.settings.device
                DEV.ocl_configure(self.settings.device)
                DEV.reset(); MF.reset_sessions(); RL.reset_session()
            self._style_mask_btn()
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
        m = self.canvas_mode.get() if hasattr(self, "canvas_mode") else "Wynik"
        for w_ in (self.src_hdr, self.res_hdr, self.ref_hdr, self.view_src, self.view_res, self.view_ref):
            w_.grid_remove()
        for c in range(3):
            self.imgs.grid_columnconfigure(c, weight=0, uniform="")
        if m in ("Wynik", "Podział"):
            self.imgs.grid_columnconfigure(1, weight=1, uniform="")
            self.res_hdr.configure(text="Podział: Przed | Po (przeciągnij kurtynę)" if m == "Podział" else ("Wynik (Oryginał)" if self.show_before else "Wynik"))
            self.res_hdr.grid(row=0, column=1, sticky="w", padx=6, pady=(2, 2))
            self.view_res.grid(row=1, column=1, sticky="nsew", padx=4, pady=(0, 4))
        elif m == "Przed | Po":
            for c in (0, 1):
                self.imgs.grid_columnconfigure(c, weight=1, uniform="v2")
            self.res_hdr.configure(text="Wynik (Oryginał)" if self.show_before else "Wynik")
            self.src_hdr.grid(row=0, column=0, sticky="w", padx=6, pady=(2, 2))
            self.res_hdr.grid(row=0, column=1, sticky="w", padx=6, pady=(2, 2))
            self.view_src.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))
            self.view_res.grid(row=1, column=1, sticky="nsew", padx=4, pady=(0, 4))
        else:  # "Z wzorcem"
            for c in range(3):
                self.imgs.grid_columnconfigure(c, weight=1, uniform="v3")
            self.res_hdr.configure(text="Wynik (Oryginał)" if self.show_before else "Wynik")
            self.src_hdr.grid(row=0, column=0, sticky="w", padx=6, pady=(2, 2))
            self.res_hdr.grid(row=0, column=1, sticky="w", padx=6, pady=(2, 2))
            self.ref_hdr.grid(row=0, column=2, sticky="w", padx=6, pady=(2, 2))
            self.view_src.grid(row=1, column=0, sticky="nsew", padx=4, pady=(0, 4))
            self.view_res.grid(row=1, column=1, sticky="nsew", padx=4, pady=(0, 4))
            self.view_ref.grid(row=1, column=2, sticky="nsew", padx=4, pady=(0, 4))

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
            self.canvas_mode.set("Wynik"); self._apply_canvas_layout()
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
        fl = 0
        if self.series:
            fl = getattr(self.series.params_for(p), "flag", 0)
        fl_p = "⚑ " if fl > 0 else ("✕ " if fl < 0 else "")
        return fl_p + ("● " if self.series and self.series.has_override(p) else "") + name

    def _refresh_strip_marks(self):
        for i, b in enumerate(self.thumb_btns):
            b.configure(text=self._thumb_text(self.photos[i]), border_color=T.ACCENT if i == self.idx else T.PANEL)

    # ================= selection / processing =================
    def step(self, d):
        if self.photos and not isinstance(self.focus_get(), ctk.CTkEntry):
            self.select(max(0, min(len(self.photos)-1, self.idx+d)))

    def select(self, i):
        self._close_inline_entry()
        self._bypass_tab_saved = None
        if hasattr(self, "tab_eye_btn"):
            self.tab_eye_btn.configure(border_color=T.BORDER, text_color=T.TEXT)
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
            m = self.canvas_mode.get() if hasattr(self, "canvas_mode") else "Wynik"
            self.res_hdr.configure(
                text="Wynik (Oryginał)" if self.show_before else ("Podział: Przed | Po (przeciągnij kurtynę)" if m == "Podział" else "Wynik"),
                text_color=T.ACCENT if self.show_before else T.MUTED
            )

    def toggle_before(self, on=None):
        if self.result is None: return
        self.show_before = (not self.show_before) if on is None else bool(on)
        self._style_before(); self.redraw_result()

    def _style_mask_btn(self):
        if hasattr(self, "mask_btn"):
            on = bool(self.settings.show_mask)
            self.mask_btn.configure(border_color=T.ACCENT if on else T.BORDER, text_color=T.ACCENT if on else T.TEXT)

    def toggle_mask(self, on=None):
        self.settings.show_mask = (not self.settings.show_mask) if on is None else bool(on)
        self._style_mask_btn()
        self.redraw_result()
        n_faces = len(self.fit.faces_n) if (self.fit is not None and self.fit.faces_n) else 0
        self._set_status(
            f"Maska AI (M): włączona — wykryte twarze: {n_faces} (bursztyn = maska skóry, turkus = granica 3D, punkty = oczy/nos/usta)."
            if self.settings.show_mask else "Maska AI (M): wyłączona."
        )

    def _decorate(self, img):
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
            self.view_res.delete("cursor")
            self._set_status("Pipeta (W): kliknij na wyniku punkt, który ma być neutralnie szary lub biały (Esc: anuluj).")

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

    def auto_wb_ai(self):
        if self.src_prev is None:
            self._set_status("Najpierw otwórz zdjęcie."); return
        faces_n = self.fit.faces_n if self.fit is not None else ()
        t_c, ti_c = WB.auto_wb_scene(self.src_prev, faces_n=faces_n)
        self._set_wb(t_c, ti_c)
        self._set_status(
            f"Auto WB (Shades-of-Gray p=6 z ochroną karnacji): Temperatura {int(round(t_c*100)):+d} · Odcień {int(round(ti_c*100)):+d}"
        )

    def _set_wb(self, t, ti):
        self._commit_history()
        self.params.temp, self.params.tint = t, ti
        for k in ("temp", "tint"):
            s_, val, kind = self.sliders[k]; pv = getattr(self.params, k); v = pv*100; s_.set(v)
            val.configure(text=self._fmt(v, kind), text_color=self._val_color(k, pv))
            self._style_slider_active(k, pv)
        self._changed(discrete=True)

    def reset_wb(self):
        self._set_wb(0.0, 0.0)

    def sync_wb(self):
        if not self.series or not self.photos: return
        if self._save_after: self.after_cancel(self._save_after); self._save_after = None
        if self.series.has_override(self.photos[self.idx]):
            self.series.set_override(self.photos[self.idx], Params.from_dict(self.params.to_dict()))
        self.series.sync_fields(self.photos, {"temp": self.params.temp, "tint": self.params.tint})
        self._refresh_strip_marks(); self._update_override_label()
        self._set_status(f"Balans bieli skopiowany do całej serii ({len(self.photos)} zdjęć).")

    # ================= local tools: crop (C/R), heal (Q), brush (K) =================
    def _frame_wh(self):
        if self.src_size: return self.src_size
        return (self.result.shape[1], self.result.shape[0]) if self.result is not None else (3, 2)

    def _crop_portrait(self):
        W, H = self._frame_wh(); c = self.params.crop
        return (c[2]-c[0])*W < (c[3]-c[1])*H if c else H > W

    def _toggle_tool_segment(self, name: str):
        if self.tool_open and self.tool_tabs.get() == name:
            self.close_tool_drawer()
        else:
            self.tool_tabs.set(name)
            self._tool_tab(name)

    def _tool_tab(self, name: str):
        if name not in self.tool_frames:
            return
        if name == TOOLS[0]:
            self.tool = "crop"
        elif name == TOOLS[1]:
            self.tool = "heal"
        else:
            self.tool = "brush"
            if self.brush_mode.get() == BRUSH_MODES[0]:
                self.brush_mode.set("Rozjaśnij")
        self.tool_open = True
        self.tool_tabs.set(name)
        if hasattr(self, "tool_drawer"):
            self.tool_drawer.place(x=0, y=0, relwidth=1)
            self.tool_drawer.lift()
        self.tool_frames[name].lift()
        self.hline_on = False; self._hline = None; self._style_hline()
        if self.tool == "crop":
            self.toggle_pick(False); self.view_res.delete("cursor")
        elif self.tool == "heal":
            self.toggle_pick(False); self._update_heal_count()
        self.redraw_result(); self._draw_crop_overlay()

    def close_tool_drawer(self):
        was_heal = (self.tool == "heal")
        self.tool_open = False
        self.tool = "brush"
        self.hline_on = False; self._hline = None; self._style_hline()
        if hasattr(self, "tool_drawer"):
            self.tool_drawer.place_forget()
        if hasattr(self, "tool_tabs"):
            self.tool_tabs.set("")
        if was_heal:
            self.view_res.delete("cursor")
        self.redraw_result(); self._draw_crop_overlay()

    def toggle_crop_tool(self):
        if self.tool_open and self.tool == "crop":
            self.close_tool_drawer()
        else:
            self._tool_tab(TOOLS[0])

    def toggle_heal_tool(self):
        if self.tool_open and self.tool == "heal":
            self.close_tool_drawer()
        else:
            self._tool_tab(TOOLS[1])

    def toggle_brush_tool(self):
        if self.tool_open and self.tool == "brush":
            self.brush_mode.set(BRUSH_MODES[0])
            self.close_tool_drawer()
        else:
            self._tool_tab(TOOLS[2])

    def _crop_done(self):
        if self.tool == "crop" or self.tool_open:
            self.close_tool_drawer()

    def _escape(self):
        if self._inline_entry is not None:
            self._close_inline_entry()
        elif self.picking:
            self.toggle_pick(False); self._update_status()
        elif self.tool == "crop" or self.tool_open:
            self.close_tool_drawer()

    def _key(self, fn):
        if isinstance(self.focus_get(), ctk.CTkEntry): return
        fn()

    def _rating_key(self, _e, stars: int):
        if isinstance(self.focus_get(), ctk.CTkEntry): return
        self._set_current_rating(stars)

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
        self._close_inline_entry()
        if self._bypass_tab_saved is not None:
            self._toggle_tab_bypass()
        if self.tool_open:
            self.close_tool_drawer()
        canon = _TAB_ALIASES.get(name, name)
        if hasattr(self, "slider_tabs") and canon in SLIDER_TABS and self.slider_tabs.get() != canon:
            self.slider_tabs.set(canon)
        self.slider_frames[name].lift()

    def auto_tone_ai(self):
        if self.src_prev is None:
            self._set_status("Najpierw otwórz zdjęcie."); return
        self._commit_history()
        faces_n = self.fit.faces_n if self.fit is not None else ()
        auto = TN.auto_tone_params(self.src_prev, faces_n=faces_n)
        for k, v in auto.items():
            setattr(self.params, k, float(v))
        self.detail_raw = None; self.view_res.detail = None
        self._load_params_into_ui(self.params)
        self._changed(discrete=True)
        ita = TN.skin_ita_deg(self.src_prev, faces_n)
        ita_msg = f" · ITA° {ita:+.0f}°" if ita is not None else ""
        ml_msg = f" · Światło mieszane {int(round(self.params.mixed_light*100))}%" if self.params.mixed_light > 0.05 else ""
        self._set_status(
            f"★ Auto Ton AI v2: Ekspozycja {self.params.exposure:+.2f} EV · Cienie {int(round(self.params.shadows*100)):+d} · "
            f"Światła {int(round(self.params.highlights*100)):+d}{ita_msg}{ml_msg}".replace(".", ",")
        )

    def _val_color(self, key, param_val):
        return T.TEXT if abs(param_val - getattr(_DEFAULTS, key)) > 1e-3 else T.MUTED

    def _style_slider_active(self, key: str, param_val: float):
        lbl = self.slider_labels.get(key)
        if lbl is not None:
            active = abs(param_val - getattr(_DEFAULTS, key)) > 1e-3
            lbl.configure(text_color="#F5D39E" if active else T.TEXT)

    def _load_params_into_ui(self, p: Params):
        for key, (s, val, kind) in self.sliders.items():
            pv = getattr(p, key); v = pv*100; s.set(v)
            val.configure(text=self._fmt(v, kind), text_color=self._val_color(key, pv))
            self._style_slider_active(key, pv)
        self._sync_hsl8_ui()
        is_auto = (p.light_angle is None and abs(p.light_x) < 1e-4 and abs(p.light_y) < 1e-4)
        if is_auto:
            self.light_auto.select()
        else:
            self.light_auto.deselect()
        if hasattr(self, "sharpen_mode_menu"):
            self.sharpen_mode_menu.set(DET.SHARPEN_MODES.get(p.sharpen_mode, DET.SHARPEN_MODES["portrait"]))
        self._update_star_btns(p.rating)
        self._update_flag_btns(getattr(p, "flag", 0))
        self._update_heal_count()
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
            if i == 0:
                sb.configure(text="0", text_color=T.TEXT if rating == 0 else T.MUTED)
            else:
                on = i <= rating
                sb.configure(text="★" if on else "☆", text_color=T.ACCENT if on else T.MUTED)

    def _set_current_rating(self, stars: int):
        self.params.rating = int(np.clip(stars, 0, 5))
        self._update_star_btns(self.params.rating)
        if self.series is not None and 0 <= self.idx < len(self.photos):
            self.series.set_override(self.photos[self.idx], self.params)
            self._refresh_strip_marks(); self._update_override_label()
        self._set_status(f"Ocena zdjęcia: {'★' * self.params.rating if self.params.rating else 'brak (0★)'}")

    def _update_flag_btns(self, flag: int):
        if hasattr(self, "flag_pick_btn"):
            self.flag_pick_btn.configure(
                border_color="#72CF88" if flag > 0 else T.BORDER,
                text_color="#72CF88" if flag > 0 else T.MUTED,
            )
        if hasattr(self, "flag_rej_btn"):
            self.flag_rej_btn.configure(
                border_color="#E06868" if flag < 0 else T.BORDER,
                text_color="#E06868" if flag < 0 else T.MUTED,
            )

    def _set_current_flag(self, flag: int):
        self.params.flag = int(np.clip(flag, -1, 1))
        self._update_flag_btns(self.params.flag)
        if self.series is not None and 0 <= self.idx < len(self.photos):
            self.series.set_override(self.photos[self.idx], self.params)
            self._refresh_strip_marks(); self._update_override_label()
            if self.library.winfo_ismapped():
                self.library.refresh()
        msg = "Wybrane ⚑ (P)" if self.params.flag > 0 else ("Odrzucone ✕ (X)" if self.params.flag < 0 else "Bez flagi (U)")
        self._set_status(f"Flaga selekcji: {msg}")

    def _auto_angle(self):
        r = self.fit.relight if self.fit is not None and self.fit.relight is not None else None
        return r.auto_angle if r is not None and r.auto_angle is not None else 120.0

    @staticmethod
    def _dir_name(a):
        names = ["→", "↗", "↑", "↖", "←", "↙", "↓", "↘"]
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
                self._style_slider_active(k, pv)
        self._show_light(); self._changed()

    def _light_auto(self):
        self._commit_history()
        if self.light_auto.get():
            self.params.light_angle = None
            self.params.light_x = 0.0
            self.params.light_y = 0.0
            for k in ("light_x", "light_y"):
                if k in self.sliders:
                    s_, val_, _ = self.sliders[k]
                    s_.set(0); val_.configure(text="0", text_color=T.MUTED)
                    self._style_slider_active(k, 0.0)
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
        self._style_slider_active(key, pv)
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
        self._style_slider_active(key, dv)
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
        self._set_status("Skopiowano ustawienia zdjęcia (Ctrl+Shift+V wkleja · Kopiuj… wybiera grupy).")

    def paste_params(self):
        if not self._clipboard: return
        self._commit_history()
        merged = {**self.params.to_dict(), **self._clipboard}
        self.params = Params.from_dict(merged)
        self.detail_raw = None; self.view_res.detail = None
        self._load_params_into_ui(self.params); self._changed(discrete=True)
        self._set_status("Wklejono ustawienia do bieżącego zdjęcia.")

    def reset_adjustments(self):
        self._commit_history()
        d = _DEFAULTS.to_dict()
        keep = {k: getattr(self.params, k) for k in ("strokes", "heal_strokes", "angle", "crop", "aspect", "rating", "flag")}
        self.params = Params.from_dict({**d, **keep})
        self.detail_raw = None; self.view_res.detail = None
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
        self.series.params = Params.from_dict({**self.params.to_dict(), "strokes": [], "heal_strokes": [], "rating": 0, "flag": 0})
        self.series.save()
        if self.idx >= 0 and not self.params.strokes and not self.params.heal_strokes and self.params.crop is None and not self.params.angle:
            if self._save_after: self.after_cancel(self._save_after); self._save_after = None
            self.series.clear_override(self.photos[self.idx])
        self._refresh_strip_marks(); self._update_override_label(); self._set_status("Zapisano jako domyślne dla całej serii.")

    # ================= healing (Q) & D&B brush (K) =================
    def _brush_px(self):
        return BRUSH_MIN*(BRUSH_MAX/BRUSH_MIN)**float(self.brush_size.get())

    def _heal_px(self):
        return HEAL_MIN*(HEAL_MAX/HEAL_MIN)**float(self.heal_size.get())

    def _active_radius_px(self):
        return self._heal_px() if self.tool == "heal" else self._brush_px()

    def _heal_label(self):
        if hasattr(self, "heal_size_lbl"):
            self.heal_size_lbl.configure(text=f"{int(round(self._heal_px()))} px")

    def _update_heal_count(self):
        if hasattr(self, "heal_cnt_lbl"):
            n = len(getattr(self.params, "heal_strokes", []))
            self.heal_cnt_lbl.configure(text=f"Punkty retuszu: {n}")

    def _brush_label(self, key):
        lab = self.brush_vals[key]
        if key == "size": lab.configure(text=f"{int(round(self._brush_px()))} px")
        elif key == "soft": lab.configure(text=f"{int(round(self.brush_soft.get()))}%")
        else: lab.configure(text=f"{int(round(self.brush_str.get()))}%")

    def _brush_step(self, d):
        if isinstance(self.focus_get(), ctk.CTkEntry): return
        if self.tool == "heal":
            self.heal_size.set(float(np.clip(self.heal_size.get() + d*0.05, 0, 1)))
            self._heal_label()
        else:
            self.brush_size.set(float(np.clip(self.brush_size.get() + d*0.04, 0, 1)))
            self._brush_label("size")
        self._redraw_cursor()

    def _brush_on(self):
        if self.result is None:
            return False
        if self.tool == "heal":
            return True
        return self.brush_mode.get() != BRUSH_MODES[0]

    def _brush_color(self):
        if self.tool == "heal":
            return "#58D68D"
        return T.ACCENT if self.brush_mode.get() == "Rozjaśnij" else "#6FA8FF"

    def _brush_cursor(self, e):
        self._mouse = (e.x, e.y); v = self.view_res; v.delete("cursor")
        if not self._brush_on() or self._space or self.picking or self.tool == "crop": return
        r = max(2.0, self._active_radius_px()*v.cur_zoom())
        v.create_oval(e.x-r, e.y-r, e.x+r, e.y+r, outline=self._brush_color(), width=1.5, tags="cursor")
        if r > 6: v.create_oval(e.x-1.5, e.y-1.5, e.x+1.5, e.y+1.5, outline=self._brush_color(), tags="cursor")

    def _redraw_cursor(self):
        if getattr(self, "_mouse", None):
            class E: pass
            e = E(); e.x, e.y = self._mouse; self._brush_cursor(e)

    def _brush_press(self, e):
        self._close_inline_entry()
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
        r = self._active_radius_px()*self.view_res.cur_zoom()
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
        W = W*G.straighten_scale(W, H, a)
        self._commit_history()
        if self.tool == "heal":
            self.params.heal_strokes.append({
                "size": round(self._heal_px()/W, 5),
                "points": pts[::max(1, len(pts)//150)],
            })
            self.detail_raw = None; self.view_res.detail = None
            self._update_heal_count()
            self._changed(discrete=True)
            self._set_status(f"Usunięto niedoskonałość z zachowaniem porów (punkty: {len(self.params.heal_strokes)}).")
        else:
            self.params.strokes.append({
                "mode": "dodge" if self.brush_mode.get() == "Rozjaśnij" else "burn",
                "size": round(self._brush_px()/W, 5), "strength": round(self.brush_str.get()/100, 3),
                "softness": round(self.brush_soft.get()/100, 3),
                "points": pts[::max(1, len(pts)//300)],
            })
            self._changed(discrete=True)

    def undo_heal(self):
        if self.params.heal_strokes:
            self._commit_history()
            self.params.heal_strokes.pop()
            self.detail_raw = None; self.view_res.detail = None
            self._update_heal_count()
            self._changed(discrete=True)

    def clear_heals(self):
        if self.params.heal_strokes:
            self._commit_history()
            self.params.heal_strokes = []
            self.detail_raw = None; self.view_res.detail = None
            self._update_heal_count()
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
        if st & 0x4:
            v.zoom_at(1.25 if d > 0 else 0.8, e.x, e.y); self._redraw_cursor(); return "break"
        if v.zoom is not None:
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
        self.view_src.set_view(None, (0.5, 0.5), notify=False)
        self.view_res.set_view(None, (0.5, 0.5), notify=False)
        self._sync_zoom_label(); self._schedule_detail()

    def zoom_100(self):
        if isinstance(self.focus_get(), ctk.CTkEntry): return
        v = self.view_res if self.view_res.winfo_ismapped() else self.view_src
        c = v.center if v.zoom else (0.5, 0.5)
        self.view_src.set_view(1.0, c, notify=False)
        self.view_res.set_view(1.0, c, notify=False)
        self._sync_zoom_label(); self._schedule_detail()

    def _sync_zoom_label(self):
        v = self.view_res if (self.view_res.winfo_ismapped() and self.view_res.rgb is not None) else self.view_src
        self.zoom_lbl.configure(text="" if v.rgb is None else f"{int(round(v.cur_zoom()*100))}%")

    def _schedule_detail(self):
        if self._detail_after: self.after_cancel(self._detail_after); self._detail_after = None
        active_v = self.view_res if self.view_res.winfo_ismapped() else self.view_src
        if self.fit is None or not active_v.needs_detail(): return
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
        if pr is None:
            pr = self.presets[0] if self.presets else None
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

    def _style_module_tabs(self, active: str):
        for name, btn in (("lib", self.mod_lib_btn), ("dev", self.mod_dev_btn), ("cmp", self.mod_cmp_btn)):
            is_on = (name == active)
            btn.configure(
                fg_color="#383842" if is_on else "transparent",
                text_color=T.TEXT if is_on else T.MUTED,
                font=T.font(11, "bold" if is_on else "normal"),
            )

    def show_library(self):
        if self.compare.winfo_ismapped(): self.compare.pack_forget()
        if not self.library.winfo_ismapped():
            self.main.pack_forget()
            self.library.pack(fill="both", expand=True, before=self.status)
        self._style_module_tabs("lib")
        self.library.refresh()

    def hide_library(self):
        if self.library.winfo_ismapped():
            self.library.pack_forget()
            self.main.pack(fill="both", expand=True, before=self.status)
            self._refresh_strip_marks()
        self._style_module_tabs("dev")

    def toggle_library(self):
        if self.library.winfo_ismapped(): self.hide_library()
        else: self.show_library()

    def show_compare(self):
        if self.result is None: self._set_status("Najpierw otwórz zdjęcie."); return
        if self.library.winfo_ismapped(): self.library.pack_forget()
        self.main.pack_forget(); self.compare.pack(fill="both", expand=True, before=self.status)
        self._style_module_tabs("cmp")
        self.compare.set_images(self._disp(self.src_prev), self._disp(self.result))

    def hide_compare(self):
        self.compare.pack_forget(); self.main.pack(fill="both", expand=True, before=self.status)
        self._style_module_tabs("dev")

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
            photos = [ph for ph in self.photos if (self.series.params_for(ph).rating >= 1 or self.series.params_for(ph).flag >= 1)]
            if not photos:
                self._set_status("Brak zdjęć z gwiazdką (★) lub flagą (⚑) — oceń zdjęcia klawiszem P lub 1..5."); return
        elif scope_key == "current":
            photos = [cur_ph]
        else:
            photos = [ph for ph in self.photos if (self.series is None or self.series.params_for(ph).flag >= 0)]
            if not photos:
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
            return batch.export_one(eng, ref, p, params, self._out_dir(p), opts, fit=fit)
        self.submit(work, lambda dst: self._set_status(f"Zapisano: {dst}"))

    def export_series(self):
        if not self.photos or self.engine is None:
            self._set_status("Najpierw otwórz folder ze zdjęciami."); return
        photos = [ph for ph in self.photos if (self.series is None or self.series.params_for(ph).flag >= 0)] or list(self.photos)
        dlg = ProgressDialog(self, f"Eksport serii · {len(photos)} zdjęć")
        out_dir = self._out_dir(photos[0])
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
