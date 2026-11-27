"""Dark studio theme with neutral grey slider tracks and #F5A33B warm accent."""
import sys
import customtkinter as ctk

BG = "#161618"; CANVAS_BG = "#1B1B1E"; PANEL = "#1E1E22"; CARD = "#242428"; BORDER = "#34343A"
TEXT = "#EAEAEF"; MUTED = "#9898A2"; ACCENT = "#F5A33B"; ACCENT_HOVER = "#E08E22"; ON_ACCENT = "#18181A"
BTN = "#2C2C32"; BTN_HOVER = "#383840"; TRACK = "#383840"; TRACK_UNI = "#585864"
THUMB = "#DCDCE2"

# Skupienie / przyciemnione panele (L): ultra-ciemne panele, wyciszony tekst, czarne tło zdjęcia
DIM_BG = "#08080A"; DIM_PANEL = "#0D0D10"; DIM_CARD = "#111115"; DIM_BORDER = "#1B1B20"
DIM_TEXT = "#666672"; DIM_MUTED = "#50505A"; DIM_CANVAS = "#060608"

FAMILY = "Segoe UI" if sys.platform == "win32" else ("SF Pro Text" if sys.platform == "darwin" else "DejaVu Sans")

def font(size=13, weight="normal"):
    return ctk.CTkFont(family=FAMILY, size=size, weight=weight)

def setup():
    ctk.set_appearance_mode("dark")

def button(master, text, command=None, accent=False, width=110, **kw):
    if accent:
        return ctk.CTkButton(master, text=text, command=command, width=width, height=34, corner_radius=7,
                             fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=ON_ACCENT, font=font(13, "bold"), **kw)
    return ctk.CTkButton(master, text=text, command=command, width=width, height=34, corner_radius=7,
                         fg_color=BTN, hover_color=BTN_HOVER, text_color=TEXT, border_width=1, border_color=BORDER,
                         font=font(13), **kw)

def slider(master, from_, to, command=None, **kw):
    return ctk.CTkSlider(master, from_=from_, to=to, command=command, height=15, button_length=0,
                         progress_color=ACCENT, button_color=ACCENT, button_hover_color=ACCENT_HOVER,
                         fg_color=TRACK, **kw)

def lr_slider(master, from_, to, command=None, **kw):
    """Calm studio slider: neutral grey track (no bright orange bar on bipolar 0)
    and crisp silver circular thumb with warm amber hover."""
    bipolar = from_ < 0
    return ctk.CTkSlider(
        master, from_=from_, to=to, command=command, height=13, button_length=0,
        progress_color=TRACK if bipolar else TRACK_UNI,
        fg_color=TRACK,
        button_color=THUMB,
        button_hover_color=ACCENT,
        **kw,
    )

def segmented(master, values, command=None, **kw):
    return ctk.CTkSegmentedButton(master, values=values, command=command, height=30, corner_radius=7,
                                  fg_color=PANEL, selected_color="#3A3A44", selected_hover_color="#464652",
                                  unselected_color=PANEL, unselected_hover_color=BTN_HOVER,
                                  text_color=TEXT, font=font(12), **kw)

def label(master, text="", size=13, color=TEXT, weight="normal", **kw):
    return ctk.CTkLabel(master, text=text, text_color=color, font=font(size, weight), **kw)
