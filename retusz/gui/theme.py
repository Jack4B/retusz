"""Dark theme with the #F5A33B accent (mockups A/B)."""
import sys
import customtkinter as ctk

BG = "#141414"; PANEL = "#1C1C1C"; CARD = "#242424"; BORDER = "#343434"
TEXT = "#EDEDED"; MUTED = "#9A9A9A"; ACCENT = "#F5A33B"; ACCENT_HOVER = "#E08E22"; ON_ACCENT = "#1A1A1A"
BTN = "#2B2B2B"; BTN_HOVER = "#363636"; TRACK = "#3A3A3A"

FAMILY = "Segoe UI" if sys.platform == "win32" else ("SF Pro Text" if sys.platform == "darwin" else "DejaVu Sans")

def font(size=13, weight="normal"):
    return ctk.CTkFont(family=FAMILY, size=size, weight=weight)

def setup():
    ctk.set_appearance_mode("dark")

def button(master, text, command=None, accent=False, width=110, **kw):
    if accent:
        return ctk.CTkButton(master, text=text, command=command, width=width, height=34, corner_radius=8,
                             fg_color=ACCENT, hover_color=ACCENT_HOVER, text_color=ON_ACCENT, font=font(13, "bold"), **kw)
    return ctk.CTkButton(master, text=text, command=command, width=width, height=34, corner_radius=8,
                         fg_color=BTN, hover_color=BTN_HOVER, text_color=TEXT, border_width=1, border_color=BORDER,
                         font=font(13), **kw)

def slider(master, from_, to, command=None, **kw):
    return ctk.CTkSlider(master, from_=from_, to=to, command=command, height=16, button_length=0,
                         progress_color=ACCENT, button_color=ACCENT, button_hover_color=ACCENT_HOVER,
                         fg_color=TRACK, **kw)

def segmented(master, values, command=None, **kw):
    return ctk.CTkSegmentedButton(master, values=values, command=command, height=32, corner_radius=8,
                                  fg_color=BTN, selected_color="#5A4120", selected_hover_color="#6A4C24",
                                  unselected_color=BTN, unselected_hover_color=BTN_HOVER,
                                  text_color=TEXT, font=font(13), **kw)

def label(master, text="", size=13, color=TEXT, weight="normal", **kw):
    return ctk.CTkLabel(master, text=text, text_color=color, font=font(size, weight), **kw)
