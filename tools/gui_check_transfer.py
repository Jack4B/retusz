"""Headless check of the 'Przenieś grading z AI' window (xvfb-run): load a pair, all preview modes,
change map, brush, slider layout stability, full-resolution save.
Usage: gui_check_transfer.py <original> <ai_version> <outdir>"""
import sys, time
from pathlib import Path
from PIL import ImageGrab, Image
import customtkinter as ctk
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retusz.gui import theme as T
from retusz.gui.transfer_view import TransferWindow

orig, ai, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
T.setup(); root = ctk.CTk(); root.withdraw()
win = TransferWindow(root, orig, ai); win.geometry("1360x860+0+0"); ok = True

def pump(n=10):
    for _ in range(n): root.update(); time.sleep(0.03)

def idle(t=240):
    t0 = time.time()
    while time.time()-t0 < t:
        root.update(); time.sleep(0.05)
        if win._busy == 0 and win._jobs.empty(): break
    pump(15)

def grab(name):
    root.update(); x, y, w, h = win.winfo_rootx(), win.winfo_rooty(), win.winfo_width(), win.winfo_height()
    ImageGrab.grab(bbox=(x, y, x+w, y+h)).save(out/name); print("saved", out/name, flush=True)

def geom():
    g = {k: (s.winfo_rootx(), s.winfo_rooty(), s.winfo_width()) for k, (s, _) in win.sliders.items()}
    g["info"] = (win.info.winfo_rootx(), win.info.winfo_rooty(), win.info.winfo_width(), win.info.winfo_height())
    g["view"] = (win.view.winfo_width(), win.view.winfo_height()); return g

class E: pass
def ev(u, v):
    x0, y0, dw, dh = win.view.geom; e = E(); e.x, e.y = x0+u*dw, y0+v*dh; return e

pump(20); idle()
if win.result is None: print("NO RESULT", win.status.cget("text")); sys.exit(1)
print("info:", win.info.cget("text").replace("\n", " | "), "· warn:", win.warn.cget("text") or "-")
grab("20_grading_ai_wynik.png")
g0 = geom()
for k in ("kolor", "jasnosc"):
    for v in (0, 37, 100, 70):
        win.sliders[k][0].set(v); win._slider(k, v); pump(3)
        if geom() != g0: print("LAYOUT CHANGED by", k, v); ok = False
idle(); print("layout stable:", ok)
grab("21_grading_ai_jasnosc70.png")
for m, n in (("Podział", "22_grading_ai_podzial.png"), ("AI", "23_grading_ai_nalozone_AI.png"), ("Oryginał", "24_grading_ai_oryginal.png")):
    win.mode.set(m); win._mode_changed(); pump(10); grab(n)
win.mode.set("Wynik"); win._mode_changed(); win.changes_sw.select(); win.redraw(); pump(5); grab("25_grading_ai_mapa_zmian.png")
win.changes_sw.deselect()
win.brush_sw.select(); win._brush_toggle()
for i in range(25): win._drag(ev(0.3 + i*0.012, 0.45)) if i else win._press(ev(0.3, 0.45))
win._release(None); idle(); grab("26_grading_ai_pedzel.png")
m = win.mask; print("mask min", float(m.min()), "mean", float(m.mean()))
if not (m.min() < 0.5 and m.mean() > 0.9): ok = False
if geom() != g0: print("LAYOUT CHANGED after brush"); ok = False
dst = out / "grading_ai_zapis.jpg"; win.save_to(dst); idle(300)
W, H = win.orig.shape[1], win.orig.shape[0]
sz = Image.open(dst).size if dst.exists() else None; print("saved size", sz, "orig", (W, H))
if sz != (W, H): ok = False
print("GUI TRANSFER CHECK:", "OK" if ok else "FAIL"); sys.exit(0 if ok else 1)
