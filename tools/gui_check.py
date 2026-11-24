"""Headless GUI check (xvfb-run): layout stability while dragging the light slider, zoom/pan, brush coordinates.
Usage: gui_check.py <folder> <reference> <outdir>"""
import sys, time
from pathlib import Path
from PIL import ImageGrab
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retusz.gui.app import RetuszApp

folder, ref, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); out.mkdir(parents=True, exist_ok=True)
app = RetuszApp(folder, ref); app.settings.asked_download = True; app.geometry("1440x900+0+0")
ok = True

def pump(n=10):
    for _ in range(n): app.update(); time.sleep(0.03)

def grab(name, win=None):
    app.update(); w = win or app
    x, y, ww, hh = w.winfo_rootx(), w.winfo_rooty(), w.winfo_width(), w.winfo_height()
    ImageGrab.grab(bbox=(x, y, x+ww, y+hh)).save(out/name); print("saved", out/name, flush=True)

def wait_result(t=90):
    t0 = time.time()
    while app.result is None and time.time()-t0 < t: app.update(); time.sleep(0.05)
    pump(20)

def geom():
    return {k: (s.winfo_x(), s.winfo_width(), s.master.winfo_width()) for k, (s, _, _) in app.sliders.items()} | \
           {"light": (app.light.winfo_x(), app.light.winfo_width())}

def script():
    global ok
    wait_result(); grab("10_glowne.png")
    g0 = geom(); app.light_auto.deselect(); app._light_auto()
    for a in range(0, 360, 15):
        app.light.set(a); app._light(a); pump(2)
        if geom() != g0: print("LAYOUT CHANGED at", a, geom(), g0); ok = False; break
    for k in ("saturation", "vibrance", "skin_even"):
        s = app.sliders[k][0]
        for v in (-100, 100, 0): s.set(v); app._slider(k, v); pump(2)
        if geom() != g0: print("LAYOUT CHANGED by", k); ok = False
    print("layout stable:", ok, flush=True)
    app.light_auto.select(); app._light_auto(); app.result = None; wait_result()
    # zoom 200% around a point, check to_norm consistency
    v = app.view_res; cx, cy = v.winfo_width()*0.5, v.winfo_height()*0.4
    u0 = v.to_norm(cx, cy)
    class E: pass
    e = E(); e.x, e.y, e.state, e.delta = cx, cy, 0x4, 120
    for _ in range(6): app._wheel(v, e, 1); pump(2)
    u1 = v.to_norm(cx, cy); print("zoom", app.zoom_lbl.cget("text"), "point drift", abs(u0[0]-u1[0]), abs(u0[1]-u1[1]))
    if max(abs(u0[0]-u1[0]), abs(u0[1]-u1[1])) > 0.01: ok = False
    print("src/res synced:", app.view_src.zoom == v.zoom and app.view_src.center == v.center)
    t0 = time.time()
    while v.detail is None and time.time()-t0 < 60: app.update(); time.sleep(0.05)
    pump(10); print("detail rendered:", v.detail is not None); grab("11_zoom.png")
    # brush at zoom: stroke lands at the normalised point under the cursor
    app.brush_mode.set("Rozjaśnij"); app.brush_size.set(0.5); app._brush_label("size")
    e.state = 0; app._brush_press(e); e.x += 30; app._brush_move(e); app._brush_cursor(e); pump(3); grab("12_pedzel_kursor.png")
    app._brush_release(e); st = app.params.strokes[-1]
    print("stroke start", st["points"][0], "expected", [round(u1[0], 4), round(u1[1], 4)], "size px", round(st["size"]*app.src_size[0]))
    if abs(st["points"][0][0]-u1[0]) > 0.005: ok = False
    app._brush_step(1); print("after ]:", app.brush_vals["size"].cget("text"))
    app.zoom_fit(); pump(5); print("fit zoom", app.zoom_lbl.cget("text"))
    from retusz.gui.dialogs import SettingsDialog
    d = SettingsDialog(app, app.settings, lambda: None); pump(20); grab("13_ustawienia_ogolne.png", d)
    d.children; 
    for w in d.winfo_children():
        if w.__class__.__name__ == "CTkTabview": w.set("Eksport")
    pump(10); grab("14_ustawienia_eksport.png", d); d.destroy()
    print("RESULT", "OK" if ok else "FAIL", flush=True)
    app.destroy()

app.after(500, script); app.mainloop()
