"""Headless GUI check (xvfb-run): layout stability (light/angle sliders, tool tabs, clipping overlay), zoom/pan,
brush coordinates (also on a straightened photo), crop drag with aspect, horizon tool, eyedropper.
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
           {"light": (app.light.winfo_x(), app.light.winfo_width()), "hist": (app.hist.winfo_rootx(), app.hist.winfo_rooty()),
            "tabs": (app.tool_tabs.winfo_rootx(), app.tool_tabs.winfo_width()), "res": (app.view_res.winfo_width(), app.view_res.winfo_height())}

class E: pass
def ev(x, y, state=0):
    e = E(); e.x, e.y, e.state, e.delta = x, y, state, 0; return e

def canvas_xy(v, u, w_):
    x0, y0, dw, dh = v.geom; return x0+u*dw, y0+w_*dh

def script():
    global ok
    wait_result(); grab("10_glowne.png")
    g0 = geom()
    # Test new tone sliders (including EV formatting) and tab switch stability
    for k, vals in (("exposure", (-200, 125, 0)), ("contrast", (-100, 100, 0)), ("whites", (-100, 100, 0)), ("blacks", (-100, 100, 0)),
                    ("saturation", (-100, 100, 0)), ("vibrance", (-100, 100, 0))):
        s = app.sliders[k][0]
        for v in vals: s.set(v); app._slider(k, v); pump(2)
        if geom() != g0: print("LAYOUT CHANGED by", k); ok = False
    # Double-click reset + Undo / Redo + Copy / Paste + Before / After
    app.sliders["exposure"][0].set(100); app._slider("exposure", 100); app._commit_history(); pump(2)
    app._reset_slider("exposure"); pump(2)
    if abs(app.params.exposure) > 1e-4: print("RESET SLIDER FAILED"); ok = False
    app.undo(); pump(2)
    if abs(app.params.exposure - 1.0) > 1e-3: print("UNDO FAILED", app.params.exposure); ok = False
    app.redo(); pump(2)
    if abs(app.params.exposure) > 1e-4: print("REDO FAILED", app.params.exposure); ok = False
    app.toggle_before(True); pump(3); app.toggle_before(False); pump(2)
    # Switch to second ("Portret i Plastyka"), third ("Detal i Ostrość"), and fourth ("▸ Wzorzec (AI)") tabs and check stability
    app.slider_tabs.set("Portret i Plastyka"); app._slider_tab("Portret i Plastyka"); pump(4)
    grab("10b_wzorzec_portret.png")
    if geom() != g0: print("LAYOUT CHANGED by slider tab"); ok = False
    app.slider_tabs.set("Detal i Ostrość"); app._slider_tab("Detal i Ostrość"); pump(4)
    grab("10d_detal_ostrosc.png")
    if geom() != g0: print("LAYOUT CHANGED by detail tab"); ok = False
    app.slider_tabs.set("▸ Wzorzec (AI)"); app._slider_tab("▸ Wzorzec (AI)"); pump(4)
    grab("10c_wzorzec_bonus.png")
    if geom() != g0: print("LAYOUT CHANGED by reference bonus tab"); ok = False
    # Test Library module (Biblioteka G -> Edycja D)
    app.show_library(); pump(8); grab("09_biblioteka.png"); app.hide_library(); pump(6)
    if geom() != g0: print("LAYOUT CHANGED by library module toggle"); ok = False
    # Test Preset application + preset_to_series + undo
    app.apply_preset_by_name("Ciepły złoty"); pump(3)
    if abs(app.params.exposure - 0.15) > 1e-4: print("PRESET APPLY FAILED"); ok = False
    app.preset_to_series(); pump(3)
    if abs(app.series.params.exposure - 0.15) > 1e-4: print("PRESET TO SERIES FAILED"); ok = False
    app.undo(); pump(3)
    if geom() != g0: print("LAYOUT CHANGED by preset"); ok = False
    app.slider_tabs.set("Portret i Plastyka"); app._slider_tab("Portret i Plastyka"); pump(3)
    app.light_auto.deselect(); app._light_auto()
    for a in range(0, 360, 15):
        app.light.set(a); app._light(a); pump(2)
        if geom() != g0: print("LAYOUT CHANGED at", a, geom(), g0); ok = False; break
    for k in ("skin_even", "skin_cast", "skin_hue", "clarity", "deshine", "blemish", "eyes_teeth"):
        s = app.sliders[k][0]
        for v in (0, 100, 0) if k in ("skin_even", "skin_cast", "deshine", "blemish", "eyes_teeth") else (-100, 100, 0): s.set(v); app._slider(k, v); pump(2)
        if geom() != g0: print("LAYOUT CHANGED by", k); ok = False
    app.slider_tabs.set("Detal i Ostrość"); app._slider_tab("Detal i Ostrość"); pump(3)
    for k in ("sharpen_amount", "texture", "denoise", "brightness", "dehaze"):
        s = app.sliders[k][0]
        for v in (0, 80, 0) if k in ("sharpen_amount", "denoise") else (-80, 80, 0): s.set(v); app._slider(k, v); pump(2)
        if geom() != g0: print("LAYOUT CHANGED by", k); ok = False
    app.slider_tabs.set("Światło i Kolor"); app._slider_tab("Światło i Kolor"); pump(3)
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
    app.brush_mode.set("Wył."); app.undo_stroke(); wait_render()
    new_features()
    from retusz.gui.dialogs import SettingsDialog, ExportAsDialog
    d = SettingsDialog(app, app.settings, lambda: None); pump(20); grab("13_ustawienia_ogolne.png", d)
    for w in d.winfo_children():
        if w.__class__.__name__ == "CTkTabview": w.set("Eksport")
    pump(10); grab("14_ustawienia_eksport.png", d); d.destroy()
    ea = ExportAsDialog(app, app.settings, out, lambda *a: None); pump(15); grab("14b_eksportuj_jako.png", ea); ea.destroy()
    print("RESULT", "OK" if ok else "FAIL", flush=True)
    app.destroy()

def wait_render(t=30):
    pump(5); t0 = time.time()
    while app._gen.get("render") and not app._jobs.empty() and time.time()-t0 < t: app.update(); time.sleep(0.05)
    pump(25)

def new_features():
    global ok
    from retusz.engine import geometry as G, wb as WB
    import numpy as np
    g0 = geom(); v = app.view_res
    # histogram + clipping overlay
    hist_ok = app.frame is not None and app.hist.find_all()
    app.toggle_clip(True); pump(5); grab("15_przepalenia.png")
    print("histogram drawn:", bool(hist_ok), "| clip label:", app.clip_lbl.cget("text"))
    app.toggle_clip(False); pump(3)
    if geom() != g0: print("LAYOUT CHANGED by clipping toggle"); ok = False
    # eyedropper: on a bright neutral-ish area of the result frame
    import cv2
    f = app.frame.astype(np.float32); sat = f.max(-1) - f.min(-1); lum = f.mean(-1)
    loc = cv2.blur(lum, (9, 9)); locsd = np.sqrt(np.maximum(cv2.blur(lum*lum, (9, 9)) - loc*loc, 0))
    score = np.where((f.max(-1) < 235) & (lum > 70) & (locsd < 3), sat, 1e9); y, x = np.unravel_index(np.argmin(score), score.shape)
    h, w = app.frame.shape[:2]
    app.toggle_pick(True); pump(3); grab("17_pipeta.png")
    r0 = app.result; cx, cy = canvas_xy(v, (x+0.5)/w, (y+0.5)/h); app._brush_press(ev(cx, cy))
    t0 = time.time()
    while app.result is r0 and time.time()-t0 < 30: app.update(); time.sleep(0.05)
    wait_render(); px, py = app._last_pick
    d2 = WB.solve(app.frame[max(0, py-3):py+4, max(0, px-3):px+4])
    print(f"eyedropper at {px},{py} (local sd {locsd[y, x]:.1f}) -> temp {app.params.temp:+.3f} tint {app.params.tint:+.3f}; residual correction {d2}")
    if d2 is None or max(abs(d2[0]), abs(d2[1])) > 0.02: ok = False
    if geom() != g0: print("LAYOUT CHANGED by eyedropper"); ok = False
    # crop tool
    app.toggle_crop_tool(); pump(5)
    if geom() != g0: print("LAYOUT CHANGED by tool tab", geom(), g0); ok = False
    app._aspect(G.ASPECTS["2:3"]); pump(5)
    W, H = app.src_size; c = app.params.crop; r = (c[2]-c[0])*W/((c[3]-c[1])*H)
    print(f"aspect 2:3 preset -> crop ratio {r:.4f}")
    if abs(r - (2/3 if H > W else 3/2)) > 0.01: ok = False
    # drag a new crop from (0.2,0.2) to (0.7,0.9): must keep 2:3
    a = canvas_xy(v, 0.2, 0.2); b = canvas_xy(v, 0.7, 0.9)
    app.params.crop = None; app._brush_press(ev(*a)); app._brush_move(ev((a[0]+b[0])/2, (a[1]+b[1])/2)); app._brush_move(ev(*b)); pump(2)
    grab("16_kadr_przeciaganie.png"); app._brush_release(ev(*b)); pump(3)
    c = app.params.crop; r = (c[2]-c[0])*W/((c[3]-c[1])*H)
    print("dragged crop", [round(t, 3) for t in c], f"ratio {r:.4f}")
    if not (abs(r-2/3) < 0.01 or abs(r-1.5) < 0.01) or abs(c[0]-0.2) > 0.003: ok = False
    # straighten via slider: layout must not move
    for ang in (-10, -3.3, 0, 4.5, 2.0):
        app.angle.set(ang); app._angle(ang); pump(2)
        if geom() != g0: print("LAYOUT CHANGED by angle", ang); ok = False; break
    print("angle label:", app.angle_val.cget("text"))
    # horizon tool on a line tilted by +3 degrees in the frame -> angle += 3
    a0 = app.params.angle; app.toggle_hline()
    Wf, Hf = app._frame_wh(); L = 0.5*Wf; dy = np.tan(np.radians(3))*L
    p1 = canvas_xy(v, 0.25, 0.5); p2 = canvas_xy(v, 0.75, 0.5 + dy/Hf)
    app._brush_press(ev(*p1)); app._brush_move(ev(*p2)); app._brush_release(ev(*p2)); pump(5)
    print(f"horizon: angle {a0:+.1f} -> {app.params.angle:+.1f} (expected {a0+3:+.1f})")
    if abs(app.params.angle - (a0+3)) > 0.25: ok = False
    wait_render(); grab("18_kadr_prostowanie.png")
    app._crop_done(); pump(5); grab("19_kadr_gotowy.png")
    # brush on a straightened view lands on the right source point
    app.brush_mode.set("Rozjaśnij"); n = len(app.params.strokes)
    q = canvas_xy(v, 0.55, 0.45); app._brush_press(ev(*q)); app._brush_move(ev(q[0]+1, q[1])); app._brush_release(ev(q[0]+1, q[1]))
    st = app.params.strokes[n]["points"][0]; want = G.frame_to_src(v.to_norm(*q), app.params.angle, Wf, Hf)
    print("rotated brush point", [round(t, 4) for t in st], "expected", [round(t, 4) for t in want])
    if max(abs(st[0]-want[0]), abs(st[1]-want[1])) > 0.003: ok = False
    app.undo_stroke(); app.brush_mode.set("Wył.")
    # sidecar keeps the geometry
    pump(20); time.sleep(0.4); pump(10)
    sp = app.series.params_for(app.photos[app.idx])
    print("sidecar crop/angle:", sp.crop is not None, sp.angle == app.params.angle)
    if sp.crop is None or sp.angle != app.params.angle: ok = False
    app.reset_crop(); app.reset_wb(); pump(5)

app.after(500, script); app.mainloop()
