# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for Retusz (author: Jack4B).
#   Windows:  pyinstaller retusz.spec            -> dist/Retusz/Retusz.exe (folder build, fast start)
#             set RETUSZ_ONEFILE=1 first          -> dist/Retusz.exe (single file, slower start)
import os, sys
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

onefile = os.environ.get("RETUSZ_ONEFILE") == "1"
datas = [("retusz/assets", "retusz/assets"), ("retusz/engine/render.py", "retusz/engine")]  # source needed for the numba disk cache
datas += collect_data_files("customtkinter")
binaries = collect_dynamic_libs("onnxruntime")
hidden = ["PIL._tkinter_finder", "numba", "llvmlite", "onnxruntime"]
excludes = ["torch", "matplotlib", "scipy", "pandas", "IPython", "pytest"]
icon = "retusz/assets/retusz.ico"

a = Analysis(["retusz_main.py"], pathex=["."], binaries=binaries, datas=datas, hiddenimports=hidden,
             excludes=excludes, noarchive=False)
pyz = PYZ(a.pure)
if onefile:
    exe = EXE(pyz, a.scripts, a.binaries, a.datas, [], name="Retusz", icon=icon, console=False,
              upx=False, version=None)
else:
    exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Retusz", icon=icon, console=False, upx=False)
    coll = COLLECT(exe, a.binaries, a.datas, name="Retusz", upx=False)
