# Budowanie Retusz dla Windows

## A. Lokalnie na Windows 10/11 (x64)
1. Zainstaluj Python 3.11 lub 3.12 x64 (python.org), Rust (`rustup`, stable x86_64-pc-windows-msvc) oraz opcjonalnie Inno Setup 6 (`iscc`).
2. W folderze projektu:
   ```bat
   cargo test --workspace --release
   cargo build --workspace --release
   py -3 -m venv .venv
   .venv\Scripts\pip install -r requirements-dev.txt
   .venv\Scripts\python -m pytest -q
   .venv\Scripts\pyinstaller --noconfirm retusz.spec
   copy target\release\retusz-rs.exe dist\Retusz\Retusz-Studio-GPU.exe
   iscc installer\retusz.iss
   ```
3. Wynik:
   - instalator `dist\Retusz-Setup-x64.exe` (Inno Setup, polski kreator instalacji),
   - folder portable `dist\Retusz\` (`Retusz.exe` + natywny edytor Rust+wgpu `Retusz-Studio-GPU.exe`).
4. `requirements.txt` instaluje na Windows `onnxruntime-directml` (GPU przez DirectML, każda karta z DirectX 12);
   natywny silnik `retusz-gpu` używa `wgpu` (DirectX 12 / Vulkan) z automatycznym przejściem na wielowątkowy `rayon` (CPU).

## B. GitHub Actions (windows-latest)
Plik `.github/workflows/build-windows.yml` uruchamia `cargo test --release` + `cargo build --release` → `pytest` → PyInstaller → smoke test CLI (Python + Rust + transfer AI) → Inno Setup (`Retusz-Setup-x64.exe`) + ZIP (`Retusz-windows-x64.zip`).
Uruchamia się na tagu `v*` albo ręcznie (Run workflow). Repozytorium: https://github.com/Jack4B/retusz

## Model AI do pobrania (jednorazowo)
Aplikacja pobiera `modflows_b6_fp16.onnx` (119 MB, SHA-256 `92b73d44…bdb09`) z
`https://github.com/Jack4B/retusz/releases/download/models-v1/modflows_b6_fp16.onnx`.
Plik jest assetem wydania `models-v1`. Alternatywy:
- zmienna środowiskowa `RETUSZ_MODEL_URL=<inny adres>`, albo
- Ustawienia → **Wskaż plik…** (instaluje lokalny plik po sprawdzeniu SHA-256).
Lekki model B0 i model głębi są wbudowane w aplikację.

## Zawartość paczki
- `retusz/assets/models/modflows_b0_fp32.onnx` 18 MB (MIT)
- `retusz/assets/models/depth_anything_v2_small_q8.onnx` 27 MB (Apache-2.0)
- `retusz/assets/models/face_detection_yunet_2023mar.onnx` 0,2 MB (MIT)
- `retusz/assets/icc/` profile sRGB, AdobeRGB-compat, Display P3 (CC0)
