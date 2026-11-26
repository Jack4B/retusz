# Licencje komponentów i sumy kontrolne modeli (SHA-256)

Kod własny aplikacji **Retusz** (Python + Rust, autor: **Jack4B**) udostępniany jest na licencji **MIT** (zob. plik `LICENSE`).

## 1. Modele AI (ONNX) — licencje kodu i wag oraz sumy SHA-256

| Plik modelu | Rola | Rozmiar | Licencja wag i kodu | Suma kontrolna SHA-256 |
|---|---|---|---|---|
| `modflows_b0_fp32.onnx` (wbudowany) | Dopasowanie kolorów AI (ModFlows B0, Larchenko i in., AAAI 2025, `maria-larchenko/modflows`) | 18 MB | **MIT** | `19cadf28524220bf773278c9bb8d4cb4bd7ef99e41fdbcb0b4d2f6eb6fbbedf7` |
| `modflows_b6_fp16.onnx` (pobierany opcjonalnie z `models-v1`) | Dopasowanie kolorów AI (ModFlows B6, Larchenko i in., AAAI 2025) | 119 MB | **MIT** | `92b73d4416be5fba85ac68623033fcc2d11e97535afe7c3ea25a13e2c34bdb09` |
| `depth_anything_v2_small_q8.onnx` (wbudowany) | Mapa głębi do Plastyki 3D i Separacji tła (Depth Anything V2 **Small** int8, Yang i in. 2024) | 27 MB | **Apache-2.0** *(wyłącznie wariant Small; warianty Base/Large mają licencję CC BY-NC i nie są używane)* | `fcf51f1b230362b28690bb9d1809bf0431f29cad20534e3f589bd7285547f20d` |
| `face_detection_yunet_2023mar.onnx` (wbudowany) | Wykrywanie twarzy i 5 punktów anatomicznych (YuNet, `opencv/opencv_zoo`) | 0,23 MB | **MIT** | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |

## 2. Biblioteki i narzędzia

| Komponent | Użycie | Licencja |
|---|---|---|
| OpenCV (`opencv-python-headless` + OpenCL `cv2.ocl`) | przetwarzanie obrazu, filtr prowadzony, inpainting, akceleracja OpenCL iGPU | Apache-2.0 |
| ONNX Runtime / `onnxruntime-directml` | inferencja modeli ONNX (CPU / DirectML GPU & iGPU) | MIT |
| NumPy, numba, llvmlite | obliczenia numeryczne i wielowątkowe jądro LUT 3D | BSD |
| Pillow | JPG, PNG, WebP, EXIF, ICC | MIT-CMU (HPND) |
| Compact ICC Profiles (Clinton Ingram / `saucecontrol`): sRGB, AdobeRGB-compat, Display P3 | profile barwne przy eksporcie | CC0 1.0 |
| CustomTkinter | interfejs graficzny (Windows / Linux) | MIT |
| rawpy (Maik Riechert) | odczyt plików RAW (NEF, CR2/CR3, ARW, DNG, RAF, ORF, RW2, PEF) | MIT |
| LibRaw 0.22 (w pakiecie `rawpy`, bez modyfikacji) | dekodowanie RAW | podwójna licencja LGPL-2.1 / CDDL-1.0 – używana na warunkach **CDDL-1.0**; kod źródłowy: https://github.com/LibRaw/LibRaw |
| Little CMS, libjpeg (w pakiecie `rawpy`) | biblioteki pomocnicze LibRaw | MIT / IJG |
| tifffile (Christoph Gohlke) | zapis bezstratny TIFF 8/16-bit (LZW + ICC) | BSD-3-Clause |
| Rust (`wgpu`, `eframe`/`egui`, `rayon`, `image`, `serde`, `bytemuck`, `pollster`, `clap`, `rfd`) | natywny silnik GPU/CPU i edytor studyjny `retusz-rs` (`Retusz-Studio-GPU.exe`) | MIT / Apache-2.0 |
| PyInstaller / Inno Setup | budowanie `.exe` oraz instalatora Windows `Retusz-Setup-x64.exe` | GPL-2.0 z wyjątkiem dla bootloadera / Inno Setup License |

## 3. Publikacje naukowe (zaimplementowane od podstaw, bez kopiowania kodu GPL/NC)

- **OKLab / OKLCh i mapowanie gamutu ze stałym odcieniem**: Björn Ottosson (2020/2021, MIT / domena publiczna).
- **Kąt typologii karnacji ITA° (Individual Typology Angle)**: A. Chardon, I. Cretois, C. Hourseau, *Int. J. Cosmetic Science*, 1991.
- **Stałość barw Shades-of-Gray (Auto WB z wykluczeniem skóry)**: G. D. Finlayson, E. Trezzi, *Color and Imaging Conference*, 2004; adaptacja chromatyczna Bradford CAT (Lam 1985 / CIE-ICC).
- **Światło mieszane (Dual-Illuminant White Balance)**: E. Hsu, T. Mertens, S. Paris, S. Avidan, F. Durand, *ACM SIGGRAPH*, 2008.
- **Optyka biofizyczna skóry**: separacja dychromatyczna połysku SUV (Shafer 1985; Mallick i in., CVPR 2005; Yang, Wang, Ahuja, ECCV 2010) oraz gęstość optyczna melaniny/hemoglobiny (Tsumura i in., *ACM SIGGRAPH*, 2003).
- **Szybka Lokalna Piramida Laplace'a (Fast LLF)**: S. Paris, S. W. Hasinoff, J. Kautz (2011); M. Aubry i in., *ACM TOG*, 2014.
- **Filtr prowadzony (Guided Filter)**: K. He, J. Sun, X. Tang, *ECCV*, 2010; *arXiv*, 2015.
- **Bilateral Guided Upsampling (BGU)**: J. Chen, A. Adams, N. Wadhwa, S. W. Hasinoff, *ACM SIGGRAPH Asia*, 2016 (`google/bgu` – Apache-2.0).
- **Kalibracja tablicy ColorChecker 24**: Root-Polynomial Color Correction (G. D. Finlayson, M. Mackiewicz, A. Hurlbert, *IEEE TIP*, 2015) oraz CIEDE2000 (G. Sharma, W. Wu, E. N. Dalal, 2005).
- **Interpolacja tetraedryczna 3D LUT**: K. Sakamoto (1974); J. M. Kasson i in. (1995).
- **Inpainting niskoczęstotliwościowy + transfer porów**: A. Telea, *JGT*, 2004 (OpenCV, Apache-2.0) połączony z separacją częstotliwości i dopasowaniem tekstury otoczenia.
- **Klasyczny transfer barw (bez AI)**: E. Reinhard i in. (2001); F. Pitié, A. Kokaram (MKL / IDT, 2005/2007).

Pliki testowe RAW (Nikon D750, D700) pochodzą z https://raw.pixls.us (CC0 1.0) i nie są dołączone do repozytorium.
Retouch4me, Evoto ani inne programy komercyjne nie są w żaden sposób wykorzystywane (ani kod, ani modele, ani wyekstrahowane dane) — mogą służyć wyłącznie jako czarna skrzynka do zewnętrznego porównania gotowych wyników.
