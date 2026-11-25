# Licencje komponentów

| Komponent | Użycie | Licencja |
|---|---|---|
| ModFlows (Larchenko i in., AAAI 2025) — kod i wagi B6/B0, przekonwertowane do ONNX | dopasowanie kolorów AI | MIT |
| Depth Anything V2 **Small** (Yang i in., 2024), ONNX int8 z onnx-community | mapa głębi do Plastyki 3D | Apache-2.0 (tylko wariant Small; Base/Large mają licencję niekomercyjną i nie są używane) |
| YuNet (opencv_zoo) | wykrywanie twarzy | MIT |
| OpenCV | przetwarzanie obrazu | Apache-2.0 |
| ONNX Runtime / onnxruntime-directml | uruchamianie modeli | MIT |
| NumPy, numba, llvmlite | obliczenia | BSD |
| Pillow | JPG, EXIF, ICC | MIT-CMU (HPND) |
| Compact ICC Profiles (Clinton Ingram / saucecontrol): sRGB, AdobeRGB-compat, Display P3 | profile barwne przy eksporcie | CC0 1.0 |
| CustomTkinter | interfejs | MIT |
| rawpy (Maik Riechert) | odczyt RAW NEF | MIT |
| LibRaw 0.22 (w pakiecie rawpy, bez zmian) | dekodowanie RAW | podwójna licencja LGPL-2.1 / CDDL-1.0 – używana na warunkach **CDDL-1.0**; kod źródłowy: https://github.com/LibRaw/LibRaw |
| Little CMS, libjpeg (w pakiecie rawpy) | biblioteki pomocnicze LibRaw | MIT / IJG |
| tifffile (Christoph Gohlke) | zapis TIFF 16-bit | BSD-3-Clause |
| Metody (zaimplementowane od nowa, bez kopiowania kodu) | Bilateral Guided Upsampling (Chen i in., SIGGRAPH Asia 2016; kod referencyjny google/bgu – Apache-2.0), ECC (Evangelidis i Psarakis 2008), SIFT / DIS optical flow / GrabCut (Rother, Kolmogorov, Blake 2004; OpenCV, Apache-2.0), CIEDE2000 (Sharma, Wu, Dalal 2005), OKLab i mapowanie gamutu (Björn Ottosson 2020/2021, MIT / domena publiczna), filtr prowadzony i szybki filtr prowadzony 3-kanałowy (He, Sun, Tang 2010; He i Sun 2015), adaptacja chromatyczna Bradford CAT (Lam 1985 / CIE-ICC), interpolacja tetraedryczna LUT 3D (Sakamoto 1974; Kasson i in. 1995) | – |
| Rust (`wgpu`, `eframe`/`egui`, `rayon`, `image`, `serde`, `bytemuck`, `pollster`, `clap`, `rfd`) | natywny silnik GPU/CPU i edytor studyjny `retusz-rs` | MIT / Apache-2.0 |
| PyInstaller / Inno Setup | budowanie `.exe` oraz instalatora Windows `Retusz-Setup-x64.exe` | GPL-2.0 z wyjątkiem dla bootloadera / Inno Setup License |

Pliki testowe RAW (Nikon D750, D700) pochodzą z https://raw.pixls.us (CC0 1.0) i nie są dołączone do repozytorium.

Metody bez modeli: Reinhard 2001, Pitié & Kokaram 2007 (MKL), Pitié 2005/2007 (IDT). Retouch4me nie jest
w żaden sposób wykorzystywany (ani kod, ani modele) — może służyć wyłącznie do zewnętrznego porównania gotowych wyników.
