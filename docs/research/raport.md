# Retusz — raport z researchu: co możemy darmowo wykorzystać (GitHub), stan 0.7.0 / 0.8.0, roadmapa do 1.0

**Autor:** Jack4B · **Data:** 2026-11-26 (uzupełnienie tego samego dnia: aplikacja będzie darmowa, rozdz. 11) · **Zakres:** przegląd repo `Jack4B/retusz` (main @ `3602faa`, tag `v0.7.0`), ocena specyfikacji 0.8.0, research GitHub (~120 repozytoriów sprawdzonych przez GitHub API: gwiazdki, ostatni push, licencja; licencje wag modeli sprawdzone osobno w README/LICENSE/model card).

> Zasady, których raport się trzyma: tylko kod MIT / BSD / Apache-2.0 (i wagi z licencją komercyjną) nadaje się do wbudowania; GPL/AGPL/NC tylko jako **inspiracja algorytmiczna** (czytamy artykuły, nie kopiujemy kodu). Żadnej dekompilacji ani ekstrakcji Retouch4me / Evoto / innych komercyjnych narzędzi — porównania wyłącznie na gotowych wynikach i publicznych materiałach.
> Gwiazdki i daty to stan z 2026-11-26 (GitHub API). „Wagi” = licencja plików modelu, która często różni się od licencji kodu.

---

## 1. Streszczenie (TL;DR)

1. **Retusz 0.7.0 jest już pełnym edytorem portretowym** (OKLab, ColorChecker, 16-bit, RAW, Biblioteka/Edycja, D&B v2, Plastyka/Światło 3D, Grading z AI, eksport ICC). Największe luki względem światowej czołówki to nie suwaki, tylko **(a) semantyczne maski AI** (skóra twarzy / skóra ciała / włosy / ubranie / tło / niebo), **(b) prawdziwe usuwanie niedoskonałości i obiektów (inpainting)**, **(c) odszumianie AI i super-rozdzielczość**, **(d) lokalne korekty (gradienty, pędzel z auto-maską, warstwy/maski)** oraz **(e) płynny podgląd GPU w głównym edytorze**.
2. **0.8.0 nie ma jeszcze w repo** — na GitHubie nie istnieje gałąź ani PR 0.8.0, a w `retusz/engine/tone.py` i `faces.py` nie ma `auto_tone`, `shadow_tone`, `highlight_tone`, `hsl_green`, `hsl_blue` ani NMS; YuNet ma domyślnie `score=0.7`. Specyfikację oceniam więc jako plan (rozdz. 3).
3. **Najmocniejsze darmowe klocki o czystej licencji:** MediaPipe Selfie Multiclass (wagi Apache-2.0: włosy / skóra ciała / skóra twarzy / ubranie / akcesoria / tło) + MediaPipe Face Landmarker (478 punktów) jako zamiennik heurystycznej maski skóry; **BiRefNet** (MIT, wagi MIT) / **MODNet** (Apache, wagi Apache) do matowania sylwetki; **LaMa** (Apache) i **MI-GAN** (MIT, wagi MIT) do inpaintingu; **Real-ESRGAN** (BSD-3), **SwinIR/HAT/Swin2SR** (Apache), **NAFNet** (MIT), **Restormer** (MIT), **SCUNet** (Apache) do odszumiania/SR; **colour-science** (BSD-3) do nauki o kolorze; **vkdt** (BSD-2, z wyjątkami) jako wzorzec GPU-pipeline'u RAW.
4. **Pułapki licencyjne:** nawet przy darmowej aplikacji **„darmowa” ≠ „niekomercyjna”** (rozdz. 11). Modele NC (CodeFormer, InsightFace, Sapiens, Depth Anything V2 Base/Large, Deep White Balance, BRIA RMBG, parsery z CelebAMask-HQ) nadają się do publicznego wydania tylko wyjątkowo i często **wcale** (klauzula „research only”, zakaz redystrybucji, użycie zawodowe przez fotografów). GPL/AGPL (darktable, RawTherapee, ART, RapidRAW, spektrafilm, OpenDRT) wymaga relicencjonowania całego programu — **nie rekomendujemy**. Linia wydania publicznego zostaje na MIT/Apache/BSD.
5. **DirectML jest w trybie utrzymania** (Microsoft: „sustained engineering”, nowe funkcje idą do Windows ML). `onnxruntime-directml` na PyPI stoi na 1.24.4, a `onnxruntime` ma 1.32.0. Na Windows 10 DirectML nadal działa, ale warto zaplanować ścieżkę Windows ML / `ort` (Rust) z wyborem EP.
6. **UI:** równolegle istnieją dwa edytory: Python/CustomTkinter (`retusz/gui/app.py`, 1750 linii) i Rust/wgpu/egui (`crates/retusz-app`). Tkinter nie ma podglądu GPU, więc przy 24 Mpx to sufit płynności. Rekomendacja: **jeden front-end Rust + egui + wgpu** (licencje MIT/Apache) z `egui_dock`, silnik Pythona tylko jako backend AI/batch w okresie przejściowym, potem `ort` w Rust. Na Androida: rdzeń Rust + UI Kotlin/Compose.

---

## 2. Stan obecny — co faktycznie jest w kodzie (main @ 3602faa = v0.7.0)

Zweryfikowane w kodzie i README (nie z pamięci). Wydania od 0.1.8 do 0.7.0 powstały 25–26.11.2026; v0.7.0 opublikowano 2026-11-26 o 13:52 (czas Warszawa, UTC+1). CI `build-windows` przeszło na wszystkich tagach.

| Obszar | Stan w 0.7.0 | Gdzie | Uwagi |
|---|---|---|---|
| Dopasowanie do wzorca (AI) | ✅ ModFlows B0 (wbudowany, MIT) i B6 (do pobrania, `models-v1`), wiele wzorców w serii (`reference_bank`) | `engine/modflows.py`, `engine/color.py`, `sidecar.py` | dobrze |
| Grading z AI (transfer z wersji AI na pełną rozdzielczość) | ✅ SIFT + ECC + DIS flow, BGU (własna implementacja), okno Obok siebie / 1:1 / Miganie | `engine/align.py`, `bgu.py`, `transfer.py`, `gui/transfer_view.py` | gałąź `fix/ai-alignment-robust` = tag v0.1.5 (wcielona) |
| Dodge & Burn | ✅ v2 (filtr prowadzony / WLS na L*, kształty z punktów twarzy) + pędzel | `engine/dnb.py`, `crates/retusz-core/src/dnb.rs` | gałąź `feature/dnb-v2` = tag v0.1.6 |
| Plastyka / Światło 3D | ✅ Depth Anything V2 **Small** int8 (Apache) → normalne, 2 osie światła, separacja tła (`bg_exposure`), GrabCut | `engine/relight.py` | maska postaci = głębia + GrabCut (brak modelu matowania) |
| Skóra | ✅ maska heurystyczna (YCrCb + Mahalanobis od rdzenia twarzy + filtr prowadzony), Melanina/Hemoglobina, De-Shine, Auto-Heal (klasyczny), Oczy i zęby, Odcień skóry | `engine/faces.py`, `skin_physics.py` | **brak semantycznej segmentacji** (włosy/zarost/okulary/ubranie) |
| Ton/kolor | ✅ OKLab, EV, Kontrast, Półtony, Dehaze, Światła/Cienie, Biele/Czernie, WB Bradford, Przejrzystość (Fast LLF), Jaskrawość | `engine/tone.py`, `llf.py`, `wb.py` | brak krzywej tonalnej, pełnego HSL 8-kanałowego, split toningu (planowane częściowo w 0.8.0) |
| ColorChecker 24 + raport ΔE00 | ✅ Root-Polynomial | `engine/colorchecker.py` | mocny wyróżnik |
| Detal | ✅ 4 tryby ostrzenia (portretowy, RL, Clamped USM, Soft Focus), Struktura, Odszumianie OKLab | `engine/detail.py` | odszumianie klasyczne, bez AI |
| RAW | ✅ rawpy/LibRaw (CDDL), NEF + inne formaty, 16-bit | `rawio.py` | brak korekcji obiektywów (Lensfun), brak odszumiania AI przed demozaikowaniem |
| Kadr / prostowanie, histogram, pipeta WB, wektorskop OKLCh | ✅ | `engine/geometry.py`, `histo.py`, `gui/histview.py` | |
| Seria / batch, presety (6 wbudowanych + własne), →seria, CLI `--preset` | ✅ | `batch.py`, `presets.py`, `cli.py` | |
| Eksport | ✅ JPG 4:4:4, TIFF/PNG 16-bit, WebP, 5 profili ICC | `export.py`, `colorman.py` | brak XMP / DNG |
| Biblioteka (G) / Edycja (D), gwiazdki, filtry | ✅ | `gui/library.py` | brak flag „wybierz/odrzuć” (P/X), brak paska filmstrip |
| Cofnij | ✅ Ctrl+Z (pojedyncze kroki) | `gui/app.py` | brak panelu Historii / migawek / wirtualnych kopii |
| Edytor GPU Rust (`Retusz-Studio-GPU.exe`) | ✅ egui 0.30 + wgpu 23, shader WGSL (LUT 33³, D&B, winieta, zebra) | `crates/` | **drugi, równoległy front-end** — ryzyko rozjazdu funkcji |
| Instalator | ✅ Inno Setup + zip portable | `installer/` | |
| Bramki jakości | ✅ `tools/quality_gate.py`, `dnb_gate.py`, `transfer_gate.py`, `colorchecker_gate.py`; 24 pliki testów pytest | `tools/`, `tests/` | dobry fundament |

**Obserwacje krytyczne:**
- README używa sformułowań typu „0% halo”, „100% porów”. Bramki mierzą halo i szwy, ale twierdzenia absolutne warto zastąpić liczbami z bramki (np. „halo < X ΔL na krawędzi”), żeby nie obiecywać czegoś, czego test nie gwarantuje.
- Kod jest zwarty (~13 tys. linii Py+Rust). Każda funkcja istnieje w dwóch implementacjach (Python + Rust), co podwaja koszt każdej zmiany algorytmu.
- `AGENTS.md` wciąż opisuje stan 0.1.4 (stos CustomTkinter, bez Rust). Trzeba go zaktualizować, bo Claude i inne agenty pracują według niego.

---

## 3. Specyfikacja 0.8.0 — weryfikacja i krytyka

### 3.1 Co jest w repo, a co tylko w specyfikacji
| Element specyfikacji 0.8.0 | W repo (main / gałęzie)? |
|---|---|
| Układ Lightroom Classic: spokojny górny pasek, Loupe Toolbar pod zdjęciem, jedna kolumna suwaków, szuflady zwinięte, stopka Cofnij/Ponów/Kopiuj/Wklej/→Seria/Zeruj | ❌ brak (main = 0.7.0: prawy dok, 2-kolumnowa siatka, podpisy pod suwakami) |
| `★ Auto Ton AI` (percentyle P1…P99 + L* skóry ≈ 67) | ❌ brak `auto_tone` w `tone.py` |
| `shadow_tone`, `highlight_tone`, `hsl_green`, `hsl_blue` | ❌ brak |
| YuNet `score=0.45` + NMS | ❌ `faces.detect(score=0.7)`, brak NMS |
| Bariera gradientowo-luminancyjna (oprawki, wąsy, zarost) | ❌ jest tylko odrzucenie najciemniejszych 25% w rdzeniu twarzy i `protect_features` wokół 5 punktów |

Wniosek: 0.8.0 budowane jest poza GitHubem (lokalnie). Przed wydaniem trzeba je wypchnąć na gałąź `feature/v0.8.0-…` i przepuścić przez bramki.

### 3.2 Ocena decyzji UI (dobre)
- Jedna kolumna suwaków (nazwa | suwak | wartość), zwinięte szuflady, pasek narzędzi pod histogramem, Loupe Toolbar pod zdjęciem i stopka Kopiuj/Wklej/Seria to **trafne kopie wzorców Lightroom Classic**. Usunięcie dwukolorowych podpisów pod 24 suwakami i zostawienie 3-pikselowego gradientu tylko pod suwakami koloru to dobra decyzja (redukcja szumu wizualnego).
- **Czego wciąż brakuje względem LR Classic** (lista do 0.9):
  1. **Panele zwijane z nagłówkami** (Podstawowe / Krzywa / HSL / Color Grading / Detal / Optyka / Kalibracja / Retusz), z trybem „solo” (Alt+klik: otwarty tylko jeden panel).
  2. **Przełącznik on/off panelu** (oczko) i **reset sekcji dwuklikiem nazwy**; dwuklik suwaka = reset jest, ale trzeba go ujednolicić wszędzie.
  3. **Precyzja suwaka:** strzałki ±1 / Shift ±10, kółko myszy nad suwakiem, wpisywanie wartości, scrubbing po etykiecie.
  4. **Panel Historii + Migawki + Wirtualne kopie** (nie tylko Ctrl+Z).
  5. **Kopiuj/Wklej z wyborem** (okno z checkboxami: ton / kolor / skóra / D&B / kadr). Kadru i pędzla nie należy synchronizować domyślnie.
  6. **Ostrzeżenia przycinania** (J): prześwietlenia na czerwono, cienie na niebiesko, plus klik w trójkąty histogramu; **przeciąganie po histogramie** zmienia Ekspozycję, Światła, Cienie, Biele i Czernie.
  7. **Skróty jak w LR:** `\` przed/po, `Y` obok siebie, `R` kadr, `K` pędzel, `W` pipeta, `M` maska, `G/D` moduły, `P/X/U` flagi, `0–5` gwiazdki, `Ctrl+Shift+C/V` kopiuj/wklej, `Ctrl+Alt+V` wklej z poprzedniego, `Ctrl+'` wirtualna kopia, `?` ściągawka skrótów.
  8. **Filmstrip** na dole w trybie Edycja i **Survey/Compare** (N/C) do selekcji serii.
  9. **Podgląd presetu po najechaniu** (hover preview) i **siła presetu / stylu** (suwak Amount, jak „Profile Amount” w LR).
  10. **Panel Masek** z listą masek, dodawaniem/odejmowaniem/przecinaniem (Subject, Background, People → Face skin / Body skin / Hair / Eyes / Lips / Teeth, Linear, Radial, Brush z Auto Mask), suwakami korekt per maska i nakładką (overlay, O).

### 3.3 Krytyka algorytmów 0.8.0 i lepsze alternatywy

**a) Auto Ton AI z celem L\* ≈ 67**
- **Mieszanie skal:** „L\* ≈ 67 w OKLab” miesza dwie skale. CIELAB L\* to 0–100, a OKLab L to 0–1 (L\* 67 ≈ OKLab L ≈ 0,70, ale relacja nie jest liniowa). Trzeba jawnie wybrać przestrzeń i przeliczać wartości.
- **Jeden twardy cel jest błędny dla różnych karnacji i intencji.** Jasność skóry zależy od fototypu: ciemna karnacja ma typowo L\* 35–50, jasna 65–75. Stały cel 67 rozjaśni ciemną skórę (efekt „wyprania”) i spłaszczy low-key / zdjęcia pod światło. Propozycja:
  - cel **zależny od karnacji**, wyznaczany z kąta ITA° (Individual Typology Angle = atan((L\*−50)/b\*), Chardon 1991) mierzonego na czystej skórze twarzy z ograniczeniem przesunięcia (np. maks. ±0,7 EV z samej twarzy);
  - **mieszanie celu sceny i celu twarzy** (np. 60/40) z wagą zależną od udziału twarzy w kadrze;
  - **wykrywanie intencji** (high-key / low-key z histogramu i średniej tła), a potem łagodniejszy cel;
  - **ograniczenia:** brak przycinania (P99,5 < 0,98 liniowo), średnia zmiana ≤ X EV, ochrona nasycenia skóry;
  - **„Auto” jako punkt startowy:** suwaki ustawiane na wartości widoczne dla użytkownika, z możliwością cofnięcia jednym Ctrl+Z (tak działa LR).
- **Modele uczone o licencjach permisywnych** (do rozważenia jako „Auto v2”): `HuiZeng/Image-Adaptive-3DLUT` (Apache-2.0, 958★) i `google/hdrnet` (Apache-2.0, zarchiwizowane). Uwaga: oba trenowane na **MIT-Adobe FiveK**, a licencja tego zbioru ogranicza użycie do badań. Wagi z FiveK traktujemy jako ryzykowne. Bezpieczniej wytrenować własny mały predyktor parametrów (EV/kontrast/światła/cienie) na **własnych, zgodnych z prawem** parach przed/po (zdjęcia Jack4B za zgodą) — bez kopiowania cudzych wag. `Li-Chongyi/Zero-DCE` i `sjmoran/curl-image-enhancement` nie mają pliku licencji (= brak prawa do użycia).

**b) YuNet score = 0.45 + NMS**
- `cv2.FaceDetectorYN` ma **wbudowane NMS** (parametr `nms_threshold`, domyślnie 0,3), więc osobny NMS jest zbędny albo powinien być tylko scaleniem między skalami/kafelkami.
- Próg 0,45 zwiększy fałszywe trafienia (twarze na plakatach, wzory na tkaninach, odbicia). W Retuszu każda „twarz” uruchamia maskę skóry, D&B i Plastykę, więc fałszywa twarz psuje obraz. Propozycja: **dwustopniowo** — kandydaci od 0,45, akceptacja tylko przy spójnych punktach (oczy poziomo, rozstaw oczu / szerokość ramki w normie), sensownym rozmiarze i potwierdzeniu przez segmentację skóry twarzy (MediaPipe multiclass: klasa face-skin w ramce > 25%). Dla grup: **detekcja kafelkowa / wieloskalowa** (twarze < 40 px na 24 Mpx giną po przeskalowaniu do 640 px).
- Alternatywy permisywne: **MediaPipe Face Detector (BlazeFace short/full-range)** i **Face Landmarker 478 pkt** (`google-ai-edge/mediapipe`, Apache-2.0, 37,9k★); `biubug6/Pytorch_Retinaface` (MIT, 3,0k★; wagi trenowane na WIDER FACE — zbiór do badań, ryzyko średnie); `yakhyo/uniface` (MIT, 2,3k★; zbiera RetinaFace/SCRFD/parsing, ale **SCRFD i wagi InsightFace są NC** — z uniface brać tylko modele o czystej licencji). **Nie używać** wag InsightFace / SCRFD / buffalo (README: „non-commercial research purposes only”).

**c) Bariera gradientowo-luminancyjna (oprawki, wąsy, zarost)**
- Heurystyka gradientowa łapie też zmarszczki, rzęsy, brwi i cienie nosa, a zarost blond czy siwy przepuszcza. Lepiej oprzeć się na **semantyce**:
  - **MediaPipe Selfie Multiclass 256×256** — klasy: tło, włosy, skóra ciała, skóra twarzy, ubranie, inne/akcesoria. Wagi **Apache-2.0** według model card; plik .tflite ~16 MB, dostępne konwersje do ONNX. Okulary trafiają do „akcesoriów”, broda zwykle do „włosów” — dokładnie to, czego potrzebujemy do wykluczeń.
  - **MediaPipe Face Landmarker (478 pkt + iris)** — wielokąty oczu, brwi, ust i nozdrzy do precyzyjnego `protect_features` zamiast 5 punktów YuNet.
  - Połączenie: maska skóry = face-skin ∪ body-skin (MediaPipe) × (1 − wielokąty oczu/brwi/ust) → dopasowanie krawędzi filtrem prowadzonym w pełnej rozdzielczości (już mamy) → obecna kalibracja Mahalanobisa jako korekta lokalna.
  - Uwaga z model card: model nie jest do „pixel-perfect masks”, więc trzeba go łączyć z guided upsamplingiem / BGU (już mamy w repo).
  - **Nie** brać BiSeNet z CelebAMask-HQ (`zllrunning/face-parsing.PyTorch`, `yakhyo/face-parsing`) do wersji komercyjnej. Kod jest MIT, ale CelebAMask-HQ jest „for non-commercial research purposes only” — wagi dziedziczą to ryzyko. Można ich używać do **porównań wewnętrznych** w bramce.

**d) Split toning w OKLab i HSL tylko dla zieleni i błękitu**
- Ochrona czystej czerni w split toningu jest dobra. Dodać **balans** (gdzie przejście cienie/światła) i **półtony**, czyli pełne Color Grading 3-kołowe jak w LR. Koła lepiej robić w OKLCh (już używane).
- Zamiast 2 suwaków HSL: **pełny mikser 8 kanałów** (czerwony, pomarańczowy, żółty, zielony, akwamaryna, niebieski, fioletowy, magenta) × (odcień, nasycenie, luminancja) w OKLCh z gładkimi wagami kątowymi i **globalną ochroną skóry** (jest już wektorskop 52°). „Zieleń (Plener)” i „Błękit (Koszule)” mogą zostać jako szybkie presety na tym mikserze.
- Biały balans koszul niebieskich lepiej robić **maską ubrania** (MediaPipe klasa „clothes”) i lokalnym WB niż globalnym HSL, który dotknie też nieba i oczu.

---

## 4. Analiza luk względem światowej czołówki (wyłącznie publiczne informacje)

| Funkcja (publicznie opisywana) | Lightroom / LR Classic | Capture One | Retouch4me | Evoto | PortraitPro | Luminar Neo | **Retusz 0.7.0** | Jak domknąć darmowo |
|---|---|---|---|---|---|---|---|---|
| Maski AI: osoba, tło, niebo, części twarzy | ✅ (Select Subject/Sky/Background/People) | ✅ (AI masks) | — | ✅ | częściowo | ✅ | ⚠️ skóra heurystyczna + głębia | MediaPipe multiclass + Face Landmarker, BiRefNet/MODNet, SAM 2 (interaktywnie) |
| Usuwanie niedoskonałości / obiektów | ✅ Generative Remove, Healing | ✅ Heal/Clone | ✅ Heal | ✅ | ✅ | ✅ GenErase | ⚠️ Auto-Heal klasyczny, brak pędzla healing/clone | LaMa / MI-GAN + klasyczny „patch” w pasmach częstotliwości |
| Dodge & Burn automatyczny | — | — | ✅ (Dodge&Burn) | ✅ | ✅ | ✅ (Skin AI) | ✅ v2 | dopracować: wieloskalowy, z semantyką |
| Wyrównanie odcienia skóry w serii | — | ✅ (Skin Tone tool) | ✅ (Skin Tone) | ✅ | — | — | ✅ (wzorzec, odcień, melanina/hemoglobina) | — (mocna strona) |
| Odszumianie AI RAW | ✅ Denoise | ✅ | — | — | — | ✅ Noiseless | ❌ | NAFNet / SCUNet / Restormer (ONNX), vkdt jddcnn jako wzorzec |
| Super-rozdzielczość | ✅ Super Resolution | — | — | — | — | ✅ Upscale | ❌ | Real-ESRGAN (BSD-3), SwinIR / HAT / Swin2SR (Apache) |
| Relighting | ⚠️ | — | ✅ (Portrait Volumes) | ✅ | ✅ | ✅ Relight AI | ✅ Światło 3D (głębia) | lepsze normalne / matte; IC-Light tylko jako referencja (ciężki, SD1.5) |
| Odblaski na okularach, czyszczenie tła / tkanin | — | — | ✅ (Clean Backdrop, Fabric) | ✅ | — | — | ❌ | inpainting + maska akcesoriów / tła |
| Korekcja obiektywu (profile) | ✅ | ✅ | — | — | — | ✅ | ❌ | Lensfun (LGPL-3 biblioteka, baza CC BY-SA 3.0) jako dynamiczna zależność albo własna baza |
| Krzywa tonalna / pełny HSL / Color Grading | ✅ | ✅ | — | ✅ | — | ✅ | ⚠️ (0.8.0 częściowo) | własna implementacja OKLCh |
| Warstwy / maski lokalne, gradienty | ✅ | ✅ (Layers) | — | ✅ | — | ✅ | ❌ (pędzel D&B) | własny model warstw + maski |
| Historia, migawki, wirtualne kopie | ✅ | ✅ | — | ✅ | ✅ | ✅ | ⚠️ tylko Ctrl+Z | własny model poleceń |
| XMP / DNG | ✅ | częściowo | — | — | — | — | ❌ | `dnglab` (LGPL-2.1, CLI) jako narzędzie zewnętrzne; XMP — własny zapis |
| Kalibracja ColorChecker + raport ΔE00 | ✅ (profile) | ✅ | — | — | — | — | ✅ | — (mocna strona) |
| Transfer gradingu z wersji AI na pełną rozdzielczość | — | — | — | — | — | — | ✅ **unikalne** | — (wyróżnik) |

**Wyróżniki Retusza, które warto chronić i eksponować:** transfer gradingu z AI na oryginał, ColorChecker z ΔE00, fizyka skóry (melanina/hemoglobina, De-Shine), wzorce per zdjęcie w serii, bramki jakości z liczbami.

---

## 5. Rekomendowane narzędzia (zweryfikowane repozytoria)

Legenda: **✅ KOMERCYJNIE** = kod i wagi permisywne · **⚠️ WARUNKOWO** = kod OK, ale wagi/dataset/zależność wymagają decyzji · **📚 TYLKO NAUKA** = GPL/AGPL/NC — czytamy artykuł, nie kopiujemy kodu ani wag.

### 5.1 Segmentacja, maski, twarz
| Repo | ★ | Ostatni push | Kod | Wagi | Co daje | Dopasowanie do stosu | Wysiłek / ryzyko | Ocena |
|---|---|---|---|---|---|---|---|---|
| [google-ai-edge/mediapipe](https://github.com/google-ai-edge/mediapipe) | 37 873 | 2026-11-25 | Apache-2.0 | Selfie Multiclass 256: Apache-2.0 (model card); Face Landmarker: model card | 6 klas (włosy, skóra ciała, skóra twarzy, ubranie, akcesoria, tło), 478 pkt twarzy, iris | .tflite → ONNX (tf2onnx), onnxruntime/DirectML; Android natywnie | Ś / niskie. Uwaga: źródła różnie opisują skalowanie wejścia (0..1 vs −1..1) — zweryfikować testem IoU | **✅** |
| [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet) | 4 458 | 2026-10-25 | MIT | MIT (oficjalne; **nie** wariant z danymi BRIA — README: NC) | matowanie sylwetki wysokiej rozdzielczości (włosy) | ONNX dostępne; ciężki (~200+ MB, wolny na CPU) — wariant lite | Ś / niskie | **✅** |
| [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet) | 4 391 | 2024-05-06 | Apache-2.0 | Apache-2.0 (README: „code, models, and demos”) | szybkie matowanie portretu bez trimapy (~25 MB) | ONNX w repo; real-time | N / niskie | **✅** |
| [PeterL1n/BackgroundMattingV2](https://github.com/PeterL1n/BackgroundMattingV2) | 7 215 | 2024-06-19 | MIT | MIT | matowanie przy znanym tle (studio) | wymaga zdjęcia czystego tła | Ś | ✅ (nisza: studio) |
| [hustvl/ViTMatte](https://github.com/hustvl/ViTMatte) | 574 | 2025-08-13 | MIT | sprawdzić (trening na Composition-1k / Distinctions-646 — zbiory badawcze) | doprecyzowanie matte z trimapy (włosy) | ciężki ViT | W / średnie | ⚠️ |
| [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | 20 245 | 2026-05-30 | Apache-2.0 | Apache-2.0 | interaktywne „kliknij obiekt” do masek | ONNX encoder ~ duży; tiny/small realne na GPU | Ś / niskie | **✅** (narzędzie „Zaznacz obiekt”) |
| [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net) | 9 903 | 2024-06-26 | Apache-2.0 | Apache-2.0 | saliency / human seg (starszy) | ONNX | N | ✅ (fallback) |
| [danielgatis/rembg](https://github.com/danielgatis/rembg) | 25 685 | 2026-11-24 | MIT | różne (BRIA = NC!) | gotowy wrapper wielu modeli | Python + onnxruntime | N / **licencję sprawdzać per model** | ⚠️ (wzorzec integracji) |
| [PramaLLC/BEN2](https://github.com/PramaLLC/BEN2) | 302 | 2026-01-02 | MIT | base: sprawdzić (lepszy model jest płatny) | usuwanie tła | ONNX | Ś | ⚠️ |
| [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo) | 1 150 | 2026-10-20 | Apache-2.0 | per model (YuNet MIT — już używamy) | YuNet, PP-HumanSeg, EfficientSAM | cv2.dnn | N | ✅ |
| [1adrianb/face-alignment](https://github.com/1adrianb/face-alignment) | 7 549 | 2026-04-06 | BSD-3 | trening 300W-LP / LS3D-W (badawcze) | 68 pkt 2D/3D | PyTorch → ONNX | Ś / średnie | ⚠️ |
| [jhb86253817/PIPNet](https://github.com/jhb86253817/PIPNet) | 557 | 2023-02-14 | MIT | trening 300W / WFLW (badawcze) | szybkie punkty twarzy | ONNX | Ś / średnie | ⚠️ |
| [GoGoDuck912/Self-Correction-Human-Parsing](https://github.com/GoGoDuck912/Self-Correction-Human-Parsing) | 1 269 | 2024-08-30 | MIT | LIP/ATR/Pascal (badawcze) | parsing całej sylwetki (20 klas) | PyTorch | Ś / średnie | ⚠️ |
| [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch), [yakhyo/face-parsing](https://github.com/yakhyo/face-parsing), [FacePerceiver/facer](https://github.com/FacePerceiver/facer) | 2 596 / 339 / 417 | 2023 / 2026-10 / 2025-02 | MIT | CelebAMask-HQ = **tylko badania** | 19 klas twarzy (okulary, brwi, usta, włosy) | ONNX (yakhyo) | ryzyko licencyjne wag | 📚 / test wewnętrzny |
| [deepinsight/insightface](https://github.com/deepinsight/insightface) | 30 309 | 2026-10-19 | MIT (kod) | **NC** (README) | SCRFD, punkty, rozpoznawanie | — | — | 📚 |
| [facebookresearch/sapiens](https://github.com/facebookresearch/sapiens) | 5 442 | 2026-05-26 | **CC BY-NC 4.0** | NC | segmentacja części ciała, normalne, głębia (top jakość) | — | — | 📚 |
| [PeterL1n/RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) | 9 646 | 2024-04-02 | **GPL-3.0** | — | matowanie | — | — | 📚 |
| [pq-yang/MatAnyone](https://github.com/pq-yang/MatAnyone) | 1 647 | 2026-03-04 | NOASSERTION (S-Lab) | NC | matowanie | — | — | 📚 |

### 5.2 Inpainting / usuwanie niedoskonałości
| Repo | ★ | Push | Kod | Wagi | Co daje | Ocena |
|---|---|---|---|---|---|---|
| [advimman/lama](https://github.com/advimman/lama) | 10 417 | 2025-02-05 | Apache-2.0 | Big-LaMa: dystrybuowane w ramach projektu (Apache); trening na Places (dataset badawczy — ryzyko niskie/średnie) | najlepszy „lekki” inpainting dużych masek, odporny na rozdzielczość; ONNX ~200 MB | **✅** (z adnotacją o datasetcie) |
| [Picsart-AI-Research/MI-GAN](https://github.com/Picsart-AI-Research/MI-GAN) | 715 | 2026-09-14 | MIT | **MIT** (`LICENSE-WEIGHTS`) | inpainting na urządzeniach mobilnych (mały, szybki) — idealny na Androida | **✅** |
| [Sanster/IOPaint](https://github.com/Sanster/IOPaint) | 23 282 | 2025-04-29 (zarchiwizowane) | Apache-2.0 | per model | wzorzec UX pędzla „usuń” + integracji LaMa/MI-GAN | ✅ (wzorzec, nie zależność) |
| [Carve-Photos/lama](https://github.com/Carve-Photos/lama) | 26 | 2024-05-14 | Apache-2.0 | — | eksport LaMa do ONNX | ✅ (narzędzie) |

**Jak użyć w portrecie:** inpainting AI tylko na **niskiej częstotliwości** (kolor/ton), a tekstura porów przeniesiona z sąsiedztwa (rozdzielenie częstotliwości). Dzięki temu nie ma „plastikowej łaty” — tak samo działa ręczna technika retuszerów (frequency separation + healing). Pędzel „Usuń” = maska → LaMa/MI-GAN na wycinku 512–1024 px → wklejenie przez BGU/guided upsampling (mamy) → przywrócenie ziarna/porów.

### 5.3 Odszumianie, ostrość, super-rozdzielczość, restauracja twarzy
| Repo | ★ | Push | Kod | Wagi | Co daje | Ocena |
|---|---|---|---|---|---|---|
| [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) | 37 678 | 2024-08-06 | BSD-3 | BSD-3 (wydania repo) | SR ×2/×4, realne zdjęcia | **✅** (uwaga: potrafi „malować” skórę — tylko tło/ubrania albo z maską) |
| [JingyunLiang/SwinIR](https://github.com/JingyunLiang/SwinIR) | 5 636 | 2024-05-14 | Apache-2.0 | Apache (wydania) | SR, odszumianie, JPEG | ✅ |
| [XPixelGroup/HAT](https://github.com/XPixelGroup/HAT) | 1 610 | 2024-06-02 | Apache-2.0 | Apache | SR top-jakość (ciężki) | ✅ |
| [mv-lab/swin2sr](https://github.com/mv-lab/swin2sr) | 698 | 2024-08-19 | Apache-2.0 | Apache | SR skompresowanych zdjęć (JPEG z AI!) | ✅ — pasuje do „Grading z AI” (małe wersje AI) |
| [zhengchen1999/DAT](https://github.com/zhengchen1999/DAT) | 546 | 2024-11-26 | Apache-2.0 | Apache | SR | ✅ |
| [megvii-research/NAFNet](https://github.com/megvii-research/NAFNet) | 3 232 | 2024-07-03 | MIT (plik LICENSE) | trening SIDD/GoPro (badawcze) | odszumianie / deblur, lekki | ✅/⚠️ (dataset) |
| [swz30/Restormer](https://github.com/swz30/Restormer) | 2 686 | 2025-10-23 | MIT | j.w. | odszumianie, deblur, deraining | ✅/⚠️ |
| [cszn/SCUNet](https://github.com/cszn/SCUNet) | 839 | 2023-12-15 | Apache-2.0 | Apache; trening na danych syntetycznych | ślepe odszumianie realnych zdjęć | **✅** |
| [cszn/KAIR](https://github.com/cszn/KAIR) | 3 536 | 2024-10-02 | MIT | per model | DnCNN, FFDNet, DRUNet, BSRGAN — toolbox | ✅ |
| [XPixelGroup/BasicSR](https://github.com/XPixelGroup/BasicSR) | 8 427 | 2024-07-21 | Apache-2.0 | — | trening własnych modeli | ✅ (narzędzie) |
| [chaiNNer-org/spandrel](https://github.com/chaiNNer-org/spandrel) | 318 | 2026-04-12 | MIT | — | ładowanie dziesiątek architektur SR/restauracji → eksport ONNX | ✅ (narzędzie build-time) |
| [cchen156/Learning-to-See-in-the-Dark](https://github.com/cchen156/Learning-to-See-in-the-Dark) | 5 566 | 2025-09-15 | MIT | MIT; dataset SID | odszumianie RAW w słabym świetle (Sony/Fuji) | ⚠️ (aparat-specyficzne) |
| [hanatos/vkdt](https://github.com/hanatos/vkdt) | 645 | 2026-11-25 | **BSD-2** (z wyjątkami per moduł!) | jddcnn: osobny pakiet wag (sprawdzić) | GPU (Vulkan) pipeline RAW: demosaic, denoise, `llap` (local laplacian), `inpaint`, `guided`, `autoexp`, `lens`, `deconv` | ✅ dla modułów BSD; **❌ `OpenDRT` (GPL-3, readme modułu)**; **⚠️ `filmsim` = implementacja spektrafilm (GPL-3)** |
| [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) | 37 737 | 2024-07-26 | Apache-2.0 z komponentami stron trzecich | trening FFHQ (CC BY-NC-SA!) | restauracja twarzy | ⚠️/📚 — w retuszu portretów i tak niewskazane (zmienia tożsamość) |
| [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) | 18 265 | 2025-11-18 | **S-Lab License 1.0 (NC)** | NC | restauracja twarzy | 📚 |
| [yangxy/GPEN](https://github.com/yangxy/GPEN) | 2 612 | 2026-03-10 | brak pliku licencji w API | — | restauracja twarzy | 📚 |
| [upscayl/upscayl](https://github.com/upscayl/upscayl) | 53 271 | 2026-11-26 | **AGPL-3.0** | — | wzorzec UX upscalera | 📚 |
| [chaiNNer-org/chaiNNer](https://github.com/chaiNNer-org/chaiNNer) | 6 196 | 2026-11-13 | **GPL-3.0** | — | węzłowy pipeline | 📚 |

### 5.4 RAW, kolor, zarządzanie barwą, film
| Repo | ★ | Push | Licencja | Co daje | Ocena |
|---|---|---|---|---|---|
| [letmaik/rawpy](https://github.com/letmaik/rawpy) | 831 | 2026-10-31 | MIT (LibRaw w środku: LGPL-2.1/CDDL) | już używamy | ✅ |
| [LibRaw/LibRaw](https://github.com/LibRaw/LibRaw) | 1 639 | 2026-11-26 | LGPL-2.1 / CDDL-1.0 | dekodowanie RAW | ✅ jako niezmieniona zależność |
| [colour-science/colour](https://github.com/colour-science/colour) | 2 697 | 2026-11-23 | BSD-3 | CAT, ΔE, przestrzenie, krzywe, spektra — referencja do testów | **✅** |
| [colour-science/colour-demosaicing](https://github.com/colour-science/colour-demosaicing) | 337 | 2026-11-23 | BSD-3 | Menon 2007, Malvar 2004, DDFAPD | ✅ (demosaic po swojej stronie) |
| [colour-science/colour-checker-detection](https://github.com/colour-science/colour-checker-detection) | 295 | 2026-11-23 | BSD-3 | detekcja ColorCheckera (porównanie z naszą) | ✅ |
| [AcademySoftwareFoundation/OpenColorIO](https://github.com/AcademySoftwareFoundation/OpenColorIO) | 2 147 | 2026-10-01 | BSD-3 | LUT .cube/.clf, ACES, wyświetlanie | ✅ (import LUT-ów klienta) |
| [AcademySoftwareFoundation/OpenImageIO](https://github.com/AcademySoftwareFoundation/OpenImageIO) | 2 409 | 2026-11-26 | Apache-2.0 | I/O formatów (EXR, JXL, HEIF) | ✅ (opcjonalnie) |
| [mm2/Little-CMS](https://github.com/mm2/Little-CMS) | 764 | 2026-11-24 | MIT | ICC, soft-proofing | ✅ |
| [awxkee/moxcms](https://github.com/awxkee/moxcms) | 62 | 2026-11-21 | BSD-3 | CMS w czystym Rust | ✅ (dla front-endu Rust) |
| [kamadak/exif-rs](https://github.com/kamadak/exif-rs) | 261 | 2026-11-05 | BSD-2 | EXIF w Rust | ✅ |
| [image-rs/image](https://github.com/image-rs/image) | 5 920 | 2026-11-11 | Apache-2.0/MIT | już używamy w Rust | ✅ |
| [dnglab/dnglab](https://github.com/dnglab/dnglab) | 723 | 2026-11-26 | LGPL-2.1 | RAW → DNG, dekoder w Rust (rawler) | ⚠️ tylko jako osobny program / niezmieniona biblioteka dynamiczna |
| [darktable-org/rawspeed](https://github.com/darktable-org/rawspeed) | 464 | 2026-11-25 | LGPL-2.1 | szybkie dekodowanie RAW | ⚠️ j.w. |
| [lensfun/lensfun](https://github.com/lensfun/lensfun) + [letmaik/lensfunpy](https://github.com/letmaik/lensfunpy) | 896 / 166 | 2026-11-17 / 2026-01-30 | LGPL-3.0 (lib), **baza CC BY-SA 3.0**, wrapper MIT | profile dystorsji/winiety/CA | ⚠️ biblioteka dynamicznie + atrybucja bazy |
| [libvips/libvips](https://github.com/libvips/libvips) | 11 841 | 2026-11-25 | LGPL-2.1 | szybkie miniatury i duże obrazy | ⚠️ opcjonalnie (Biblioteka) |
| [maria-larchenko/modflows](https://github.com/maria-larchenko/modflows) | 74 | 2026-10-30 | MIT | ModFlows — już używamy | ✅ |
| [HuiZeng/Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT) | 958 | 2023-12-16 | Apache-2.0 | uczony auto-ton przez 3D LUT | ⚠️ (wagi z FiveK) |
| [google/hdrnet](https://github.com/google/hdrnet) | 877 | 2023 (zarch.) | Apache-2.0 | bilateral learning (rodzina BGU) | ⚠️ (FiveK) |
| [mahmoudnafifi/C5](https://github.com/mahmoudnafifi/C5) | 136 | 2024-01-14 | Apache-2.0 | uczony auto-WB międzyaparatowy | ⚠️ (sprawdzić wagi) |
| [google/ffcc](https://github.com/google/ffcc) | 225 | 2021 (zarch.) | Apache-2.0 | Fast Fourier Color Constancy — szybki auto-WB | ✅ (algorytm + kod) |
| [mahmoudnafifi/Deep_White_Balance](https://github.com/mahmoudnafifi/Deep_White_Balance) | 629 | 2023-07-05 | **CC BY-NC-SA 4.0** | — | 📚 |
| [mahmoudnafifi/Exposure_Correction](https://github.com/mahmoudnafifi/Exposure_Correction), [mixedillWB](https://github.com/mahmoudnafifi/mixedillWB) | 595 / 128 | 2023 | NOASSERTION | — | 📚 dopóki licencja niejasna |
| [andreavolpato/spektrafilm](https://github.com/andreavolpato/spektrafilm) | 2 438 | 2026-07-14 | **GPL-3.0** | spektralna symulacja filmu | 📚 (pomysły; mamy własny Film-Simulator) |
| [sobotka/AgX](https://github.com/sobotka/AgX), [EaryChow/AgX](https://github.com/EaryChow/AgX) | 337 / 152 | 2025 / 2026 | brak licencji w API | transformacja wyświetlania „filmowa” | 📚 (implementować z opisu) |
| [darktable-org/darktable](https://github.com/darktable-org/darktable), [RawTherapee](https://github.com/RawTherapee/RawTherapee), [artraweditor/ART](https://github.com/artraweditor/ART), [aurelienpierreeng/ansel](https://github.com/aurelienpierreeng/ansel) | 13 507 / 4 311 / 529 / 1 067 | 2026-11 | **GPL-3.0** | referencje algorytmów (filmic/sigmoid, diffuse-or-sharpen, CA, capture sharpening) | 📚 |

### 5.5 Relighting, głębia, normalne
| Repo | ★ | Push | Kod | Wagi | Ocena |
|---|---|---|---|---|---|
| [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) | 9 236 | 2026-03-24 | Apache-2.0 | **Small: Apache; Base/Large: CC BY-NC** | ✅ Small (już używamy) |
| [prs-eth/Marigold](https://github.com/prs-eth/Marigold) | 3 262 | 2026-09-06 | Apache-2.0 | sprawdzić per checkpoint (oparte na SD) | ⚠️ — jakość głębi/normalnych świetna, ale ciężkie (diffusion) |
| [Stable-X/StableNormal](https://github.com/Stable-X/StableNormal) | 791 | 2025-08-02 | Apache-2.0 | sprawdzić | ⚠️ (ciężkie) |
| [lllyasviel/IC-Light](https://github.com/lllyasviel/IC-Light) | 8 587 | 2025-02-20 | Apache-2.0 | oparte na SD 1.5 (CreativeML OpenRAIL-M) | ⚠️ — generatywne (zmienia detale), nie dla retuszu 1:1; można jako „wzorzec AI” + nasz transfer gradingu |
| [baegwangbin/DSINE](https://github.com/baegwangbin/DSINE) | 934 | 2024-07-10 | **akademicka (Imperial)** | — | 📚 |
| [apple-aiml-research/ml-depth-pro](https://github.com/apple-aiml-research/ml-depth-pro) | 5 806 | 2026-09-11 | **Apple licence** | wagi: osobna licencja | 📚 |
| [gojasper/LBM](https://github.com/gojasper/LBM) | 870 | 2025-07-24 | NOASSERTION | relighting NC | 📚 |

**Ciekawy wniosek:** najlepszy praktyczny relighting dla Retusza to **połączenie istniejących elementów** — generatywny relight (IC-Light lub dowolne AI, także zewnętrzne), a potem nasz **transfer gradingu z AI** (BGU) na oryginał. Dzięki temu zachowujemy 100% detalu oryginału. To już jest unikalna przewaga; brakuje tylko lepszego matte (BiRefNet/MODNet) do maski światła.

### 5.6 Inferencja AI
| Projekt | Status | Licencja | Uwagi |
|---|---|---|---|
| ONNX Runtime ([microsoft/onnxruntime](https://github.com/microsoft/onnxruntime)) | aktywny (v1.31.x; PyPI `onnxruntime` 1.32.0) | MIT | API GitHuba zwróciło nam 404 dla repo (prawdopodobnie przejściowo); PyPI i dokumentacja potwierdzają projekt |
| DirectML EP (`onnxruntime-directml` 1.24.4) | **sustained engineering / maintenance** | MIT | działa na Win10, ale bez nowych funkcji; opset do 20 |
| Windows ML | zalecane przez Microsoft dla nowych projektów | — | NPU/GPU EP tylko Win11 24H2+; na Win10 i tak DirectML |
| [pykeio/ort](https://github.com/pykeio/ort) | 2 628★, 2026-11-26 | Apache-2.0 (+MIT) | ONNX Runtime w Rust (DirectML/CUDA/CoreML/NNAPI) — klucz do jednego front-endu Rust |
| [huggingface/candle](https://github.com/huggingface/candle) | 21 425★ | Apache-2.0/MIT | czysty Rust; alternatywa dla prostych modeli |
| [Tencent/ncnn](https://github.com/Tencent/ncnn) | 24 102★ | BSD-3 (+ komponenty) | Vulkan, świetny na Androidzie (Real-ESRGAN-ncnn) |

### 5.7 UI / UX — biblioteki i aplikacje wzorcowe
| Repo | ★ | Push | Licencja | Rola |
|---|---|---|---|---|
| [emilk/egui](https://github.com/emilk/egui) | 32 511 | 2026-11-26 | MIT/Apache-2.0 | **już używamy** w `retusz-app` (0.30 — warto podnieść wersję) |
| [Adanos020/egui_dock → anhosh/egui_dock](https://github.com/anhosh/egui_dock) | 650 | 2026-11-20 | MIT | dokowane panele (Biblioteka / Edycja / Porównanie) |
| [rerun-io/egui_crates](https://github.com/rerun-io/egui_crates) (dawniej egui_tiles) | 605 | 2026-11-26 | Apache-2.0 | kafelkowe układy |
| [woelper/oculante](https://github.com/woelper/oculante) | 1 743 | 2026-11-23 | MIT | **przeglądarka/edytor w egui** — wzorzec szybkiego podglądu i UX w tym samym stosie |
| [GraphiteEditor/Graphite](https://github.com/GraphiteEditor/Graphite) | 28 046 | 2026-11-26 | Apache-2.0 | Rust, nieniszcząca edycja węzłowa, nowoczesne UX — wzorzec modelu dokumentu i historii |
| [iced-rs/iced](https://github.com/iced-rs/iced) | 32 139 | 2026-11-19 | MIT | alternatywa (retained), nie warto zmieniać, skoro mamy egui |
| [slint-ui/slint](https://github.com/slint-ui/slint) | 24 737 | 2026-11-26 | GPL-3 / Royalty-free / komercyjna | ⚠️ royalty-free wymaga atrybucji, warunki — nie potrzebne |
| [tauri-apps/tauri](https://github.com/tauri-apps/tauri) | 113 197 | 2026-11-26 | Apache-2.0/MIT | web UI + Rust; podgląd GPU trudniejszy (WebGPU w WebView2) |
| [ocornut/imgui](https://github.com/ocornut/imgui) | 77 262 | 2026-11-24 | MIT | C++, nie dla nas |
| [hoffstadt/DearPyGui](https://github.com/hoffstadt/DearPyGui) | 15 670 | 2026-11-14 | MIT | jeśli zostajemy w Pythonie: GPU-renderowany UI, szybkie tekstury |
| PySide6 / Qt ([qtproject/pyside-pyside-setup](https://github.com/qtproject/pyside-pyside-setup)) | 120 (mirror) | 2026-11-26 | LGPL-3 | dojrzały, ale LGPL (dynamiczne linkowanie, możliwość podmiany) i duży rozmiar |
| [flet-dev/flet](https://github.com/flet-dev/flet), [flutter/flutter](https://github.com/flutter/flutter) | 17 591 / 181 522 | 2026-11-26 | Apache / BSD-3 | wspólny UI Windows+Android — ale słaby dla podglądu 16-bit GPU |
| [scaleflex/filerobot-image-editor](https://github.com/scaleflex/filerobot-image-editor) | 1 936 | 2026-06-16 | MIT | wzorce prostego UX narzędzi (web) |
| [CyberTimon/RapidRAW](https://github.com/CyberTimon/RapidRAW) | 11 271 | 2026-11-24 | **AGPL-3.0** | 📚 bardzo dobry wzorzec UX „Lightroom-like, GPU, nieniszczący” — tylko oglądać |
| [KDE/krita](https://github.com/KDE/krita), [GNOME/gimp](https://github.com/GNOME/gimp), darktable, RawTherapee, ART, digiKam | — | — | GPL | 📚 wzorce: Krita (pędzle, popup palette, ciśnienie pióra), darktable (maski parametryczne + rysowane, historia), ART (prostota RT), digiKam (DAM) |

---

## 6. Top 10 priorytetów (kolejność, wpływ, wysiłek)

| # | Co | Wpływ na jakość / UX | Wysiłek | Licencyjnie | Dlaczego teraz |
|---|---|---|---|---|---|
| 1 | **Semantyczna maska osoby / skóry / włosów / ubrania / akcesoriów** (MediaPipe Multiclass + Face Landmarker 478 → guided/BGU upsampling) zamiast heurystyki i bariery gradientowej | ★★★★★ (D&B, wyrównanie skóry, De-Shine, Plastyka, HSL — wszystko zależy od maski) | Ś (1–2 tyg.) | ✅ Apache | naprawia problemy z okularami/zarostem z 0.8.0 u źródła |
| 2 | **Matte sylwetki** (MODNet szybki, BiRefNet „jakość”) do Światła 3D i Separacji tła | ★★★★ (koniec z halo wokół ludzi — zgłaszane przez użytkownika) | Ś | ✅ MIT/Apache | halo przy D&B/Plastyce było głównym zarzutem |
| 3 | **Pędzel „Usuń / Popraw” + Auto-Heal v2**: LaMa/MI-GAN na niskich częstotliwościach + przeniesienie tekstury porów (frequency separation); klon/healing ze źródłem | ★★★★★ (podstawowa funkcja retuszera; Retouch4me Heal / Evoto) | Ś–W | ✅ | największa luka funkcjonalna |
| 4 | **Auto Ton v2** (cel zależny od ITA karnacji + mieszanie scena/twarz + intencja high/low-key + limity) i **Auto WB v2** (FFCC/gray-world z maską skóry wykluczoną) | ★★★★ | M–Ś | ✅ | 0.8.0 ma cel L\*67 — ryzyko dla ciemnej skóry |
| 5 | **Jeden front-end: Rust + egui + wgpu** (podgląd GPU 60 fps na 24 Mpx, mipmapy, kafelki), Python tylko jako backend AI/batch w okresie przejściowym → potem `ort` w Rust | ★★★★★ (UX) | W (4–8 tyg.) | ✅ | dwa edytory = podwójny koszt; Tk nie da płynności |
| 6 | **Panel Masek + lokalne korekty** (gradient liniowy/radialny, pędzel z auto-maską krawędzi, maski AI z p.1, add/subtract/intersect, overlay) | ★★★★★ | W | ✅ | standard LR/C1; bez tego „pro” to tylko globalne suwaki |
| 7 | **Historia + Migawki + Wirtualne kopie + Kopiuj/Wklej selektywnie** (model poleceń w `*.retusz.json`) | ★★★★ (UX) | Ś | ✅ | stopka 0.8.0 tego wymaga |
| 8 | **Odszumianie AI** (SCUNet/NAFNet, opcjonalnie RAW-owe) + **SR ×2** (Swin2SR/Real-ESRGAN, z ochroną skóry maską) | ★★★★ | Ś | ✅ (adnotacje datasetów) | Lightroom Denoise to dziś standard; SR poprawi małe wersje z AI |
| 9 | **Pełny Color Grading (3 koła + balans) i HSL 8×3 w OKLCh, krzywa tonalna (parametryczna + punktowa + RGB)** | ★★★★ | Ś | ✅ | 0.8.0 robi tylko 2 kanały HSL |
| 10 | **Korekcja obiektywów (Lensfun, dynamicznie) + XMP sidecar + import LUT .cube** | ★★★ | Ś | ⚠️ LGPL/CC BY-SA — tylko dynamicznie + atrybucja | kompletność workflow pro |

Stałe zadanie w tle: **bramki jakości** rozszerzone o nowe moduły (IoU masek na syntetycznych i publicznych zdjęciach CC0, halo na krawędziach matte, ΔE00 skóry przed/po, LPIPS/SSIM dla odszumiania) i **testy na zróżnicowanych karnacjach** (publiczne zbiory na licencjach CC0/CC BY — nie zdjęcia klientów).

---

## 7. Rekomendacje UX/UI (konkretne)

1. **Architektura UI:** jeden front-end Rust (egui ≥ najnowsza, `egui_dock`), renderowanie podglądu w wgpu:
   - piramida mipmap i kafelki 512 px, podgląd interaktywny 1/4 rozdzielczości podczas przeciągania suwaka, potem doostrzenie w tle;
   - ten sam kod WGSL do podglądu i eksportu (zachowujemy regułę ΔE < 1).
2. **Moduły:** Biblioteka (G) · Edycja (D) · Porównaj (C/N) · Grading z AI · Eksport. Górny pasek minimalny (zgodnie z 0.8.0).
3. **Prawy panel (jedna kolumna):** Histogram (interaktywny) → pasek narzędzi [Kadr R] [Usuń Q] [Maski M] [Pędzel K] [Pipeta W] → Profil/Preset (siła) → Podstawowe → Krzywa → HSL/Kolor → Color Grading → Skóra (Wzorzec, Odcień, Melanina/Hemoglobina, De-Shine, Heal) → Światło 3D → Detal → Optyka → Kalibracja (ColorChecker). Panele zwijane, tryb solo, oczko on/off.
4. **Maski:** lista po lewej stronie maski (jak LR), kolorowa nakładka (O), suwaki korekt per maska. Maski AI generowane raz i cache'owane w sidecarze (hash pliku).
5. **Pędzel:**
   - rozmiar `[`/`]`, twardość Shift+`[`/`]`, przepływ, Alt = odejmij;
   - **Auto Mask** (krawędzie na guided filter), wsparcie pióra (ciśnienie) przez `winit`;
   - podgląd okręgu z wewnętrzną strefą twardości.
6. **Porównania:**
   - `\` przed/po, `Y` obok siebie z synchronizacją zoomu, suwak podziału przeciągany myszą;
   - „Z wzorcem” w 3 kolumnach (już jest);
   - **Miganie** (jest w Grading z AI — przenieść do głównego).
7. **Wydajność odczuwalna:**
   - natychmiastowa reakcja suwaka (< 16 ms na podglądzie);
   - ciężkie AI (maski, inpainting) w tle z paskiem postępu na miniaturze;
   - kolejka zadań batch z anulowaniem.
8. **Onboarding:** ściągawka skrótów (`?`), krótkie podpowiedzi przy pierwszym użyciu narzędzia (zamiast stałych podpisów), spójne polskie nazwy, liczby z jednostkami (EV, K, °).
9. **Dostępność:** kontrast UI zgodny z WCAG AA na ciemnym motywie, neutralna szarość tła (18%), żeby nie fałszować oceny koloru; skalowanie DPI 100–200%.
10. **Selekcja serii:** flagi P/X/U, gwiazdki 0–5 (0.8.0), kolorowe etykiety 6–9, filtr w Bibliotece, Survey (N).

---

## 8. Ryzyka licencyjne — podsumowanie

| Kategoria | Przykłady | Co robić |
|---|---|---|
| **Bezpieczne (kod + wagi permisywne)** | MediaPipe Multiclass (Apache), MODNet, BiRefNet (oficjalne), MI-GAN, SAM 2, U-2-Net, Real-ESRGAN, SwinIR, HAT, Swin2SR, SCUNet, colour-science, OpenColorIO, Little-CMS, egui, ort, Depth Anything V2 **Small**, YuNet, ModFlows | dopisać do `THIRD_PARTY_LICENSES.md` z SHA-256 plików wag i linkiem do model card |
| **Kod OK, dataset badawczy** | LaMa (Places), NAFNet/Restormer (SIDD/GoPro), face-alignment, PIPNet, SCHP, RetinaFace (WIDER FACE), 3D-LUT/HDRNet (FiveK), GFPGAN (FFHQ NC) | decyzja właściciela; preferować modele trenowane na danych syntetycznych lub własnych; dokumentować |
| **LGPL / CC BY-SA** | LibRaw (CDDL ok), Lensfun (LGPL-3 + baza CC BY-SA 3.0), dnglab/rawspeed (LGPL-2.1), libvips, PySide6 | tylko niezmienione biblioteki dynamiczne lub osobny program; atrybucja bazy Lensfun |
| **Tylko nauka (nie kopiować kodu/wag)** | darktable, RawTherapee, ART, ansel, Krita, GIMP, digiKam, RapidRAW (AGPL), upscayl (AGPL), chaiNNer, RobustVideoMatting, spektrafilm, OpenDRT (moduł vkdt), vkdt `filmsim` | implementować z artykułów; w komentarzu cytować artykuł, nie repo GPL |
| **Zakaz (NC / brak licencji / własnościowe)** | CodeFormer, InsightFace wagi, Sapiens, DSINE, Deep White Balance, Depth Pro, BRIA RMBG, Depth Anything V2 Base/Large, MatAnyone, LBM, parsery CelebAMask-HQ, AgX (brak licencji), Zero-DCE, CURL (brak licencji), Retouch4me / Evoto / PortraitPro / Luminar | nie używać; Retouch4me i inne tylko jako czarna skrzynka do porównań wyników |

Dodatkowe uwagi:
- **vkdt** ma licencję BSD-2 dla całości, ale moduły mają własne zastrzeżenia (np. `OpenDRT`: „this module is licenced GPLv3”, `filmsim` to implementacja GPL-owego spektrafilm). Przed wzięciem czegokolwiek z vkdt trzeba sprawdzić `readme.md` danego modułu.
- **Generatywne modele SD (IC-Light, Marigold)**: licencja CreativeML OpenRAIL-M ma ograniczenia użycia (use-based restrictions) i duże wagi. Nie wbudowywać, najwyżej jako opcjonalny zewnętrzny krok.
- **Prywatność:** wszystkie modele działają lokalnie, bez wysyłania zdjęć klientów. To argument sprzedażowy i wymóg (brak zgody klientów).

---

## 9. Proponowana roadmapa 0.9 → 1.0

**0.8.x (domknięcie tego, co jest w budowie)**
- Wypchnąć 0.8.0 na gałąź `feature/v0.8.0-lr-classic`, przepuścić przez bramki i zaktualizować `AGENTS.md` (stos Rust + Python, wersja 0.8).
- Poprawki do specyfikacji: wspólna skala L (CIELAB vs OKLab), cel Auto Ton zależny od ITA z limitem EV, YuNet dwustopniowo (0,45 kandydaci → weryfikacja) i bez podwójnego NMS.

**0.9 — „Maski i czysta skóra” (jakość)**
1. MediaPipe Multiclass + Face Landmarker → nowy `skin_mask` v3 (face-skin / body-skin / hair / clothes / accessories) z guided upsamplingiem; testy IoU na syntetycznych i publicznych (CC0) zdjęciach.
2. MODNet / BiRefNet → matte postaci do Światła 3D, Separacji tła i D&B (koniec halo); metryka halo w bramce.
3. Pędzel „Usuń” (LaMa / MI-GAN + przeniesienie porów) i Auto-Heal v2 oparty na masce skóry v3.
4. Auto Ton v2 / Auto WB v2.
5. HSL 8×3 + Color Grading 3 koła + krzywa tonalna.

**0.10 — „Jeden edytor GPU” (UX)**
1. Front-end Rust + egui + wgpu jako jedyny edytor; Python tylko jako serwis AI/batch (proces potomny, JSON-RPC), potem stopniowo `ort` w Rust.
2. Panel Masek, gradienty liniowe/radialne, pędzel z Auto Mask.
3. Historia + Migawki + Wirtualne kopie + Kopiuj/Wklej selektywnie; skróty LR; filmstrip; flagi P/X.

**0.11 — „Detal pro”**
1. Odszumianie AI (SCUNet / NAFNet) z podglądem w lupie 100%; SR ×2 (Swin2SR — szczególnie dla wersji z AI w Grading z AI).
2. Lensfun (dynamicznie) + profile; import LUT .cube (OpenColorIO / własny parser); XMP sidecar.
3. Soft-proofing (Little-CMS) i eksport z wyostrzaniem pod medium.

**1.0 — „Stabilność i sprzedaż”**
- Pełne bramki jakości (zróżnicowane karnacje, grupy, okulary, zarost, ciemne tła).
- Wydajność: 24 Mpx — suwak < 16 ms na podglądzie, eksport < 3 s na GPU średniej klasy.
- Kompletne `THIRD_PARTY_LICENSES.md` z SHA-256 wag; instalator; auto-aktualizacja modeli z weryfikacją SHA.
- Dokumentacja użytkownika PL, ściągawka skrótów.
- Przygotowanie Androida: rdzeń Rust (`retusz-core`) przez JNI/uniffi, UI Kotlin/Compose, modele: MI-GAN, MediaPipe (natywnie), Depth Anything Small, ncnn lub ORT-NNAPI.

---

## 10. Źródła i metoda
- Repo: https://github.com/Jack4B/retusz (stan rozdz. 1–10: main @ `3602faa`, tagi v0.1.1–v0.7.0, gałęzie `feature/dnb-v2` = v0.1.6, `fix/ai-alignment-robust` = v0.1.5; w chwili pierwszego zapisu nie było gałęzi/PR 0.8.0). **Uzupełnienie tego samego dnia (Europe/Warsaw):** na `origin/main` pojawił się merge `27bb9a3` (`feature/v0.8.0-lightroom-classic-1col-hsl`). Gałęzi `docs/research-2026-11` nie scalano z `main`.
- Dane repozytoriów: GitHub REST API (`gh api repos/<owner>/<repo>`), pola `stargazers_count`, `pushed_at`, `license.spdx_id`. Dla `NOASSERTION` lub braku licencji odczytano plik LICENSE/README (fragmenty w rozdz. 5 i 8). Surowe dane: `repos.tsv`, `repos2.tsv` obok tego pliku (kopia robocza na boxie).
- Model card MediaPipe Multiclass Segmentation (Apache-2.0): https://storage.googleapis.com/mediapipe-assets/Model%20Card%20Multiclass%20Segmentation.pdf; dokumentacja: https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter
- DirectML maintenance mode: https://github.com/microsoft/DirectML (README), https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html; Windows ML: https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview
- Funkcje programów komercyjnych: wyłącznie publiczne strony produktów i materiały marketingowe (bez inżynierii wstecznej).
- ITA°: Chardon, Cretois, Hourseau, „Skin colour typology and suntanning pathways”, Int. J. Cosmetic Science 1991.

---

## 11. Aplikacja darmowa — co dodatkowo wolno użyć

**Uzupełnienie z 2026-11-26 (Europe/Warsaw), po decyzji Jack4B, że Retusz będzie darmowy.** To nie jest porada prawna — przed wbudowaniem modelu NC lub zmianą licencji repo warto skonsultować konkretny tekst licencji (albo uzyskać pisemną zgodę autora).

Obecna licencja samego Retusza jest **niespójna** i trzeba ją uporządkować niezależnie od reszty tego rozdziału:
- `Cargo.toml` (workspace Rust): `license = "MIT"`;
- `pyproject.toml` (Python): `Proprietary (Jack4B)`.

Dopóki `pyproject.toml` mówi „Proprietary”, formalnie to nie jest jeszcze projekt open-source, nawet jeśli program nic nie kosztuje. Do darmowego wydania trzeba wybrać jedną licencję kodu (rekomendacja: **MIT**, spójnie z Rustem i z dotychczasowym `THIRD_PARTY_LICENSES.md`) i zapisać ją w obu manifestach oraz w pliku `LICENSE`.

### 11.1 Najważniejsze rozróżnienie: darmowa aplikacja ≠ użycie niekomercyjne

Większość licencji „Non-Commercial” (CC BY-NC, S-Lab 1.0, „non-commercial research only”) **nie otwiera się automatycznie** tylko dlatego, że program jest za darmo.

| Pytanie | Znaczenie dla Retusza |
|---|---|
| Czy program jest płatny? | Nie — to spełnia jeden warunek wielu licencji NC (brak „monetary compensation” za sam software). |
| Czy fotograf używa programu do retuszu zdjęć klientów (praca płatna)? | **Tak, to jest typowy scenariusz Retusza.** Creative Commons definiuje NonCommercial jako użycie „not primarily intended for or directed towards commercial advantage or monetary compensation”. Retusz zdjęć w ramach działalności zarobkowej jest zwykle **użyciem komercyjnym modelu**, nawet gdy aplikacja nic nie kosztuje. |
| Czy „research purposes only” obejmuje darmowy edytor dla fotografów? | **Nie.** To węższe niż „niekomercyjne”: typowo tylko badania i edukacja, nie produkt dla użytkowników końcowych. |
| Czy wolno rozprowadzać model w instalatorze? | Tylko jeśli licencja zezwala na *redistribution* / *Share*. Część umów (DSINE, BRIA RMBG-1.4) wprost tego zabrania. |
| Darowizny, reklamy, płatny Android, płatne presety, „pro pack”? | Prawie na pewno robią z wydania użytek komercyjny i zamykają drogę modelom NC. |

**Wniosek praktyczny:** linia wydania publicznego Retusza zostaje na modelach **MIT / Apache-2.0 / BSD** (rozdz. 5 i 8). Modele NC rozważamy wyłącznie:
- (A) po pisemnej licencji od autora, albo
- (B) w osobnej kompilacji „tylko do użytku prywatnego / badawczego”, z regulaminem zakazującym użytku zawodowego i **bez** dołączania wag do instalatora publicznego (użytkownik sam je pobiera po akceptacji licencji) — słabe dopasowanie do edytora portretowego, bo Retusz jest narzędziem zawodowym,
- (C) nigdy, gdy licencja mówi „research only”, „internal only”, „no transfer / no distribution”.

Poniższe oceny **Tak / Warunkowo / Nie** dotyczą właśnie publicznego, darmowego wydania używanego też przez fotografów przy zleceniach. „Tak” oznacza: wolno wbudować przy typowych warunkach atrybucji. „Warunkowo” = da się bronić tylko w wąskim wariancie (hobby-only + kliknięcie licencji, albo wagi poza instalatorem). „Nie” = nie wbudowywać.

### 11.2 Modele i kod z licencją niekomercyjną — karta po karcie

Gwiazdki i daty jak w rozdz. 5, chyba że podano inaczej (stan API 2026-11-26).

| Projekt | Licencja (dokładnie) | Warunki, których nie wolno pominąć | Co daje / stos / rozmiar | Werdykt |
|---|---|---|---|---|
| [sczhou/CodeFormer](https://github.com/sczhou/CodeFormer) (18 265★, 2025-11-18) | **S-Lab License 1.0**: „Redistribution and use for **non-commercial purpose** in source and binary forms… permitted”. Użycie komercyjne wymaga kontaktu z autorami. | Atrybucja przy redystrybucji kodu i binariów. Brak definicji „commercial” tak precyzyjnej jak w CC — bezpiecznie przyjąć, że praca zarobkowa fotografa i każda monetyzacja są poza zakresem. | Restauracja / „naprawa” twarzy (codebook transformer). Ciężki (PyTorch/ONNX, setki MB). W retuszu portretowym często **zmienia rysy** — słabe dopasowanie nawet licencyjnie. | **Nie** (wydanie publiczne dla fotografów). Ewentualnie osobna kompilacja hobby po akceptacji S-Lab, nie w instalatorze głównym. |
| [deepinsight/insightface](https://github.com/deepinsight/insightface) (30 309★) | Kod: **MIT** (także komercyjnie). Wagi z repo i z pakietu PyPI: **„non-commercial research purposes only”** (README). Od 2025-11-24 osobne licencje komercyjne na inswapper, buffalo_l i SDK (kontakt: insightface.ai). | **Badania**, nie darmowy produkt. Sam brak opłaty za aplikację nie wystarcza. | Detekcja SCRFD, punkty, rozpoznawanie. Nie potrzebujemy rozpoznawania tożsamości (i model card / etyka tego nie lubi). | **Nie** (wagi). Kod MIT można czytać. Do detekcji: YuNet (już mamy) albo MediaPipe (Apache). |
| [briaai/RMBG-1.4](https://huggingface.co/briaai/RMBG-1.4) | Własna umowa **bria-rmbg-1.4** / [BRIA Hugging Face Model License](https://bria.ai/bria-huggingface-model-license-agreement/). README mówi o CC dla użytku niekomercyjnego, ale tekst LICENSE na Hugging Face jest **węższy**: licencja **czasowa, osobista, bez sublicencji**, „for the sole purpose of **evaluating and examining**”; zakaz **sprzedaży, użyczania, dystrybucji i hostowania** modelu dla innych; zakaz łącznia z OSS, które narzuci inną licencję. | **Nie wolno dołączyć do instalatora ani rozprowadzać użytkownikom.** Wymaga akceptacji umowy (18+). Użycie komercyjne = osobna umowa + często agent atrybucji BRIA. | Usuwanie tła, bardzo dobre na włosach. | **Nie** (nawet przy darmowej aplikacji). Zostaje BiRefNet / MODNet. |
| [Bria-AI/RMBG-2.0](https://github.com/Bria-AI/RMBG-2.0) / RMBG-3.x | README i `LICENSE`: **CC BY-NC 4.0**. Użycie komercyjne = umowa z BRIA. | Atrybucja (BY). Share dozwolony tylko na cele niekomercyjne. Użycie przez fotografa przy zleceniach — jak w 11.1, zwykle poza NC. Część wag BiRefNet trenowanych na danych BRIA jest w README BiRefNet oznaczona jako NC i nieświadoma przezroczystości (rozdz. 5.1). | Lepsze matte niż 1.4. Ciężki. | **Nie** do wydania głównego. Oficjalny BiRefNet (MIT) wystarcza. |
| [DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) Base / Large / Giant | README: **CC-BY-NC-4.0**. Wariant **Small pozostaje Apache-2.0** (już w aplikacji). | Atrybucja. NC jak wyżej. Small nie wymaga tej dyskusji. | Lepsza mapa głębi (krawędzie włosów, grupy). Modele znacznie większe i wolniejsze niż Small (~90–335 MB+). | **Warunkowo** tylko w kompilacji hobby-only, z atrybucją, poza głównym instalatorem. Wydanie główne: zostajemy przy **Small (Apache)**. |
| [mahmoudnafifi/Deep_White_Balance](https://github.com/mahmoudnafifi/Deep_White_Balance) (629★, `LICENSE.md`) | **CC BY-NC-SA 4.0**. | **ShareAlike:** adaptacje (w tym model wbudowany w większy program, jeśli powstaje utwór zależny) trzeba publikować na CC BY-NC-SA. To **zderza się** z MIT/Proprietary całego Retusza: albo cały program (lub co najmniej warstwa WB) idzie na BY-NC-SA, albo wagi zostają poza kodem jako osobny, jasno oznaczony składnik i i tak tylko do użytku NC. | Uczony auto-WB, w tym mieszane oświetlenie. | **Nie** (ShareAlike + NC + użycie zawodowe). Alternatywy permisywne: `google/ffcc` (Apache), własne Auto WB v2 (rozdz. 3.3). |
| [facebookresearch/sapiens](https://github.com/facebookresearch/sapiens) (5 442★) | **CC BY-NC 4.0** (plik LICENSE). | Atrybucja, tylko NC. Wagi duże (setki MB–GB). | Segmentacja ciała, normalne, głębia — jakość referencyjna. | **Nie** do wydania głównego (NC + rozmiar). Jakość jako punkt odniesienia do testów wewnętrznych, bez dystrybucji wag. |
| Parsery twarzy uczone na CelebAMask-HQ: [zllrunning/face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch), [yakhyo/face-parsing](https://github.com/yakhyo/face-parsing), [FacePerceiver/facer](https://github.com/FacePerceiver/facer); zbiór: [switchablenorms/CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) (2 333★) | Kod parserów często **MIT**. Zbiór (strona projektu i README): *„available for **non-commercial research purposes** only”*; zakaz komercyjnego wykorzystania obrazów i **„any portion of derived data”**; **zakaz dalszego kopiowania, publikowania i dystrybucji** zbioru (kopie tylko wewnątrz jednej organizacji). | Wagi wytrenowane na tym zbiorze są poważnym kandydatem na „derived data”. Nawet darmowa aplikacja rozprowadzająca te wagi **wykracza poza „research only” i poza zakaz dystrybucji**. Kod MIT bez wag jest w porządku. | 19 klas: skóra, brwi, oczy, usta, włosy, okulary, biżuteria — teoretycznie lepsze niż 6 klas MediaPipe. | **Nie** (wagi w wydaniu publicznym). Do masek: **MediaPipe Selfie Multiclass + Face Landmarker (Apache-2.0)**. Parsery z CelebAMask-HQ tylko do wewnętrznych porównań IoU, bez publikacji wag. |
| [baegwangbin/DSINE](https://github.com/baegwangbin/DSINE) (934★) | Własna umowa Imperial College London. Użycie **„solely for non-commercial, internal or academic research”**. Wprost zakaz integracji w produkt na sprzedaż **lub licencję** dla osób trzecich oraz świadczenia usługi za opłatą. Licencja **osobista, bez transferu i sublicencji**; „You shall not permit any third party to access… the Software”. Kopie zapasowe tylko u licencjobiorcy. | **Nie wolno rozprowadzać użytkownikom końcowym** — nawet za darmo. Komercja wymaga e-maila `researchcontracts.engineering@imperial.ac.uk`. | Świetne normalne powierzchni (CVPR 2024 Oral), przydatne do Światła 3D. | **Nie**. |
| [apple-aiml-research/ml-depth-pro](https://github.com/apple-aiml-research/ml-depth-pro) (5 806★) | Własna licencja Apple (plik LICENSE). README: **kod i wagi na tych samych warunkach**. Apple udziela **osobistej, niewyłącznej** licencji na używanie, kopiowanie, modyfikację i redystrybucję; przy redystrybucji **całości bez zmian** trzeba zachować zawiadomienie. Brak licencji patentowej. Zakaz używania znaków Apple do promocji. | To **nie jest licencja NC** — nie ma zakazu użytku komercyjnego w tekście. Ryzyko: słowo „personal”, brak patent grant, typowa licencja „Apple sample code”. Wbudowanie zmodyfikowanego modelu w produkt jest szare prawnie. Model duży (`depth_pro.pt`). | Metryczna głębia monokularowa, ostre krawędzie, < 1 s na GPU (deklaracja autorów). | **Warunkowo / raczej nie** w instalatorze. Wolno rozważyć redystrybucję **niezmienionego** pakietu z pełnym tekstem LICENSE Apple, albo zostawić jako opcjonalny plik pobierany przez użytkownika. Bezpieczniej zostać przy Depth Anything V2 Small. |
| [Li-Chongyi/Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE) (1 093★) | README: *„The code is made available for **academic research purpose only**. Under Attribution-NonCommercial 4.0 International License.”* Brak pliku LICENSE w korzeniu. | Podwójne ograniczenie: badania akademickie **oraz** CC BY-NC. Darmowy edytor nie jest „academic research”. | Krzywe do rozjaśniania zdjęć z ciemności (auto-ekspozycja). | **Nie**. Auto Ton v2 liczymy sami (percentyle + ITA, rozdz. 3.3). |
| [sjmoran/curl-image-enhancement](https://github.com/sjmoran/curl-image-enhancement) (223★, wcześniej w rozdz. 5 jako „brak licencji”) | **Korekta względem rozdz. 5:** README i `pyproject.toml` deklarują **BSD-3-Clause**. Osobnego pliku `LICENSE` w korzeniu **nie ma** (API 2026-11-26). Trening typowo na Adobe FiveK (zbiór badawczy). | Deklaracja BSD-3 w manifeście to zwykle ważna licencja kodu, mimo braku pliku LICENSE. Wagi z FiveK — jak inne modele FiveK, ryzyko zbioru; lepiej wytrenować własne krzywe na zgodnych zdjęciach. | Warstwy krzywych neuronowych do auto-enhancement (Auto Ton). | **Warunkowo (kod Tak, wagi ostrożnie).** Można czytać / portować kod jako BSD-3; w wydaniu nie dołączać cudzych wag z FiveK bez sprawdzenia zbioru. |
| [TencentARC/GFPGAN](https://github.com/TencentARC/GFPGAN) (37 737★) | Kod: **Apache-2.0** (z zastrzeżeniem komponentów stron trzecich w LICENSE). Trening na [FFHQ](https://github.com/NVlabs/ffhq-dataset) (NVlabs: obrazy z Flickra na mieszance CC BY / BY-NC / PD; *„All of these licenses allow free use… for **non-commercial purposes**”*). | Kod wolno używać komercyjnie. **Wagi przedtrenowane** są szare: część danych źródłowych jest BY-NC, a dataset jako całość jest przedstawiany jako niekomercyjny. NVIDIA kieruje zapytania biznesowe do swojego licencjonowania. Dodatkowo: restauracja GAN zmienia tożsamość — w portrecie studyjnym zwykle niechciana. | „Naprawa” twarzy z prioru generatywnego. | **Warunkowo / raczej nie.** Nie jako domyślny etap pipeline'u. Jeśli kiedyś, tylko jako opcjonalny filtr z ostrzeżeniem i bez obietnicy wierności. |
| [wzhouxiff/RestoreFormer](https://github.com/wzhouxiff/RestoreFormer) (367★) i [RestoreFormerPlusPlus](https://github.com/wzhouxiff/RestoreFormerPlusPlus) (293★) | Kod: **Apache-2.0**. README: trening **FFHQ** (to samo ryzyko wag co GFPGAN). | Jak GFPGAN. | Restauracja twarzy (HQ dictionary). | **Warunkowo / raczej nie** (FFHQ + zmiana rysów). |
| [yangxy/GPEN](https://github.com/yangxy/GPEN) (2 612★) | **Brak pliku licencji** w API GitHuba. | Brak licencji = brak zgody na użycie. | Restauracja twarzy. | **Nie.** |
| [lllyasviel/IC-Light](https://github.com/lllyasviel/IC-Light) (8 587★) | Kod: **Apache-2.0**. Model oparty na Stable Diffusion 1.5 (**CreativeML OpenRAIL-M** — licencja z ograniczeniami *użycia*, nie jest typowym NC). Autor sam pisze, że w projektach komercyjnych trzeba **zastąpić BRIA RMBG 1.4** (np. BiRefNet). | OpenRAIL-M zezwala na darmowe i płatne użycie przy przestrzeganiu zakazów (m.in. nielegalne zastosowania, wprowadzanie w błąd). Jest **niekompatybilna z GPL** (dodatkowe ograniczenia użycia). Ciężki model dyfuzyjny (GB, wolny), potrafi zmienić detal — nie nadaje się jako domyślny relight 1:1. | Relighting generatywny. | **Warunkowo** jako **zewnętrzny, opcjonalny** krok (użytkownik sam odpala / wgrywa wynik), a potem nasz **transfer gradingu z AI** na oryginał. Nie wbudowywać SD 1.5 w instalator. BRIA wyciąć. |
| [pq-yang/MatAnyone](https://github.com/pq-yang/MatAnyone) (1 647★) | **S-Lab License 1.0** (jak CodeFormer). | NC, atrybucja, kontakt przy komercji. | Stabilne matowanie wideo. | **Nie** do wydania głównego (mamy MODNet/BiRefNet na MIT/Apache). |
| [gojasper/LBM](https://github.com/gojasper/LBM) (870★) | `LICENCE.txt` i README: **CC BY-NC 4.0** (badge w README bywał mylący, tekst to BY-NC). | Atrybucja, tylko NC. | Relighting / usuwanie obiektów / estymacja normalnych (ICCV 2025). | **Nie** do wydania głównego. |
| [mahmoudnafifi/Exposure_Correction](https://github.com/mahmoudnafifi/Exposure_Correction), [mixedillWB](https://github.com/mahmoudnafifi/mixedillWB) | API: **NOASSERTION** — brak jasnej licencji. | Bez licencji nie używamy. | Auto-ekspozycja, WB przy mieszanym świetle. | **Nie**, dopóki autor nie wskaże licencji. |

**Podsumowanie tabeli:** darmowość programu **prawie nic nie odblokowuje** z listy NC, bo Retusz jest narzędziem do pracy z portretami klientów, a część umów i tak zabrania samej dystrybucji. Jedyna istotna korekta vs rozdz. 5: **CURL** można traktować jako kod BSD-3 (bez cudzych wag FiveK). Depth Pro nie jest NC, ale licencja Apple jest niewygodna w produkcie. GFPGAN/RestoreFormer mają kod Apache, lecz wagi z FFHQ są szare i i tak nie są pożądane w pipeline'ie wiernego retuszu.

### 11.3 Opcja GPL — co dałoby relicencjonowanie repo na GPL-3 i dlaczego tego nie robić

**Stan dziś:** kod Jack4B może relicencjonować (jest jedynym autorem według manifestów). Zależności stron trzecich są głównie MIT / Apache-2.0 / BSD (kompatybilne z **GPL-3**, ale **Apache-2.0 nie jest kompatybilne z GPL-2**). LibRaw jest używany na CDDL-1.0 (to trzeba by potwierdzić przy każdej zmianie licencji). Inno Setup i bootloader PyInstallera mają własne wyjątki — zostają jako narzędzia budowania, nie jako kod linkowany na GPL.

**Co wolno by wtedy skopiować (dziś jest tylko inspiracją, rozdz. 5 i 8):**

| Projekt | Licencja | Co byłoby realnie cenne | Koszt / uwaga |
|---|---|---|---|
| [darktable-org/darktable](https://github.com/darktable-org/darktable) (13 507★) | GPL-3.0 | Filmic/sigmoid, diffuse-or-sharpen, maski parametryczne + rysowane, historia, korekcja CA, profile obiektywów, color balance RGB | Ogromna baza C; przenoszenie fragmentów wiąże **cały** program (i przyszłe wkładki) licencją GPL-3. |
| [RawTherapee/RawTherapee](https://github.com/RawTherapee/RawTherapee) (4 311★), [artraweditor/ART](https://github.com/artraweditor/ART) (529★) | GPL-3.0 (ART: ten sam rodowód, API czasem `NOASSERTION`, w praktyce fork RT) | Capture sharpening, wstążki wieloskalowe, narzędzia skóry/HSL, demosaic | j.w. |
| [CyberTimon/RapidRAW](https://github.com/CyberTimon/RapidRAW) (11 271★) | **AGPL-3.0** | Nowoczesny UX „Lightroom + GPU, nieniszczący” | AGPL jest **ostrzejsza** od GPL: objąłby też ewentualne udostępnienie edytora przez sieć. Wciąganie AGPL do rdzenia jest najgorszym wariantem. |
| [andreavolpato/spektrafilm](https://github.com/andreavolpato/spektrafilm) i moduł `filmsim` w [hanatos/vkdt](https://github.com/hanatos/vkdt) | GPL-3.0 (vkdt jako całość jest BSD-2, ale readme `filmsim` wskazuje implementację spektrafilm) | Spektralna emulacja filmu | Można by skopiować **tylko jeśli** cały Retusz jest na GPL-3; prościej zostawić własny Film-Simulator / LUT-y. |
| Moduł `OpenDRT` w vkdt | Readme modułu: **GPLv3**, na bazie [jedypod/open-display-transform](https://github.com/jedypod/open-display-transform) | Transformacja wyświetlania | Nie mieszać z resztą vkdt (BSD-2). Albo cały app GPL, albo implementacja z opisu, albo pominięcie. |
| [upscayl/upscayl](https://github.com/upscayl/upscayl) (AGPL-3.0), [chaiNNer-org/chaiNNer](https://github.com/chaiNNer-org/chaiNNer) (GPL-3.0) | AGPL / GPL | UX upscalera, graf węzłów | Nie potrzebujemy ich kodu — mamy Real-ESRGAN/SwinIR (BSD/Apache) i własny UI. |
| Krita, GIMP, digiKam | GPL | Pędzle, historia, DAM | Tylko pomysły UX, jak dotychczas. |

**Konsekwencje relicencjonowania całego Retusza na GPL-3:**
1. Każdy, kto dostanie `.exe`, ma prawo żądać **odpowiadającego kodu źródłowego** (w praktyce: repo publiczne + skrypt budowy). To jest do zaakceptowania przy darmowym projekcie, ale jest zobowiązaniem na stałe.
2. Wszelkie przyszłe poprawki od innych osób (w tym Claude / PR-y) spadają na GPL-3; **powrót do MIT albo wersji zamkniętej** wymaga zgody wszystkich współautorów albo przepisania ich wkładu.
3. Nie można później sprzedać zamkniętej wersji „Pro” opartej na tym samym kodzie (chyba że równoległa licencja od początku, dual-license, i wszyscy kontrybutorzy się zgodzą).
4. Zależności Apache-2.0 (OpenCV, egui częściowo, OpenImageIO, SAM 2, wiele modeli) są **kompatybilne z GPL-3, nie z GPL-2**. Wybierać tylko GPL-3.
5. **Wagi NC i GPL nie idą w parze w jednym pakiecie.** GPL daje odbiorcy prawo używać programu **w dowolnym celu, także komercyjnym**. Licencja NC tego zabrania. Jeśli w instalatorze GPL są wagi NC, odbiorca dostaje sprzeczne prawa. Dopuszczalny (acz delikatny) wariant: program GPL, który *opcjonalnie* ładuje wagi pobrane osobno przez użytkownika, z jawnym komunikatem, że wagi mają **inną** licencję i nie są częścią utworu GPL. Tego i tak nie chcemy (rozdz. 11.2).
6. LGPL (Lensfun, libvips, dnglab) nadal wymaga dynamicznego linkowania albo udostępnienia obiektów do relinku — to jest niezależne od GPL rdzenia.

**Rekomendacja:** **nie relicencjonować Retusza na GPL-3 ani AGPL** tylko po to, by kopiować darktable / RawTherapee / RapidRAW / spektrafilm. Algorytmy implementujemy z artykułów (jak dotychczas), UI bierzemy wzrokiem, kod zostaje MIT (po uporządkowaniu `pyproject.toml`). Dzięki temu zostają otwarte drzwi do współpracowników, Androida i ewentualnego dual-licensing w przyszłości. Jeśli kiedyś naprawdę potrzebny będzie jeden moduł GPL, jedyna czysta droga to **osobny program / wtyczka w osobnym procesie** (użytkownik instaluje dobrowolnie), nie linkowany do `retusz-core`.

### 11.4 Zaktualizowana lista priorytetów (przy darmowej aplikacji)

Kolejność z rozdz. 6 **zostaje w mocy**, bo darmowość nie odblokowała bezpiecznie ani CodeFormera, ani BiSeNet z CelebAMask-HQ, ani BRIA, ani Depth Anything V2 Large. Korekty:

| # | Co | Zmiana vs rozdz. 6 |
|---|---|---|
| 1 | Semantyczne maski **MediaPipe Multiclass + Face Landmarker** (Apache) | **Bez zmian.** BiSeNet / facer nadal nie wchodzą do wydania publicznego (zbiór „research only” + zakaz dystrybucji derived data), mimo darmowej aplikacji. |
| 2 | Matte sylwetki **MODNet / oficjalny BiRefNet** | **Bez zmian.** BRIA RMBG-1.4 jest zakazany także w darmowym instalatorze (tylko ewaluacja, zakaz dystrybucji); RMBG-2.0 to NC. |
| 3 | Pędzel Usuń / Heal (LaMa, MI-GAN) | Bez zmian. |
| 4 | Auto Ton v2 / Auto WB v2 | Można wzorować się na kodzie **CURL (BSD-3)**; nie brać Deep White Balance (BY-NC-SA) ani Zero-DCE (academic + NC). |
| 5–7 | Jeden front-end Rust, panel masek, historia | Bez zmian. |
| 8 | Odszumianie AI + SR (SCUNet, NAFNet, Swin2SR, Real-ESRGAN) | **Bez CodeFormera / GFPGAN / RestoreFormer jako etapu domyślnego** (licencja NC albo wagi FFHQ + zmiana rysów twarzy). Ewentualny filtr „restauracja starych zdjęć” tylko jako opcja, z ostrzeżeniem. |
| 9 | HSL 8×3, Color Grading, krzywa | Bez zmian. |
| 10 | Lensfun dynamicznie, LUT, XMP | Bez zmian (LGPL/CC BY-SA bazy — dynamicznie + atrybucja). |
| + | Uporządkować licencję repo (**MIT** w `LICENSE` + `pyproject.toml` + `Cargo.toml`) i listę sprawdzającą poniżej | Nowe, powinno być **przed** dokładaniem kolejnych modeli. |
| + | Depth Anything V2 Base/Large i Sapiens | Nie w instalatorze głównym. |

### 11.5 Lista sprawdzająca zgodność licencji (przed każdym wydaniem)

1. **Jedna licencja kodu Retusza** w `LICENSE`, `pyproject.toml` i `Cargo.toml` (rekomendacja: MIT). Żadnego „Proprietary” przy darmowym wydaniu publicznym, chyba że świadomie zostawiamy zamknięty kod — wtedy też to trzeba napisać w jednym miejscu.
2. **`THIRD_PARTY_LICENSES.md` (albo katalog `LICENSES/`)** — każdy komponent i **każdy plik wag** osobno: nazwa, autor, URL, SPDX, czy wagi mają inną licencję niż kod, SHA-256, model card, ograniczenia (NC, research-only, OpenRAIL, CDDL, LGPL).
3. **Atrybucja w oknie „O programie”** oraz w instalatorze: lista modeli i bibliotek, linki do licencji, informacje wymagane przez BY (CC) i przez S-Lab / Apache (NOTICE).
4. **Żadnych wag NC, „research only”, BRIA 1.4, DSINE, InsightFace, CelebAMask-HQ w instalatorze publicznym.** Jeśli kiedyś powstanie kompilacja hobby, wagi tylko jako opcjonalne pobranie po **osobnym kliknięciu akceptacji licencji**, z komunikatem, że użycie zawodowe jest zabronione — i tak to jest wariant niezalecany.
5. **Nie mieszać pakietu GPL z wagami NC** (sprzeczne prawa odbiorcy). Przy MIT tego problemu nie ma, bo i tak nie relicencjonujemy na GPL.
6. **Lensfun / LibRaw / inne LGPL lub CDDL:** tylko niezmienione biblioteki, dynamicznie albo na warunkach już opisanych (LibRaw: CDDL); baza Lensfun: atrybucja CC BY-SA 3.0.
7. **Regulamin / README dla użytkownika:** program jest darmowy; zdjęcia liczy lokalnie (nic nie wychodzi do chmury); oryginały są tylko do odczytu; brak zgody klientów = brak wrzucania zdjęć do repo i do usług sieciowych.
8. **Bramka wydania:** skrypt albo pozycja w `quality_gate` / CI, która sprawdza, czy w paczce nie pojawił się nowy `.onnx` / `.tflite` / `.pth` bez wpisu w `THIRD_PARTY_LICENSES.md` i bez SHA-256.
9. **Bez dekompilacji i bez kodu/wag Retouch4me, Evoto, PortraitPro, Luminar** — bez zmian, także przy darmowej aplikacji.
10. Ta lista nie zastępuje lektury konkretnej licencji w dniu dodawania komponentu (teksty się zmieniają; zapisujemy wersję i datę).

