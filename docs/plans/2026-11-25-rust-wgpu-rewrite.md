# Retusz 0.2.0 – Plan: pełnoprawny edytor fotograficzny w Rust + wgpu (GPU) + egui, Presety, Dopasowanie do wzorca jako sekcja rozwijana i instalator Windows

> Autor: Jack4B. Pełna zgodność z zasadami `AGENTS.md`: wyłączny autor `Jack4B`, żadnych prywatnych zdjęć w repo, żadnych elementów Retouch4me, wyłącznie licencje MIT / BSD / Apache-2.0 / CDDL-1.0 (niemodyfikowane) oraz opublikowane artykuły naukowe, oryginały tylko do odczytu (`*.retusz.json`, `*.retusz-preset.json` i eksporty w `Retusz\`), UI po polsku ze stabilnym układem, podgląd = eksport (ΔE < 1).

---

## 1. Wizja produktu w `0.2.0`: Pełnoprawny edytor portretowy (Lightroom / Capture One style)

Dotychczas Retusz kręcił się wokół „Dopasowania do wzorca” (3 sztywne kolumny `Źródło | Wynik | Wzorzec`, a bez wybranego wzorca główny widok był w połowie pusty). W wersji **0.2.0** przebudowujemy aplikację w **pełnoprawny, samodzielny edytor fotograficzny**:

1. **Układ studyjny (Studio Layout):**
   - **Główne płótno na środku (80% ekranu):** domyślnie **Pojedynczy duży widok Wyniku** albo jednym kliknięciem **Obok siebie (`Przed | Po`)**, **Podział (`Przed | Po`)**, a gdy włączono wzorzec — także **`Wynik | Wzorzec`**.
   - **Prawy dokowany inspektor (jak w Capture One / Lightroom):**
     - **Histogram na żywo** na samej górze prawego panelu z trójkątami ostrzeżeń o przepaleniach/cieniach (`J`).
     - **Presety (Style):** rozwijana lista wbudowanych presetów portretowych (*Portret naturalny*, *Ciepły złoty*, *Filmowy miękki*, *High-Key*, *Czysty kontrast*, *B&W Klasyczny*) + własne presety użytkownika (`Zapisz preset…`, `Importuj…`, `Eksportuj…`, `Usuń`, pliki `*.retusz-preset.json` w `%APPDATA%\Retusz\presets\`).
     - **Sekcja 1 (otwarta): Światło i Ton (OKLab / EV)** — `Ekspozycja (EV)`, `Kontrast`, `Światła`, `Cienie`, `Biele`, `Czernie`.
     - **Sekcja 2 (otwarta): Kolor i Balans bieli (Bradford CAT + OKLab)** — `Temperatura`, `Odcień`, przycisk `Pipeta (W)`, `Balans → seria`, `Jaskrawość`, `Nasycenie`, `Pełne barwy`.
     - **Sekcja 3 (otwarta): Portret AI i Plastyka** — `Dodge&Burn` (v2, filtr prowadzony na L* + kontury twarzy z YuNet), `Plastyka 3D` (Depth Anything V2 Small + segmentacja postaci + normalne), `Kierunek światła 3D` (`Auto` + kąt), `Przejrzystość` (lokalny kontrast półtonów chroniący skórę), `Winieta`, podgląd `Maska skóry (M)`.
     - **Sekcja 4 (zbijana / otwierana na żądanie — BONUS): Dopasowanie do wzorca (AI Color Match)** — zwinięta domyślnie (z wyraźnym przełącznikiem / nagłówkiem `▸ Dopasowanie do wzorca (AI)`). Po rozwinięciu pokazuje miniaturę wzorca, przycisk `Wybierz wzorzec…`, `Pokaż wzorzec obok` oraz suwaki `Siła kolorów`, `Skóra`, `Równanie skóry`. Edytor działa w 100% samodzielnie bez wzorca!
     - **Sekcja 5 (narzędzia lokalne):** `Kadr i prostowanie (C)` (proporcje, kąt, linia horyzontu) oraz `Pędzel Dodge & Burn (B)` (Rozjaśnij/Przyciemnij, rozmiar `[` `]`, miękkość, przepływ, wyczyść).
   - **Dolny lub boczny pasek serii (Filmstrip):** miniatury wszystkich zdjęć w folderze z oznaczeniem edytowanych zdjęć (`●`) i szybką nawigacją (`←` / `→`).
   - **Górny pasek narzędzi:** `Otwórz zdjęcie`, `Otwórz folder`, przełącznik widoku (`Wynik` | `Przed | Po` | `Podział` | `Z wzorcem`), zoom (`Dopasuj`, `100%`, `200%`, `Przed/Po \`), `Cofnij` (`Ctrl+Z`), `Ponów` (`Ctrl+Y`), `Kopiuj` (`Ctrl+Shift+C`), `Wklej` (`Ctrl+Shift+V`), `Zeruj`, `Grading z AI`, `Ustawienia`, `Zapisz`, `Eksportuj serię`.

2. **Architektura Rust + `wgpu` (GPU) + `egui` (`eframe`):**
   - Natywny workspace Cargo (`crates/retusz-core`, `crates/retusz-gpu`, `crates/retusz-app`) kompilowany do pojedynczego, szybkiego pliku wykonywalnego `Retusz.exe` (+ `retusz-cli.exe` / subkomendy `cli` i `transfer`).
   - Równolegle przebudowane GUI w Pythonie (`retusz/gui/app.py`, `retusz/presets.py`) zachowuje 100% parytetu funkcji (w tym nowe **Presety**, nowy układ studyjny z prawym inspektorem i otwieraną sekcją **Dopasowanie do wzorca**), dzięki czemu zarówno natywny Rust (`cargo test`, `Retusz.exe`), jak i cały zestaw `pytest` + bramki jakości działają bez przerw.

3. **Instalator Windows (`Retusz-Setup-x64.exe` + `Retusz-windows-x64.zip`):**
   - Skrypt **Inno Setup (`installer/retusz.iss`)** budowany automatycznie w GitHub Actions (`.github/workflows/build-windows.yml`) obok paczki portable `.zip`:
     - instalacja w `%LOCALAPPDATA%\Programs\Retusz` (nie wymaga uprawnień administratora) lub `Program Files`,
     - skróty w Menu Start i na Pulpicie z ikoną `retusz.ico`,
     - powiązanie plików `*.retusz-preset.json` i opcjonalne „Otwórz w Retusz” w menu kontekstowym folderów/zdjęć,
     - czysty deinstalator (zostawia zdjęcia i presety użytkownika).

---

## 2. Struktura Cargo Workspace i modułów

```text
Cargo.toml                     # Workspace: crates/retusz-core, crates/retusz-gpu, crates/retusz-app
crates/
  retusz-core/                 # Silnik algorytmów w Ruście (OKLab, Bradford, LUT 33^3 tetraedryczny, D&B v2, Plastyka 3D, BGU, align, transfer, presety, sidecary, I/O)
    src/
      lib.rs
      params.rs                # Params, TransferParams, Preset (wbudowane + własne *.retusz-preset.json), Sidecar *.retusz.json, Settings
      color.rs                 # sRGB <-> Linear, CIELAB, CIEDE2000, LUT 33^3 tetraedryczny, MKL (Pitié 2007), IDT, guided_filter_rgb
      tone.rs                  # OKLab (Ottosson 2020/2021), gamut clipping o stałym odcieniu, Ekspozycja EV, Kontrast, Światła/Cienie, Biele/Czernie, Jaskrawość/Nasycenie, Przejrzystość
      wb.rs                    # Bradford CAT (Lam 1985), Temperatura/Odcień, jedno-klikowa pipeta WB::solve()
      histo.rs                 # Histogram 256-bin R/G/B/L + statystyki przepaleń i cieni
      geometry.rs              # Obrót z wpisaniem maksymalnego prostokąta o zadanym aspekcie, kadrowanie
      faces.rs                 # Wykrywanie twarzy + adaptacyjna maska skóry (rdzeń policzków/czoła, Mahalanobis, spójność CC, guided RGB)
      dnb.rs                   # Dodge & Burn v2: ważony filtr prowadzony na L* (twarz=256 px), rzeźbienie konturów z 5 punktów twarzy, fast_guided_rgb
      relight.rs               # Plastyka 3D: segmentacja postaci (Otsu + GMM/GrabCut + guided RGB), normalne z mapy głębi, wyśrodkowane światło kierunkowe
      bgu.rs                   # Bilateral Guided Upsampling (Chen et al. 2016): trilinear splat, blur, ridge solve 4x4, wielowątkowy slice (rayon)
      align.rs                 # Rejestracja oryginał <-> AI: CLAHE + LCN(σ=6), cechy + RANSAC, wieloskalowe ECC (afiniczne/homografia), gęsty przepływ optyczny
      transfer.rs              # changed_map (pasma DoG powyżej szumu Immerkaera), prepare(), PreviewCache, mix_cached(), render w paskach
      pipeline.rs              # Pełny potok renderowania zdjęcia (z wzorcem lub bez wzorca) + wyostrzanie Rec.709 + eksport JPG/TIFF
  retusz-gpu/                  # Akceleracja GPU na wgpu (WGSL compute shaders)
    src/
      lib.rs
      shaders/
        render_main.wgsl       # Compute shader WGSL: LUT 33^3 tetraedryczny + D&B/Plastyka/Clarity gain + winieta + pędzel + zebra przepaleń
        render_bgu.wgsl        # Compute shader WGSL: BGU slice_ 3D + mieszanie Kolor/Jasność na żywo
  retusz-app/                  # Natywna aplikacja desktopowa (eframe/egui + wgpu) oraz CLI (`cli`, `transfer`)
    src/
      main.rs                  # Punkt wejścia: subkomendy CLI (`cli`, `transfer`) lub start okna GUI
      theme.rs                 # Ciemny motyw studyjny (polskie znaki, stałe szerokości pól wartości, zero skakania układu)
      app.rs                   # Główne okno studyjne: lewy/dolny filmstrip, centralne płótno (Wynik / Przed|Po / Podział / Z wzorcem), prawy inspektor z Presetami i otwieraną sekcją Wzorca
      transfer_win.rs          # Okno „Przenieś grading z AI”: widok Obok siebie (Oryginał/AI | Wynik), 1:1 pełna rozdzielczość, Podział, Miganie, mapa zmian, pędzel z Cofnij
installer/
  retusz.iss                   # Skrypt Inno Setup budujący instalator Windows `Retusz-Setup-x64.exe`
```

---

## 3. Harmonogram realizacji i weryfikacji (TDD)

1. **Krok 1 — `retusz-core` (Silnik Rust + Presety + Sidecary + Algorytmy):**
   - Implementacja wszystkich modułów `retusz-core` oraz pełnego zestawu testów jednostkowych `cargo test` (CIEDE2000, OKLab gamut clipping, Ekspozycja EV, Kontrast, Biele/Czernie, Bradford WB + pipeta, LUT 33³ tetraedryczny, MKL, IDT, filtr prowadzony RGB, maska skóry, D&B v2, Plastyka 3D, BGU, rejestracja ECC/flow, `changed_map`, transfer, presety wbudowane i użytkownika, sidecar JSON).
2. **Krok 2 — `retusz-gpu` (Shadery WGSL `wgpu`) i `retusz-app` (Natywne GUI `eframe`/`egui` + CLI):**
   - Implementacja shaderów WGSL i potoku `wgpu` (z automatycznym fallbackiem CPU `rayon`, gdy brak fizycznego GPU w headless CI).
   - Implementacja natywnego okna studyjnego `eframe`/`egui` po polsku (prawy inspektor, histogram na górze, Presety, otwarte sekcje Światło/Kolor/Portret, zwijana sekcja **Dopasowanie do wzorca**, widoki `Wynik` / `Przed | Po` / `Podział` / `Z wzorcem`, okno **Przenieś grading z AI**) oraz CLI (`retusz cli`, `retusz transfer`).
3. **Krok 3 — System Presetów (`retusz/presets.py`) i przebudowa UI na układ studyjny w `retusz/gui/app.py`:**
   - Moduł `retusz/presets.py` (6 dopracowanych presetów wbudowanych + zapis/odczyt/import/eksport/usuwanie własnych presetów `*.retusz-preset.json`) kompatybilny 1:1 ze strukturą `Preset` w `retusz-core`.
   - Przebudowa `retusz/gui/app.py`:
     - centralne płótno z przełącznikiem widoków **`Wynik` | `Przed | Po` | `Podział` | `Z wzorcem`** (domyślnie bez pustego kafelka wzorca!),
     - prawy dokowany inspektor z histogramem na górze, paskiem **Presety** (`Wybierz`, `Zapisz…`, `Usuń`), zakładkami obróbki i **rozwijaną sekcją „▸ Dopasowanie do wzorca (AI)”** (bonus otwierany jednym kliknięciem lub automatycznie po wybraniu wzorca).
4. **Krok 4 — Instalator Windows (`installer/retusz.iss`) i CI (`.github/workflows/build-windows.yml`):**
   - Skrypt Inno Setup tworzący podpisany metadanymi `Jack4B` instalator **`Retusz-Setup-x64.exe`**.
   - Rozbudowa `.github/workflows/build-windows.yml`: budowa i testy Rust (`cargo test --release`, `cargo build --release`) + Python, pakowanie `Retusz-windows-x64.zip` oraz budowa instalatora `Retusz-Setup-x64.exe` (oba pliki dołączane do GitHub Releases przy tagu `v*`).
5. **Krok 5 — Pełna weryfikacja i wydanie `v0.2.0` (ZREALIZOWANE):**
   - `cargo test --workspace --release`: 10/10 testów przeszło (`retusz-core` 9/9, `retusz-gpu` 1/1), 0 ostrzeżeń.
   - `cargo build --workspace --release`: zbudowana natywna aplikacja `retusz-rs` (na Windows pakowana jako `Retusz-Studio-GPU.exe`), przetestowane zarówno CLI (`retusz-rs cli --preset "Ciepły złoty"`), jak i natywne okno `eframe`/`egui` + `wgpu` pod `xvfb-run`.
   - `.venv/bin/python -m pytest -q`: `146 passed, 7 skipped` (w tym `tests/test_presets.py` oraz `test_standalone_fit_without_reference_works_and_exports`).
   - `tools/gui_check.py`: `RESULT OK` (stabilność układu 3 zakładek `Światło i Kolor`, `Portret i Plastyka`, `▸ Wzorzec (AI)`, przełącznika widoków `Wynik | Przed | Po | Z wzorcem`, rozwijanej listy `Presety` + `+ Zapisz`).
   - `tools/quality_gate.py`: `GATE: PASS` (podgląd vs eksport ΔE 0.71–0.80 < 1.0).
   - Skrypt Inno Setup `installer/retusz.iss` oraz workflow `.github/workflows/build-windows.yml` generujący zarówno `Retusz-Setup-x64.exe`, jak i `Retusz-windows-x64.zip`.

