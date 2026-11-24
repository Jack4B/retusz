# Retusz 0.1.2

Aplikacja na Windows: przenosi wygląd Twojego wzorca (np. zdjęcia obrobionego w Photoshopie) na całą serię zdjęć,
dodaje **Plastykę 3D** (światło kierunkowe z mapy głębi) i subtelny **dodge & burn**. Autor: Jack4B.

## Kontrola jakości

`tools/quality_gate.py` – bramka uruchamiana przed każdym wydaniem: porównuje wynik z wzorcem (LAB średnia/odchylenie, chroma i odcień skóry, kolorowość Hasler–Süsstrunk ≤ wzorzec +5%, przepalenia), sprawdza szwy/halo od Dodge&Burn i Plastyki 3D, identyczność (zdjęcie = wzorzec) oraz zgodność podglądu z eksportem (ΔE < 1). Tryb `--pair-export EKSPORT WZORZEC` ocenia gotowy eksport.

### 0.1.2
- Dodge&Burn liczone jednym obszarem dla wszystkich twarzy z wtapianiem krawędzi (koniec z twardymi szwami przy bliskich twarzach).
- Plastyka 3D: mikrokontrast na filtrze kierowanym (guided filter) – bez jasnych obwódek wokół okularów i głów; łagodniejsze ustawienia domyślne.
- Skóra: dopasowanie barwy a/b (MKL) z ograniczeniem chromy; bezpieczniki kolorów tła i skóry w pętli zamkniętej.
- Suwak „Pełne barwy” – wygląd jak na monitorze szerokogamutowym bez zarządzania barwą.

## Co umie
- **Otwórz zdjęcie** albo **Otwórz folder** (seria). **Wzorzec** wybierasz raz dla całego folderu (`retusz-seria.json`).
- Suwaki: **Siła kolorów**, **Skóra**, **Równanie skóry** (wyrównuje odcień skóry przy mieszanym świetle), **Winieta**,
  **Cienie**, **Światła**, **Nasycenie**, **Jaskrawość** (vibrance, oszczędza skórę), **Dodge&Burn** (auto, domyślnie 40%),
  **Plastyka 3D** (domyślnie 60%) i **Kierunek światła** (Auto = kierunek ze wzorca, albo ręcznie 0–355°).
- **Pędzel dodge/burn**: Rozjaśnij / Przyciemnij, rozmiar 5–600 px (suwak, klawisze `[` `]`, Alt+kółko), miękkość, przepływ,
  widoczny okrąg pędzla, Cofnij (Ctrl+Z), Wyczyść.
- **Przybliżanie**: Ctrl+kółko (25–400%, wokół kursora), przesuwanie środkowym przyciskiem albo spacja+przeciąganie,
  kółko przewija przybliżony obraz, Ctrl+0 = Dopasuj, Ctrl+1 = 100%. Źródło i Wynik przybliżają się razem; powyżej
  rozdzielczości podglądu Retusz liczy w tle pełną rozdzielczość. Pędzel działa przy każdym przybliżeniu.
- **Zarządzanie barwą**: podgląd jest przeliczany z profilu zdjęcia (np. AdobeRGB) na profil monitora z Windows,
  więc wygląda jak plik po eksporcie. Wzorzec w innym profilu jest najpierw przeliczany do profilu zdjęcia.
- **Edycja pojedynczych zdjęć w serii**: zmiany zapisują się obok zdjęcia w `NAZWA.jpg.retusz.json` i nie ruszają innych zdjęć.
- **Porównaj**: Podział, Obok siebie, Przełącz (przytrzymaj spację); Dopasuj albo 100%.
- **Eksport** (Ustawienia → Eksport): jakość JPG, próbkowanie koloru 4:4:4 / 4:2:2 / 4:2:0, profil barwny
  (jak oryginał / sRGB / AdobeRGB / Display P3), DPI, dłuższy bok w px, przyrostek nazwy, metadane (wszystkie / bez GPS / usuń),
  **wyostrzanie** na wyjście (Ekran, Druk albo własne: promień, ilość, próg; słabsze na skórze).
  Pliki trafiają do podfolderu `Retusz\` obok zdjęć. Oryginały nigdy nie są nadpisywane.
- **Model AI** (ModFlows B6, 119 MB) pobierany przy pierwszym starcie z wydania
  [models-v1](https://github.com/Jack4B/retusz/releases/tag/models-v1); do tego czasu działa lekki model B0 (18 MB, wbudowany).
  Bez modeli działa tryb „Bez AI (IDT)”.
- **Obliczenia AI**: Automatycznie (jednorazowy test: karta graficzna przez DirectML albo procesor, wybiera szybsze),
  albo na stałe CPU / GPU. Na zintegrowanych kartach Intel procesor bywa 2× szybszy.

![Ustawienia](docs/screenshots/05_ustawienia_ogolne.png) ![Eksport](docs/screenshots/06_ustawienia_eksport.png)

## Uruchomienie ze źródeł (Windows)
```bat
py -3.11 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python -m retusz                      & rem okno
.venv\Scripts\python -m retusz D:\Sesja --reference D:\Sesja\wzorzec.jpg
.venv\Scripts\python -m retusz cli D:\Sesja --reference D:\Sesja\wzorzec.jpg --long-edge 2048 --suffix _web
```
Budowanie `.exe`: zobacz [BUILD.md](BUILD.md). Gotowa paczka Windows: zakładka *Releases* albo artefakt z GitHub Actions.

## Czasy (zdjęcie 24 Mpx, eksport z wyostrzaniem)
| Komputer | Na zdjęcie |
|---|---|
| procesor 8 rdzeni (serwer, bez GPU) | ~2,7 s |
| i7-7700 + Intel HD 630 (Windows 10, CPU) | zobacz BUILD.md |
Pierwsze zdjęcie po instalacji: +kilka sekund na jednorazową kompilację (potem z pamięci podręcznej).

## Dane użytkownika
`%LOCALAPPDATA%\Retusz\` — `settings.json`, `device.json` (wybór CPU/GPU), `models\modflows_b6_fp16.onnx`, `numba-cache\`.

## Testy
`python -m pytest -q`

## Licencja
Kod: © Jack4B, wszelkie prawa zastrzeżone (repozytorium jest publiczne do wglądu). Komponenty i modele: [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
