# Retusz na Androida – notatki architektoniczne (szkic, bez kodu)

Status: tylko plan. Wersja na Androida powstaje po domknięciu funkcji Windows (histogram, balans bieli, RAW, kadr – 0.1.3).

## Cel
Ten sam wygląd co na PC: telefon i komputer muszą dawać identyczny wynik z tego samego zdjęcia, wzorca i pliku
`NAZWA.retusz.json`, żeby można było zacząć serię na telefonie i skończyć na PC (albo odwrotnie).

## Proponowany podział
| Warstwa | Android | Uwagi |
|---|---|---|
| UI | Kotlin + Jetpack Compose, Material 3, ciemny motyw z akcentem #F5A33B | ekrany: Seria, Edycja (Wynik pełnoekranowy, przełącznik Źródło/Wzorzec), Kadr, Eksport, Ustawienia |
| Silnik kolorów | ONNX Runtime Mobile (NNAPI / XNNPACK) dla ModFlows B0 (18 MB); B6 opcjonalnie do pobrania | LUT 33³ liczony tak samo jak na PC; render LUT + mapy gain w shaderze OpenGL ES 3 / Vulkan (podgląd na żywo) |
| Głębia (Plastyka 3D) | Depth Anything V2 Small int8 przez ONNX Runtime | ta sama licencja Apache-2.0 |
| Twarze | YuNet (OpenCV) albo ML Kit Face Detection | YuNet = identyczne ramki jak na PC |
| RAW | LibRaw przez NDK (CDDL-1.0, bez zmian) albo `ImageDecoder`/DNG z Camera2 | NEF z karty przez SAF (Storage Access Framework) |
| Kolor | sRGB/Display P3: `ColorSpace` Androida + własne profile ICC CC0 | ekran telefonu jest zwykle P3 – podgląd przez zarządzanie barwą jak na PC |
| Pliki | SAF (`DocumentFile`), sidecary `.retusz.json` obok zdjęć, `retusz-seria.json` w folderze | ten sam format JSON (wersja 1) co na PC |

## Wspólny kod
Najmniej ryzykowne: przepisać czyste funkcje (wb, geometry, histo, dobór LUT, D&B) do Kotlina 1:1 i trzymać
**wspólne testy wzorcowe**: zestaw małych obrazów + oczekiwane wyniki (JSON z wartościami pikseli) generowany na PC;
test na Androidzie porównuje wynik z ΔE < 1. Alternatywa (później): rdzeń w C++ wspólny dla obu platform.

## Wydajność (cel)
Podgląd 1600 px < 150 ms po zmianie suwaka (shader), dopasowanie kolorów ze wzorcem < 3 s na średnim telefonie (B0),
eksport 24 Mpx < 6 s; praca w kafelkach 1024 px, żeby nie przekroczyć pamięci.

## Ryzyka
- Różnice float GPU vs CPU – rozwiązane testami wzorcowymi ΔE.
- Pamięć przy 24 Mpx RAW – dekodowanie w pasach, jak `rawio.develop`.
- Uprawnienia do plików (Android 11+): tylko SAF, bez `MANAGE_EXTERNAL_STORAGE`.
