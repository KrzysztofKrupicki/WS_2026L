# Red Box Detection Tool

Automatyczne i interaktywne narzędzie do detekcji czerwonych ramek na arkuszach dokumentów, przygotowane na potrzeby analizy danych wizualnych.

## Opis Projektu
System został opracowany w celu automatyzacji procesu lokalizacji i wycinania zaznaczonych na czerwono obszarów (np. pól podpisów lub pieczątek) ze skanów dokumentów. Projekt łączy algorytmy wizji komputerowej oparte na bibliotece OpenCV z interaktywnym interfejsem graficznym, co pozwala na szybką weryfikację i korektę wyników przez człowieka.

## Kluczowe Funkcjonalności
- **Wydajna Detekcja (Ultra-Extreme Sensitivity)**: Wykorzystanie przestrzeni barw HSV, operacji morfologicznych (Domykanie 5x5) oraz technologii `RETR_LIST` do wykrywania nawet najdrobniejszych i zagnieżdżonych elementów.
- **Inteligentna Filtracja**: Automatyczne usuwanie kontenerów nadrzędnych (np. ramek tabel), jeśli zawierają w sobie mniejsze, docelowe obiekty.
- **Batch Processing**: Równoległe przetwarzanie zestawów obrazów (`Parallel Mode`), co pozwala na analizę 50+ dokumentów w sekundy.
- **Interaktywny Reviewer**: autorski interfejs GUI pozwalający na:
    - Ręczne dodawanie, usuwanie i edycję etykiet ramek.
    - Szybką nawigację po liście plików (klawisz `L`) z podglądem breadcrumbów.
    - Pełną obsługę polskich znaków (Unicode) w interfejsie i plikach wyjściowych.
- **Ustrukturyzowany Eksport**: 
    - Automatyczne wycinanie fragmentów do dedykowanych podfolderów.
    - Sekwencyjne nazewnictwo plików (`f001`, `f002`...).
    - Generowanie raportów CSV (lokalnych i globalnego `summary.csv`) w formacie UTF-8 z BOM (kompatybilność z programem Excel).

## Wymagania Systemowe
- Python 3.10 lub nowszy.
- Biblioteki: `opencv-python`, `numpy`, `pillow`.

## Struktura Projektu
- `detect.py` – Skrypt do masowego wykrywania ramek na zdjęciach (High Performance).
- `review.py` – Główne narzędzie do manualnego przeglądu oraz finalnej, ustrukturyzowanej ekstrakcji danych.
- `red_box_tool/` – Rdzeń aplikacji (Engine, UI, Utilities).
- `img/` – Katalog ze zdjęciami wejściowymi.
- `output/` – Wynikowy katalog z hierarchicznymi folderami i raportami CSV.

## Instrukcja Obsługi
1. **Instalacja**:
   ```bash
   pip install opencv-python numpy pillow
   ```
2. **Przegląd i weryfikacja (metoda rekomendowana)**:
   ```bash
   python review.py
   ```
   Jeśli plik `approved.json` nie istnieje, skrypt **automatycznie** uruchomi `detect.py` w trybie równoległym, przygotowując wstępne dane do przeglądu.
   
   Klawiszologia w GUI:
   - `L` - lista wszystkich plików w sesji.
   - `S` / `Enter` (bez zaznaczenia) - zapis i zamknięcie.
   - `Enter` (przy zaznaczeniu) - potwierdzenie ramki i skok do następnej.
   - `Del` / `Backspace` - usuwanie ramki.
   - `1` / `2` - masowe OK / Odrzuć dla wszystkich ramek na zdjęciu.
   - `Lewo/Prawo` - nawigacja między zdjęciami.

3. **Ręczne uruchomienie detekcji (opcjonalne)**:
   ```bash
   python detect.py
   ```
   Przydatne, gdy chcesz wykonać samą analizę wsadową bez otwierania interfejsu graficznego.

4. **Finalna ekstrakcja**:
   ```bash
   python review.py --extract
   ```
   Wycięte fragmenty trafią do `output/` w formie uporządkowanej struktury folderów wraz z raportem `summary.csv`.

## Konfiguracja (`red_box_tool/core/config.py`)
- `SAVE_DEBUG_IMAGES`: (True/False) Włącza/wyłącza generowanie masek diagnostycznych w `debug_tuning/`.
- `HSV_LOWER/UPPER`: Parametry czułości koloru czerwonego.
- `MIN_BOX_W/H`: Minimalne wymiary wykrywanych ramek.