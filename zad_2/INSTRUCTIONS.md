# Instrukcja Obsługi: Red Box Tool (zad_2)

Narzędzie **Red Box Tool** służy do automatycznej detekcji, manualnej weryfikacji oraz ustrukturyzowanego eksportu czerwonych ramek (np. pól podpisów, pieczątek) z dokumentów graficznych.

---

## 1. Instalacja i Przygotowanie

### Wymagania
*   **Python 3.10** lub nowszy.
*   Biblioteki: `opencv-python`, `numpy`, `pillow`.

### Instalacja zależności
Otwórz terminal w folderze projektu i wykonaj:
```bash
pip install opencv-python numpy pillow
```

### Przygotowanie danych
1.  Umieść zdjęcia dokumentów w formacie `.jpg`, `.png` lub `.bmp` w katalogu `img/`.
2.  (Opcjonalnie) Jeśli masz listę oczekiwanej liczby ramek dla poszczególnych plików, wpisz ją do `correct_count_of_red_boxes.txt` w formacie: `nazwa_pliku.jpg, liczba`.

---

## 2. Przepływ Pracy (Workflow)

Aplikacja wspiera dwuetapowy proces: automatyczną detekcję wsadową oraz interaktywny przegląd.

### KROK 1: Automatyczna Detekcja
Uruchom skrypt, aby szybko przetworzyć wszystkie zdjęcia w folderze `img/`:
```bash
python detect.py
```
*   Skrypt wykorzystuje **wszystkie rdzenie procesora** (tryb równoległy).
*   Wyniki zostaną zapisane w pliku `approved.json`.
*   W folderze `debug_tuning/` pojawią się maski diagnostyczne (jeśli włączone w konfiguracji).

### KROK 2: Weryfikacja i Edycja (GUI)
Uruchom interfejs graficzny, aby sprawdzić poprawność wykrytych ramek:
```bash
python review.py
```
*   Jeśli `approved.json` nie istnieje, skrypt sam zaproponuje uruchomienie detekcji.
*   Możesz tu poprawiać błędy automatu, dodawać brakujące ramki i nadawać im etykiety.

### KROK 3: Finalna Ekstrakcja
Gdy wszystkie ramki są zweryfikowane, wyeksportuj je do plików:
```bash
python review.py --extract
```
*   Wycięte fragmenty trafią do folderu `output/`.
*   Zostanie wygenerowany plik `summary.csv` (zgodny z Excel).

---

## 3. Sterowanie w Interfejsie (review.py)

### Myszka
*   **Lewy Przycisk (LPM)**: Zaznaczenie ramki / Przeciąganie obrazu (Pan).
*   **Prawy Przycisk (PPM)**: Rysowanie nowej ramki (przeciągnij, aby utworzyć obszar).
*   **Rolka (Scroll)**: Zoom (przybliżanie/oddalanie) w punkcie kursora.

### Klawiatura - Nawigacja Ogólna
*   `L` – Otwiera **listę plików**. Użyj `Góra/Dół` i `Enter`, aby szybko skoczyć do konkretnego zdjęcia.
*   `Enter` (gdy nic nie jest zaznaczone) – Zapisuje postęp i przechodzi do **następnego zdjęcia**.
*   `F` – Dopasowuje widok do okna (Fit).
*   `0` – Skala 1:1 (100%).
*   `1` – Oznacz **wszystkie** ramki na zdjęciu jako **OK** (zielone).
*   `2` – Oznacz **wszystkie** ramki na zdjęciu jako **Odrzucone** (szare).
*   `Esc` / `Q` – Zamknięcie programu (bez zapisu bieżącego zdjęcia).

### Klawiatura - Edycja Zaznaczonej Ramki
*   `Enter` – Zatwierdza ramkę jako OK i zaznacza **następną**.
*   `Del` / `Backspace` – Usuwa zaznaczoną ramkę.
*   **Pisanie na klawiaturze** – Automatycznie dodaje tekst (etykietę) do ramki (obsługuje polskie znaki).

---

## 4. Struktura Wyników (output/)

Po wykonaniu komendy `--extract`, struktura folderów wygląda następująco:
*   `output/`
    *   `nazwa_zdjęcia_1/`
        *   `f001.jpg` (wycięta ramka)
        *   `f002.jpg` ...
        *   `report.csv` (dane dla tego konkretnego pliku)
    *   `summary.csv` (zbiorczy raport dla całej sesji)

---

## 5. Zaawansowana Konfiguracja

Parametry detekcji można dostosować w pliku `red_box_tool/core/config.py`:
*   `HSV_LOWER` / `HSV_UPPER`: Zakres koloru czerwonego (jeśli ramki są zbyt jasne/ciemne).
*   `MIN_BOX_W` / `MIN_BOX_H`: Minimalny rozmiar ramki, który ma być brany pod uwagę.
*   `SAVE_DEBUG_IMAGES`: Ustaw na `False`, aby przyspieszyć działanie i nie zaśmiecać dysku maskami.
