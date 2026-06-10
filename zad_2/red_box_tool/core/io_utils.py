"""
io_utils.py
-----------
Narzędzia pomocnicze do obsługi plików (Wczytywanie/Zapisywanie).
Obsługuje kodowanie UTF-8 z BOM dla pełnej kompatybilności z Excel/Windows.
"""

import json
from pathlib import Path


def load_approved(path: Path) -> dict:
    """Wczytuje zatwierdzone ramki z pliku JSON. Zwraca pusty słownik, jeśli plik nie istnieje."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def save_approved(data: dict, path: Path):
    """Zapisuje słownik ramek do pliku JSON w sposób czytelny (UTF-8 z BOM)."""
    with open(path, "w", encoding="utf-8-sig") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  [Zapisano] {path.name}")


def load_expected_counts(path: Path) -> dict:
    """
    Wczytuje oczekiwaną liczbę ramek dla poszczególnych plików.
    Format pliku: nazwa_pliku;liczba
    """
    counts = {}
    if not path.exists():
        return counts
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line or ";" not in line:
                continue
            fname, count = line.split(";")
            counts[fname] = int(count)
    return counts
