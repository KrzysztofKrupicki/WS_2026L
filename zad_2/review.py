"""
review.py
---------
Interaktywne narzędzie do weryfikacji wykrytych ramek oraz finalnej ekstrakcji danych.
"""

import argparse
from pathlib import Path

import cv2

from red_box_tool.core.config import APPROVED_FILE as DEFAULT_APPROVED_FILE
from red_box_tool.core.config import CORRECT_COUNTS_FILE, DEBUG_DIR, IMG_DIR
from red_box_tool.core.io_utils import (
    load_approved,
    load_expected_counts,
    save_approved,
)
from red_box_tool.detection.engine import detect_super_hybrid
from red_box_tool.interface.exporter import extract_approved
from red_box_tool.interface.reviewer import Reviewer

OUTPUT_DIR = Path("output")


def main() -> None:
    """Główna funkcja narzędzia do przeglądu i eksportu."""
    parser = argparse.ArgumentParser(
        description="Narzędzie do przeglądu i etykietowania ramek."
    )
    parser.add_argument(
        "--image", "-i", default=None, help="Ścieżka do konkretnego obrazu."
    )
    parser.add_argument(
        "--extract",
        "-e",
        nargs="?",
        const=str(DEFAULT_APPROVED_FILE),
        default=None,
        help="Wyodrębnij ramki z pliku JSON.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Usuń plik approved.json i zacznij od nowa.",
    )
    args = parser.parse_args()

    if args.reset and DEFAULT_APPROVED_FILE.exists():
        DEFAULT_APPROVED_FILE.unlink()
        print(f"Usunięto {DEFAULT_APPROVED_FILE}")
        return

    # Automatyczne uruchomienie detekcji, jeśli brakuje pliku wynikowego (approved.json)
    if not DEFAULT_APPROVED_FILE.exists():
        # Uruchamiamy tylko jeśli jesteśmy w trybie przeglądu lub domyślnej ekstrakcji
        if args.extract is None or Path(args.extract) == DEFAULT_APPROVED_FILE:
            print(
                f"\n[INFO] Plik {DEFAULT_APPROVED_FILE} nie istnieje. Uruchamiam detekcję automatyczną (detect.py)..."
            )
            import detect

            detect.main()

    # Tryb EKSTRAKCJI: wycinanie ramek do osobnych plików
    if args.extract is not None:
        extract_file = Path(args.extract)
        if not extract_file.exists():
            print(f"BŁĄD: Plik {extract_file} nie istnieje.")
            return
        data = load_approved(extract_file)
        if data:
            extract_approved(data, IMG_DIR, OUTPUT_DIR, DEBUG_DIR)
        return

    # Tryb PRZEGLĄDU: uruchomienie GUI
    approved_data = load_approved(DEFAULT_APPROVED_FILE)
    expected_counts = load_expected_counts(CORRECT_COUNTS_FILE)

    if args.image:
        images = [Path(args.image)]
    else:
        # Znajdowanie wszystkich wspieranych formatów obrazów
        extensions = ["jpg", "JPG", "jpeg", "png", "bmp"]
        images = sorted({p for ext in extensions for p in IMG_DIR.glob(f"*.{ext}")})

    if not images:
        print(f"Brak obrazów do przetworzenia w {IMG_DIR}/")
        return

    print(f"Znaleziono {len(images)} obrazów. Rozpoczynanie przeglądu...")
    all_fnames = [p.name for p in images]

    cur_idx = 0
    total = len(images)

    while 0 <= cur_idx < total:
        img_path = images[cur_idx]
        fname = img_path.name
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  [BŁĄD] Nie można otworzyć: {fname}")
            cur_idx += 1
            continue

        existing = approved_data.get(fname, [])
        target = expected_counts.get(fname)

        # Automatyczne uruchomienie detekcji, jeśli brak danych dla pliku
        if not existing and fname in expected_counts:
            print(f"\n[{fname}] Detekcja automatyczna (cel: {target})...")
            existing = detect_super_hybrid(
                image, target, debug_name=img_path.stem, debug_dir=DEBUG_DIR
            )
        else:
            print(
                f"\n[{fname}] ({cur_idx+1}/{total}) Wczytywanie ramek ({len(existing)} wpisów)"
            )

        # Uruchomienie klasy interfejsu
        reviewer = Reviewer(image, fname, existing, target=target, all_files=all_fnames)
        was_saved = reviewer.run()

        # Zapisanie zmian w sesji
        approved_data[fname] = reviewer.get_result()
        save_approved(approved_data, DEFAULT_APPROVED_FILE)

        # Obsługa nawigacji (skok do konkretnego pliku lub następnego)
        if reviewer.requested_file:
            try:
                cur_idx = all_fnames.index(reviewer.requested_file)
                print(f"  -> Skok do: {reviewer.requested_file}")
            except ValueError:
                cur_idx += 1
        elif not was_saved:
            print("  Wychodzenie.")
            break
        else:
            cur_idx += 1

    print("\nPrzegląd ukończony.")


if __name__ == "__main__":
    main()
