"""
exporter.py
-----------
Moduł odpowiedzialny za wycinanie fragmentów obrazów i generowanie raportów CSV.
Zoptymalizowany pod kątem szybkości poprzez przetwarzanie równoległe oraz
organizację hierarchiczną plików wyjściowych.
"""

import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
from ..core.config import EXPORT_PADDING


def _process_image_crops(args) -> list:
    """
    Pomocnicza funkcja do przetwarzania ramek dla pojedynczego obrazu.
    Tworzy podfolder, zapisuje pliki f00X.jpg oraz lokalny plik CSV.
    Zwraca listę metadanych dla raportu zbiorczego.
    """
    fname, boxes, img_dir, output_dir = args
    img_path = img_dir / fname
    img = cv2.imread(str(img_path))
    if img is None:
        return []

    # Utworzenie podfolderu dla danego obrazu
    subfolder = output_dir / img_path.stem
    subfolder.mkdir(parents=True, exist_ok=True)

    metadata = []
    img_h, img_w = img.shape[:2]

    # Kolejne numery ramek (f001, f002...)
    frame_idx = 1

    for b in boxes:
        if not b.get("ok", True) or b.get("deleted", False):
            continue

        # Wycinanie z zabezpieczeniem zakresów i paddingiem
        p = EXPORT_PADDING
        x = max(0, int(b["x"]) - p)
        y = max(0, int(b["y"]) - p)
        w = min(int(b["w"]) + 2 * p, img_w - x)
        h = min(int(b["h"]) + 2 * p, img_h - y)

        if w <= 0 or h <= 0:
            continue

        crop = img[y : y + h, x : x + w]
        if crop.size == 0:
            continue

        # Nowe nazewnictwo f001, f002...
        frame_name = f"f{frame_idx:03d}"
        file_name = f"{frame_name}.jpg"
        label = b.get("label", "")

        cv2.imwrite(str(subfolder / file_name), crop)

        # Dodanie do metadanych obrazu i zbiorczych
        metadata.append(
            {"original_file": fname, "frame_name": frame_name, "label": label}
        )
        frame_idx += 1

    # Zapis lokalnego pliku CSV wewnątrz podfolderu
    if metadata:
        csv_path = subfolder / "labels.csv"
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["frame_name", "label"])
            writer.writeheader()
            for m in metadata:
                writer.writerow({"frame_name": m["frame_name"], "label": m["label"]})

    return metadata


def extract_approved(data: dict, img_dir: Path, output_dir: Path, debug_dir: Path):
    """
    Główna funkcja eksportu. Agreguje wyniki z wielu procesów i tworzy summary.csv.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Rozpoczynanie ustrukturyzowanej ekstrakcji dla {len(data)} plików...")

    tasks = [(fname, boxes, img_dir, output_dir) for fname, boxes in data.items()]

    # Równoległa ekstrakcja plików i zbieranie metadanych
    all_metadata = []
    with ProcessPoolExecutor() as executor:
        per_image_results = list(executor.map(_process_image_crops, tasks))

    # Spłaszczenie listy wyników
    for results in per_image_results:
        all_metadata.extend(results)

    # Zapis zbiorczego pliku summary.csv w głównym katalogu output
    if all_metadata:
        summary_path = output_dir / "summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
            fieldnames = ["original_file", "frame_name", "label"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_metadata)

    print(f"\nUkończono ekstrakcję.")
    print(f"  - Lokalizacja: {output_dir}")
    print(f"  - Liczba ramek: {len(all_metadata)}")
    print(f"  - Raport zbiorczy: {output_dir / 'summary.csv'}")
