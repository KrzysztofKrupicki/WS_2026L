"""
detect.py
---------
Narzędzie do wsadowego (batch) wykrywania czerwonych ramek na zestawach zdjęć.
Wykorzystuje przetwarzanie równoległe w celu oszczędności czasu.
"""

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Optional

import cv2

from red_box_tool.core.config import (
    APPROVED_FILE,
    CORRECT_COUNTS_FILE,
    DEBUG_DIR,
    IMG_DIR,
)
from red_box_tool.core.io_utils import (
    load_approved,
    load_expected_counts,
    save_approved,
)
from red_box_tool.detection.engine import detect_super_hybrid


def process_single_image(
    img_path: Path, expected: Optional[int]
) -> tuple[str, list[dict]]:
    """Przetwarza pojedyncze zdjęcie (wywoływane w osobnym procesie)."""
    image = cv2.imread(str(img_path))
    if image is None:
        return img_path.name, []
    results = detect_super_hybrid(
        image, target_count=expected, debug_name=img_path.stem, debug_dir=DEBUG_DIR
    )
    return img_path.name, results


def main():
    """Główna funkcja skryptu detekcji."""
    extensions = ["jpg", "JPG", "jpeg", "png", "bmp"]
    images = sorted({p for ext in extensions for p in IMG_DIR.glob(f"*.{ext}")})
    if not images:
        print(f"Brak zdjęć w {IMG_DIR}")
        return

    expected_counts = load_expected_counts(CORRECT_COUNTS_FILE)
    approved_data = load_approved(APPROVED_FILE)

    print(f"Rozpoczynanie detekcji dla {len(images)} plików (Parallel Mode)...")
    start_t = time.time()

    # Wykorzystanie ProcessPoolExecutor do równoległego przetwarzania
    with ProcessPoolExecutor() as executor:
        futures = []
        for img_path in images:
            if img_path.name not in approved_data:
                target = expected_counts.get(img_path.name)
                futures.append(executor.submit(process_single_image, img_path, target))
            else:
                print(f"  [Skip] {img_path.name} (już istnieje)")

        for future in futures:
            fname, boxes = future.result()
            approved_data[fname] = boxes
            print(f"  [OK] {fname}: wykryto {len(boxes)} ramek")

    # Zapis zbiorczy wyników
    save_approved(approved_data, APPROVED_FILE)
    print(
        f"\nUkończono w {time.time() - start_t:.2f}s. Wyniki zapisano w {APPROVED_FILE}"
    )


if __name__ == "__main__":
    main()
