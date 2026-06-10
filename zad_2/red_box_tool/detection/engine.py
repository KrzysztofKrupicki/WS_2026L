"""
engine.py
---------
Główny silnik detekcji czerwonych ramek oparty na przestrzeni barw HSV
oraz operacjach morfologicznych. Zoptymalizowany pod kątem wysokiej czułości.
"""

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..core.config import (
    HSV_LOWER1,
    HSV_LOWER2,
    HSV_UPPER1,
    HSV_UPPER2,
    MIN_BOX_H,
    MIN_BOX_W,
    SAVE_DEBUG_IMAGES,
)


def detect_super_hybrid(
    image: np.ndarray,
    target_count: Optional[int] = None,
    debug_name: Optional[str] = None,
    debug_dir: Optional[Path] = None,
) -> list[dict]:
    """
    Wykonuje zaawansowaną detekcję czerwonych prostokątów na obrazie.
    Wspiera wykrywanie zagnieżdżone i automatycznie usuwa kontenery nadrzędne.
    """
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Segmentacja koloru czerwonego
    mask1 = cv2.inRange(img_hsv, HSV_LOWER1, HSV_UPPER1)
    mask2 = cv2.inRange(img_hsv, HSV_LOWER2, HSV_UPPER2)
    mask = cv2.bitwise_or(mask1, mask2)

    # Optymalizacja morfologiczna: Wysoka separacja
    k3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 1. Lekka dylatacja by połączyć "kropki"
    dilated = cv2.dilate(mask, k3, iterations=1)
    # 2. Domykanie rygorystyczne (5x5) by zamknąć kształty
    morphed = cv2.morphologyEx(
        dilated,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=2,
    )

    # Zapis maski diagnostycznej (jeśli włączony w config)
    if SAVE_DEBUG_IMAGES and debug_name and debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_dir / f"{debug_name}_hsv.png"), mask)
        cv2.imwrite(str(debug_dir / f"{debug_name}_morph.png"), morphed)

    # Wykrywanie konturów: RETR_LIST pozwala na znalezienie ramek wewnątrz innych konturów
    contours, _ = cv2.findContours(morphed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w >= MIN_BOX_W and h >= MIN_BOX_H:
            if w < image.shape[1] * 0.9 and h < image.shape[0] * 0.9:
                candidates.append(
                    {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "ok": True,
                        "manual": False,
                        "label": "",
                    }
                )

    # FILTRACJA KONTENERÓW:
    # Jeśli ramka A zawiera w sobie ramkę B, to ramkę A usuwamy.
    results = []
    for i, b1 in enumerate(candidates):
        is_container = False
        for j, b2 in enumerate(candidates):
            if i == j:
                continue
            if (
                b2["x"] > b1["x"] - 2
                and b2["y"] > b1["y"] - 2
                and b2["x"] + b2["w"] < b1["x"] + b1["w"] + 2
                and b2["y"] + b2["h"] < b1["y"] + b1["h"] + 2
            ):
                if (b1["w"] * b1["h"]) > (b2["w"] * b2["h"] * 1.15):
                    is_container = True
                    break
        if not is_container:
            results.append(b1)

    # Sortowanie wyników: rzędami (tolerancja 30px)
    results.sort(key=lambda b: (b["y"] // 30, b["x"]))

    # Opcjonalne rysowanie debug (jeśli włączony w config)
    if SAVE_DEBUG_IMAGES and debug_name and debug_dir:
        dbg_img = image.copy()
        for r in results:
            cv2.rectangle(
                dbg_img,
                (r["x"], r["y"]),
                (r["x"] + r["w"], r["y"] + r["h"]),
                (0, 255, 0),
                2,
            )
        cv2.imwrite(str(debug_dir / f"dbg_{debug_name}.jpg"), dbg_img)

    return results
