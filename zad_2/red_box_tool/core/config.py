from pathlib import Path

# Ścieżki
IMG_DIR = Path("img")
DEBUG_DIR = Path("debug_tuning")
CORRECT_COUNTS_FILE = Path("correct_count_of_red_boxes.txt")
APPROVED_FILE = Path("approved.json")
APPROVED_AUTO_FILE = Path("approved_auto.json")

# Parametry detekcji (Ekstremalna czułość)
HSV_LOWER1 = (0, 5, 5)
HSV_UPPER1 = (20, 255, 255)
HSV_LOWER2 = (160, 5, 5)
HSV_UPPER2 = (180, 255, 255)

# Rozmiary kerneli
MORPH_KERNEL_SIZE = (7, 7)

# Filtry rozmiaru
MIN_BOX_W = 12
MIN_BOX_H = 10

# Inne
CELL_W_ESTIMATE = 80.0
YIELD_LIMIT_MULTIPLIER = 5
YIELD_LIMIT_MIN = 200
EXPORT_PADDING = 5  # Dodatkowy margines przy wycinaniu ramek

# Debugowanie
SAVE_DEBUG_IMAGES = True
