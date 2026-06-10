"""
geometry.py
-----------
Funkcje pomocnicze do obliczeń geometrycznych na obrazie.
"""


def iou(boxA: tuple, boxB: tuple) -> float:
    """
    Oblicza Intersection over Union (IoU) dla dwóch prostokątów.
    Pomaga w eliminacji duplikatów i nakładających się ramek.
    Format: (x, y, w, h)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0

    areaA = boxA[2] * boxA[3]
    areaB = boxB[2] * boxB[3]
    return interArea / float(areaA + areaB - interArea)
