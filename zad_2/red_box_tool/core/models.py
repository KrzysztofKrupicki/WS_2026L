from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Box:
    """
    Reprezentuje pojedynczą ramkę (Bounding Box) wykrytą na obrazie.
    Przechowuje współrzędne, stan zatwierdzenia oraz opcjonalną etykietę tekstową.
    """

    x: int
    y: int
    w: int
    h: int
    ok: bool = True
    manual: bool = False
    label: str = ""
    deleted: bool = False

    @property
    def rect(self) -> tuple[int, int, int, int]:
        """Zwraca geometrię ramki w formacie (x, y, w, h)."""
        return (self.x, self.y, self.w, self.h)

    def to_dict(self) -> dict:
        """Konwertuje obiekt Box na słownik (serializacja do JSON)."""
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "ok": self.ok,
            "manual": self.manual,
            "label": self.label,
            "deleted": self.deleted,
        }
