"""
reviewer.py
-----------
Moduł odpowiedzialny za interfejs graficzny (GUI) narzędzia do przeglądu ramek.
Umożliwia interaktywną weryfikację, edycję i nawigację po zbiorze danych.
"""

from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..core.geometry import iou
from ..core.models import Box

# Parametry okna i nawigacji
WINDOW = "OCR Review"
PANEL_W = 400
ZOOM_STEP = 0.15
ZOOM_MIN = 0.1
ZOOM_MAX = 8.0

# Definicje kolorów (format BGR)
COL_OK = (30, 210, 30)
COL_REJECT = (90, 90, 90)
COL_DRAW = (0, 180, 255)
COL_MANUAL = (255, 130, 0)
COL_SELECTED = (200, 255, 255)
COL_STATUS = (20, 20, 20)
COL_PANEL = (40, 40, 45)
COL_TEXT = (220, 220, 220)
COL_HIGHLIGHT = (0, 255, 255)

# Kody specjalne klawiszy
KEY_ESC = 27
KEY_ENTER = (13, 10)
KEY_BACKSPACE = (8, 127)
KEY_DELETE_WIN = 3014656
KEY_DELETE_LINUX = 255
KEY_DELETE_CODES = (KEY_DELETE_WIN, KEY_DELETE_LINUX)
KEY_UP = 2490368
KEY_DOWN = 2621440


class Reviewer:
    """
    Główna klasa interfejsu graficznego.
    Obsługuje wyświetlanie obrazu, rysowanie nakładek (overlays), zdarzenia myszy
    oraz klawiatury (skróty klawiszowe i wprowadzanie tekstu).
    """

    def __init__(
        self,
        image: np.ndarray,
        fname: str,
        existing_boxes: list[dict],
        target: Optional[int] = None,
        all_files: Optional[list[str]] = None,
        row_tolerance: int = 30,
        min_w: int = 30,
        min_h: int = 20,
    ) -> None:
        """
        Inicjalizuje obiekt Reviewer dla konkretnego zdjęcia.

        Args:
            image: Macierz obrazu wejściowego.
            fname: Nazwa przetwarzanego pliku.
            existing_boxes: Lista już istniejących (wykrytych) ramek.
            target: Oczekiwana liczba ramek na tym zdjęciu.
            all_files: Pełna lista plików w sesji (do nawigacji).
            row_tolerance: Tolerancja grupowania ramek w rzędy (px).
            min_w, min_h: Minimalne wymiary nowo rysowanych ramek.
        """
        self.orig = image.copy()
        self.ih, self.iw = image.shape[:2]
        self.fname = fname
        self.target = target
        self.all_files = all_files or [fname]
        self.row_tolerance = row_tolerance
        self.min_w = min_w
        self.min_h = min_h

        # Konwersja słowników na obiekty klasy Box
        self.boxes: list[Box] = []
        self.deleted_boxes: list[Box] = []
        for e in existing_boxes:
            if e.get("ok", True) and not e.get("deleted", False):
                b_rect = (e["x"], e["y"], e["w"], e["h"])
                # Usuwanie duplikatów (jeśli IoU > 15%)
                if not any(iou(b_rect, box.rect) > 0.15 for box in self.boxes):
                    self.boxes.append(
                        Box(
                            e["x"],
                            e["y"],
                            e["w"],
                            e["h"],
                            ok=e.get("ok", True),
                            manual=e.get("manual", False),
                            label=e.get("label", ""),
                        )
                    )

        # Sortowanie logiczne (czytelność przy nawigacji Enter)
        self.boxes.sort(key=lambda b: (b.y // self.row_tolerance, b.x))

        self.win_w, self.win_h = 1400, 800
        self.img_w = self.win_w - PANEL_W
        self.scale = 1.0
        self.offset = [0.0, 0.0]

        # Stan interfejsu
        self._pan_start = None
        self._pan_off_start = None
        self._draw_start = None
        self._draw_cur = None
        self._selected: Optional[int] = None
        self.saved = False
        self._dirty = True

        self._list_active = False
        self._list_idx = 0
        try:
            self._list_idx = self.all_files.index(fname)
        except ValueError:
            pass
        self.requested_file: Optional[str] = None

        # Czcionki PIL (wsparcie Unicode)
        def _get_font(size):
            for f in ["arial.ttf", "segoeui.ttf", "C:\\Windows\\Fonts\\arial.ttf"]:
                try:
                    return ImageFont.truetype(f, size)
                except:
                    continue
            return ImageFont.load_default()

        self.font_main = _get_font(16)
        self.font_large = _get_font(26)
        self.font_small = _get_font(12)

    def _draw_text(
        self, img: np.ndarray, text: str, pos: tuple[int, int], font, color, anchor="la"
    ):
        """
        Rysuje tekst na obrazie z obsługą Unicode (Polskie znaki).
        Dla czystego ASCII używa szybkich funkcji OpenCV.
        """
        is_unicode = any(ord(c) > 127 for c in text)
        if not is_unicode:
            scale = 0.4
            if font == self.font_main:
                scale = 0.5
            elif font == self.font_large:
                scale = 0.8
            y_off = (
                12
                if font == self.font_small
                else (16 if font == self.font_main else 26)
            )
            cv2.putText(
                img,
                text,
                (pos[0], pos[1] + y_off),
                cv2.FONT_HERSHEY_SIMPLEX,
                scale,
                color,
                1,
                cv2.LINE_AA,
            )
        else:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            draw.text(
                pos, text, font=font, fill=(color[2], color[1], color[0]), anchor="la"
            )
            img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    @staticmethod
    def _shorten(name: str, max_len: int = 25) -> str:
        """Skraca długie nazwy plików do formatu: poczatek...koniec.rozszerzenie."""
        if len(name) <= max_len:
            return name
        ext = name.split(".")[-1] if "." in name else ""
        part = (max_len - 5) // 2
        return f"{name[:part]}...{name[-(part+len(ext)+1):]}"

    def _fit(self, reset_view: bool = True) -> None:
        """Dostosowuje parametry widoku (zoom/offset) do aktualnego rozmiaru okna."""
        rect = cv2.getWindowImageRect(WINDOW)
        if rect[2] > 0 and rect[3] > 0:
            self.win_w, self.win_h = rect[2], rect[3]
            self.img_w = self.win_w - PANEL_W
        if reset_view:
            sx = self.img_w / self.iw
            sy = (self.win_h - 40) / self.ih
            self.scale = min(sx, sy, 1.0)
            self.offset = [0.0, 0.0]

    def _img2scr(self, px: float, py: float) -> tuple[int, int]:
        """Konwertuje współrzędne obrazu na współrzędne ekranowe."""
        return (
            int((px - self.offset[0]) * self.scale),
            int((py - self.offset[1]) * self.scale),
        )

    def _scr2img(self, sx: int, sy: int) -> tuple[float, float]:
        """Konwertuje współrzędne ekranowe na współrzędne obrazu."""
        return (sx / self.scale + self.offset[0], sy / self.scale + self.offset[1])

    def _box_at(self, imgx: float, imgy: float) -> Optional[int]:
        """Wskazuje indeks ramki pod podanym punktem (kołowa nawigacja wstecz)."""
        for i in range(len(self.boxes) - 1, -1, -1):
            b = self.boxes[i]
            if b.x <= imgx <= b.x + b.w and b.y <= imgy <= b.y + b.h:
                return i
        return None

    def _remove_box(self, idx: int) -> None:
        """Przenosi ramkę do listy usuniętych (soft-delete)."""
        box = self.boxes.pop(idx)
        box.deleted = True
        self.deleted_boxes.append(box)
        if self._selected == idx:
            self._selected = None
        elif self._selected is not None and self._selected > idx:
            self._selected -= 1

    def _render(self) -> np.ndarray:
        """Generuje finalną klatkę interfejsu (Obraz + Canvas + Panele)."""
        self._fit(reset_view=False)
        vis_w, vis_h = self.img_w / self.scale, (self.win_h - 40) / self.scale
        self.offset[0] = max(0.0, min(self.offset[0], max(0.0, self.iw - vis_w)))
        self.offset[1] = max(0.0, min(self.offset[1], max(0.0, self.ih - vis_h)))

        ox, oy = int(self.offset[0]), int(self.offset[1])
        x2, y2 = min(ox + int(vis_w) + 1, self.iw), min(oy + int(vis_h) + 1, self.ih)
        crop = self.orig[oy:y2, ox:x2].copy()

        cw, ch = int((x2 - ox) * self.scale), int((y2 - oy) * self.scale)
        if cw < 1 or ch < 1:
            canvas = np.zeros((10, 10, 3), np.uint8)
        else:
            canvas = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_LINEAR)

        for i, box in enumerate(self.boxes):
            sx1, sy1 = int((box.x - ox) * self.scale), int((box.y - oy) * self.scale)
            sx2, sy2 = int((box.x + box.w - ox) * self.scale), int(
                (box.y + box.h - oy) * self.scale
            )
            if sx2 < 0 or sy2 < 0 or sx1 > cw or sy1 > ch:
                continue

            if i == self._selected:
                color, thick = COL_SELECTED, 3
            elif box.manual:
                color, thick = COL_MANUAL, (2 if box.ok else 1)
            elif box.ok:
                color, thick = COL_OK, 2
            else:
                color, thick = COL_REJECT, 1

            cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), color, thick)
            if self.scale >= 0.3:
                if box.label:
                    self._draw_text(
                        canvas,
                        f"{i+1} '{box.label}'",
                        (sx1 + 2, sy1 - 18 if sy1 > 20 else sy1 + 2),
                        self.font_small,
                        color,
                    )
                else:
                    cv2.putText(
                        canvas,
                        str(i + 1),
                        (sx1 + 2, sy1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )

        if self._draw_start and self._draw_cur:
            dx1, dy1 = int(
                (min(self._draw_start[0], self._draw_cur[0]) - ox) * self.scale
            ), int((min(self._draw_start[1], self._draw_cur[1]) - oy) * self.scale)
            dx2, dy2 = int(
                (max(self._draw_start[0], self._draw_cur[0]) - ox) * self.scale
            ), int((max(self._draw_start[1], self._draw_cur[1]) - oy) * self.scale)
            cv2.rectangle(canvas, (dx1, dy1), (dx2, dy2), COL_DRAW, 2)

        full = np.zeros((self.win_h, self.win_w, 3), np.uint8)
        full[:ch, :cw] = canvas
        panel = np.full((self.win_h, PANEL_W, 3), COL_PANEL, np.uint8)
        self._render_panel_into(panel)
        full[:, self.img_w :] = panel

        cv2.rectangle(
            full, (0, self.win_h - 36), (self.img_w, self.win_h), COL_STATUS, -1
        )
        ok_count = sum(1 for b in self.boxes if b.ok)
        target_str = f"/{self.target}" if self.target is not None else ""
        st = f" Plik: {self._shorten(self.fname, 55)}  |  OK={ok_count}{target_str} ({len(self.boxes)})"
        self._draw_text(full, st, (10, self.win_h - 26), self.font_main, COL_TEXT)

        if self._list_active:
            self._render_file_list(full)
        return full

    def _render_panel_into(self, panel: np.ndarray) -> None:
        """Rysuje zawartość bocznego panelu (detale zaznaczenia, help)."""
        py = 35
        self._draw_text(panel, "PANEL EDYCJI", (20, py), self.font_large, COL_HIGHLIGHT)
        py += 45
        if self._selected is not None:
            box = self.boxes[self._selected]
            self._draw_text(
                panel,
                f"Zaznaczono nr {self._selected + 1}",
                (20, py),
                self.font_main,
                COL_TEXT,
            )
            py += 25
            cx1, cy1 = max(box.x - 15, 0), max(box.y - 15, 0)
            cx2, cy2 = min(box.x + box.w + 15, self.iw), min(
                box.y + box.h + 15, self.ih
            )
            z_crop = self.orig[cy1:cy2, cx1:cx2]
            if z_crop.size > 0:
                scale_z = min((PANEL_W - 40) / z_crop.shape[1], 150 / z_crop.shape[0])
                nh, nw = int(z_crop.shape[0] * scale_z), int(z_crop.shape[1] * scale_z)
                z_crop_r = cv2.resize(z_crop, (nw, nh))
                panel[py : py + nh, 20 : 20 + nw] = z_crop_r
                py += nh + 25
            self._draw_text(
                panel, box.label + "_", (30, py), self.font_large, (0, 255, 0)
            )
            py += 45
            hints = [
                ("[Klawisze] - pisanie", (180, 180, 180)),
                ("[BS] - usuń znak", (180, 180, 180)),
                ("[DEL] - usuń ramkę", (120, 120, 255)),
                ("[ENTER] - OK / Dalej", COL_DRAW),
            ]
            for txt, col in hints:
                self._draw_text(panel, txt, (20, py), self.font_small, col)
                py += 20
        else:
            help_lines = [
                ("[Enter] - Zapisz", COL_TEXT),
                ("[L] - Lista plików", COL_HIGHLIGHT),
                ("[1/2] OK/Odrzuć", (150, 150, 150)),
                ("[F/0] Dopasuj/100%", (150, 150, 150)),
                ("PPM: Nowa ramka", (150, 150, 150)),
                ("Drag LPM: Przesuń", (150, 150, 150)),
            ]
            for txt, col in help_lines:
                self._draw_text(panel, txt, (20, py), self.font_small, col)
                py += 22

    def _render_file_list(self, full: np.ndarray) -> None:
        """Wyświetla centralną nakładkę z listą wszystkich plików."""
        lw, lh = 700, 500
        lx, ly = (self.img_w - lw) // 2, (self.win_h - lh) // 2
        cv2.rectangle(full, (lx, ly), (lx + lw, ly + lh), (20, 20, 25), -1)
        cv2.rectangle(full, (lx, ly), (lx + lw, ly + lh), COL_HIGHLIGHT, 2)
        self._draw_text(
            full, "WYBIERZ PLIK", (lx + 20, ly + 20), self.font_large, COL_HIGHLIGHT
        )
        visible = 12
        start = max(0, self._list_idx - visible // 2)
        end = min(len(self.all_files), start + visible)
        cur_py = ly + 70
        for i in range(start, end):
            is_cur = i == self._list_idx
            if is_cur:
                cv2.rectangle(
                    full,
                    (lx + 10, cur_py - 5),
                    (lx + lw - 10, cur_py + 32),
                    (60, 60, 70),
                    -1,
                )
            self._draw_text(
                full,
                f"{i+1}. {self._shorten(self.all_files[i], 55)}",
                (lx + 30, cur_py),
                self.font_large if is_cur else self.font_main,
                COL_HIGHLIGHT if is_cur else COL_TEXT,
            )
            cur_py += 35

    def _mouse(self, event, sx, sy, flags, param):
        """Obsługuje zdarzenia myszy (zoom, przesuwanie, zaznaczanie, rysowanie)."""
        rect = cv2.getWindowImageRect(WINDOW)
        if rect[2] > 0 and rect[3] > 0:
            sx, sy = int(sx * (self.win_w / rect[2])), int(sy * (self.win_h / rect[3]))
        if self._list_active:
            if event == cv2.EVENT_LBUTTONDOWN:
                self._list_active = False
                self._dirty = True
            return
        if sx > self.img_w:
            return
        imgx, imgy = self._scr2img(sx, sy)

        if event == cv2.EVENT_MOUSEWHEEL:
            new_scale = max(
                ZOOM_MIN,
                min(ZOOM_MAX, self.scale * (1 + (1 if flags > 0 else -1) * ZOOM_STEP)),
            )
            self.offset[0] += imgx * (1 - new_scale / self.scale)
            self.offset[1] += imgy * (1 - new_scale / self.scale)
            self.scale, self._dirty = new_scale, True
        elif event == cv2.EVENT_LBUTTONDOWN:
            self._pan_start, self._pan_off_start = (sx, sy), self.offset[:]
        elif event == cv2.EVENT_MOUSEMOVE and self._pan_start:
            self.offset[0] = (
                self._pan_off_start[0] - (sx - self._pan_start[0]) / self.scale
            )
            self.offset[1] = (
                self._pan_off_start[1] - (sy - self._pan_start[1]) / self.scale
            )
            self._dirty = True
        elif event == cv2.EVENT_LBUTTONUP:
            if (
                self._pan_start
                and abs(sx - self._pan_start[0]) + abs(sy - self._pan_start[1]) <= 10
            ):
                idx = self._box_at(imgx, imgy)
                if idx is not None:
                    if self._selected == idx:
                        self.boxes[idx].ok = not self.boxes[idx].ok
                    else:
                        self._selected = idx
                else:
                    self._selected = None
            self._pan_start, self._dirty = None, True
        elif event == cv2.EVENT_RBUTTONDOWN:
            self._draw_start = self._draw_cur = (imgx, imgy)
        elif event == cv2.EVENT_MOUSEMOVE and self._draw_start:
            self._draw_cur, self._dirty = (imgx, imgy), True
        elif event == cv2.EVENT_RBUTTONUP and self._draw_start:
            x1, y1 = int(min(self._draw_start[0], imgx)), int(
                min(self._draw_start[1], imgy)
            )
            x2, y2 = int(max(self._draw_start[0], imgx)), int(
                max(self._draw_start[1], imgy)
            )
            if x2 - x1 >= self.min_w and y2 - y1 >= self.min_h:
                self.boxes.append(Box(x1, y1, x2 - x1, y2 - y1, manual=True))
                self._selected = len(self.boxes) - 1
            self._draw_start = self._draw_cur = None
            self._dirty = True

    def run(self) -> bool:
        """Uruchamia pętlę zdarzeń GUI. Zwraca True, jeśli użytkownik wybrał wyjście z zapisem."""
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW, self.win_w, self.win_h)
        cv2.setMouseCallback(WINDOW, self._mouse)
        self._fit()
        while True:
            if self._dirty:
                cv2.imshow(WINDOW, self._render())
                self._dirty = False
            raw_key = cv2.waitKeyEx(15)
            if raw_key == -1:
                continue
            key = raw_key & 0xFF
            if self._list_active:
                if key == KEY_ESC or key == ord("l") or key == ord("L"):
                    self._list_active = False
                elif raw_key == KEY_UP:
                    self._list_idx = max(0, self._list_idx - 1)
                elif raw_key == KEY_DOWN:
                    self._list_idx = min(len(self.all_files) - 1, self._list_idx + 1)
                elif key in KEY_ENTER:
                    self.requested_file = self.all_files[self._list_idx]
                    self.saved = True
                    break
                self._dirty = True
                continue
            if self._selected is not None:
                if key == KEY_ESC:
                    self._selected = None
                elif key in KEY_ENTER:
                    self.boxes[self._selected].ok = True
                    self._selected = (
                        self._selected + 1
                        if self._selected + 1 < len(self.boxes)
                        else None
                    )
                elif raw_key in KEY_DELETE_CODES:
                    self._remove_box(self._selected)
                elif key in KEY_BACKSPACE:
                    if self.boxes[self._selected].label:
                        self.boxes[self._selected].label = self.boxes[
                            self._selected
                        ].label[:-1]
                    else:
                        self._remove_box(self._selected)
                else:
                    char = self._decode_key(raw_key)
                    if char and char.isprintable():
                        self.boxes[self._selected].label += char
                self._dirty = True
            else:
                if key in (KEY_ESC, ord("q"), ord("Q")):
                    break
                elif key in KEY_ENTER:
                    self.saved = True
                    break
                elif key == ord("l") or key == ord("L"):
                    self._list_active = True
                    self._dirty = True
                elif key in (ord("1"), ord("!")):
                    [setattr(b, "ok", True) for b in self.boxes]
                    self._dirty = True
                elif key in (ord("2"), ord("@")):
                    [setattr(b, "ok", False) for b in self.boxes]
                    self._dirty = True
                elif key in (ord("f"), ord("F")):
                    self._fit(True)
                    self._dirty = True
                elif key in (ord("0"), ord(")")):
                    self.scale, self.offset = 1.0, [0.0, 0.0]
                    self._dirty = True
        cv2.destroyAllWindows()
        return self.saved

    @staticmethod
    def _decode_key(raw_key: int) -> str:
        """Dekoduje kod klawisza uwzględniając polskie znaki diakrytyczne (CP1250)."""
        pl_map = {
            185: "ą",
            230: "ć",
            234: "ę",
            179: "ł",
            241: "ń",
            243: "ó",
            156: "ś",
            159: "ź",
            191: "ż",
            161: "Ą",
            198: "Ć",
            202: "Ę",
            163: "Ł",
            209: "Ń",
            211: "Ó",
            140: "Ś",
            143: "Ź",
            175: "Ż",
        }
        if raw_key in pl_map:
            return pl_map[raw_key]
        if raw_key < 256:
            try:
                return bytes([raw_key]).decode("cp1250")
            except:
                pass
        try:
            return chr(raw_key)
        except:
            return ""

    def get_result(self) -> list[dict]:
        """Zwraca listę wszystkich ramek (również tych usuniętych) w formacie słownikowym."""
        return [b.to_dict() for b in self.boxes] + [
            b.to_dict() for b in self.deleted_boxes
        ]
