import re

_RE_SANITIZE = re.compile(r"[^a-zA-Z0-9_\-\.]")


def strip_pl(txt: str) -> str:
    """
    Zastępuje polskie znaki ich odpowiednikami ASCII.
    Użyteczne dla OpenCV, który domyślnie nie renderuje poprawnie UTF-8.
    """
    return txt.translate(str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ"))


def sanitize_filename(text: str) -> str:
    """
    Oczyszcza ciąg tekstowy, aby mógł być bezpiecznie użyty jako nazwa pliku.
    """
    return _RE_SANITIZE.sub("_", text.strip())
