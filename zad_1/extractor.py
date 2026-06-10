import os
import docx
import pdfplumber
import openpyxl
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

class SmartExtractor:
    """Narzędzia do wyciągania tekstu z plików Word, PDF, Excel i TXT.
    
    _SUPPORTED - lista obsługiwanych rozszerzeń plików
    """
    _SUPPORTED = {"txt", "docx", "xlsx", "xls", "pdf"}

    @staticmethod
    def extract_text(path: str, ftype: str) -> str:
        """Wybiera odpowiednią metodę czytania pliku na podstawie rozszerzenia."""
        ft = (ftype or "").lower().strip(".")
        try:
            if ft == "txt":
                return SmartExtractor._txt(path)
            if ft == "docx":
                return SmartExtractor._docx(path)
            if ft in ("xlsx", "xls"):
                return SmartExtractor._xlsx(path)
            if ft == "pdf":
                return SmartExtractor._pdf(path)
            return SmartExtractor._txt(path)
        except Exception as e:
            print(f"Błąd ekstrakcji {path}: {e}")
            return ""

    @staticmethod
    def _txt(p: str) -> str:
        with open(p, "r", encoding="utf-8-sig", errors="ignore") as f:
            return f.read()

    @staticmethod
    def _docx(p: str) -> str:
        doc = docx.Document(p)
        return "\n".join(para.text for para in doc.paragraphs)

    @staticmethod
    def _xlsx(p: str) -> str:
        wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
        lines = []
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                line = " ".join(str(c) for c in row if c is not None)
                if line:
                    lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _pdf(p: str) -> str:
        with pdfplumber.open(p) as pdf:
            return "\n".join(filter(None, (page.extract_text() for page in pdf.pages)))

def _extract_worker(args: Tuple) -> Tuple[str, str, str, str]:
    """Pracownik wczytujący pojedynczy plik."""
    path, file_name, file_type = args
    text = SmartExtractor.extract_text(path, file_type)
    return text, path, file_name, file_type

def load_texts_parallel(files: List[Tuple[str, str]], file_root: str, workers: int = 4) -> List[Tuple[str, str, str]]:
    """Wczytuje wiele plików naraz przy użyciu wielu wątków."""
    tasks = []
    for file_name, file_type in files:
        path = os.path.join(file_root, file_name)
        if os.path.exists(path):
            tasks.append((path, file_name, file_type))
        else:
            print(f"Plik nie istnieje: {path}")

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_extract_worker, t): t for t in tasks}
        for future in as_completed(futures):
            try:
                text, _, file_name, file_type = future.result()
                if text.strip():
                    results.append((text, file_name, file_type))
                    print(f"Wczytano: {file_name}")
            except Exception as e:
                print(f"Błąd wczytywania: {e}")

    return results
