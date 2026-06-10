import re
import spacy
from datetime import datetime
from collections import defaultdict
from difflib import SequenceMatcher
from functools import lru_cache
from typing import List, Dict, Tuple
from config import Config

@lru_cache(maxsize=4096) # zapamiętuje wyniki podobieństwa, żeby nie liczyć dwa razy
def _cached_similarity(a: str, b: str) -> float:
    """Szybkie porównywanie podobieństwa tekstów."""
    return SequenceMatcher(None, a, b).ratio()

class NLPProcessor:
    """Przetwarzanie tekstu w celu znalezienia imion i nazwisk (spaCy).
    
    _HYPHEN_RE - usuwa spacje wokół myślnika
    _INVALID_RE - usuwa liczby i znaki specjalne
    """
    
    _HYPHEN_RE = re.compile(r"\s*-\s*")
    _INVALID_RE = re.compile(r"[0-9@#]")

    def __init__(self, config: Config):
        self.config = config
        print("Ładowanie modelu języka polskiego...")
        self.nlp = spacy.load("pl_core_news_lg", disable=["parser", "senter"]) # wyłączamy niepotrzebne moduły, żeby było szybciej

    def _lemmatize_name(self, ent) -> str:
        """Sprowadza imiona i nazwiska do formy podstawowej. 
        
        ent - obiekt zidentyfikowany przez spaCy
        """
        parts = []
        for i, token in enumerate(ent):
            lemma = token.lemma_.strip().capitalize()
            parts.append(lemma)
        return self._HYPHEN_RE.sub("-", " ".join(parts))

    def _is_match(self, lone_val: str, lone_parts: List[str], full_last: str, full_parts: List[str]) -> bool:
        """Sprawdza czy samo nazwisko pasuje do kogoś istniejacego już w cache."""
        thr = self.config.LONE_THRESHOLD
        if _cached_similarity(lone_val, full_last) > thr:
            return True
        return any(_cached_similarity(lp, fp) > thr for lp in lone_parts for fp in full_parts)

    def _extract_persons(self, doc) -> List[Dict]:
        """Wyciąga osoby znalezione w tekście."""
        raw_map: Dict[str, set] = defaultdict(set)
        for ent in doc.ents:
            if ent.label_ != "persName":
                continue
            orig = ent.text.strip()
            if len(orig) < 3 or self._INVALID_RE.search(orig):
                continue
            lemma = self._lemmatize_name(ent)
            raw_map[lemma].add(orig)

        persons = []
        for lemma, forms in raw_map.items():
            parts = lemma.split()
            persons.append({
                "full_name": lemma,
                "first_name": parts[0] if len(parts) > 1 else "",
                "last_name": parts[-1] if len(parts) > 1 else lemma,
                "forms": forms,
            })
        return persons

    def _merge_lone_entries(self, full_entries: List[Dict], lone_entries: List[Dict]) -> List[Dict]:
        """Dopasowuje same nazwiska do pełnych rekordów (imię + nazwisko)."""
        used_lone: set = set()
        full_cache = [(p["last_name"].lower(), p["last_name"].lower().split("-")) for p in full_entries]

        for lone in lone_entries:
            lone_val = lone["last_name"].lower()
            lone_parts = lone_val.split("-")
            for full, (full_last, full_parts) in zip(full_entries, full_cache):
                if self._is_match(lone_val, lone_parts, full_last, full_parts):
                    full["forms"].update(lone["forms"])
                    used_lone.add(lone["full_name"])
                    break
        return full_entries + [l for l in lone_entries if l["full_name"] not in used_lone]

    def process_batch(self, texts_meta: List[Tuple[str, str, str]]) -> List[Dict]:
        """Przetwarza całą grupę tekstów naraz."""
        texts = [t for t, _, _ in texts_meta]
        meta = [(fn, ft) for _, fn, ft in texts_meta]
        all_results = []
        
        for doc, (file_name, file_type) in zip(self.nlp.pipe(texts, batch_size=self.config.NLP_BATCH_SIZE), meta):
            persons = self._extract_persons(doc)
            full_entries = [p for p in persons if p["first_name"]]
            lone_entries = [p for p in persons if not p["first_name"]]
            final_list = self._merge_lone_entries(full_entries, lone_entries)

            now = datetime.now()
            for p in final_list:
                all_results.append({
                    "full_name": p["full_name"],
                    "first_name": p["first_name"],
                    "last_name": p["last_name"],
                    "forms": ", ".join(sorted(p["forms"])),
                    "forms_count": len(p["forms"]),
                    "searched_at": now,
                    "file_name": file_name,
                    "file_type": file_type,
                })
        return all_results
