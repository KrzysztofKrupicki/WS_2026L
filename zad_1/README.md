# Automatyzacja Ekstrakcji Danych z Dokumentów (Zadanie 1)

## Opis Projektu
System zaprojektowany do automatycznej identyfikacji, ekstrakcji oraz normalizacji danych osobowych z rozproszonych zbiorów dokumentów (PDF, DOCX, XLSX, TXT). Rozwiązanie optymalizuje proces przeszukiwania dużych wolumenów danych, eliminując konieczność ręcznej analizy plików.

Program został podzielony na moduły, co ułatwia jego rozwój i utrzymanie.

## Struktura Projektu

- **`zad_1.py`** – Główny punkt startowy aplikacji. Koordynuje pracę pozostałych modułów.
- **`config.py`** – Plik konfiguracyjny (dane serwera, bazy danych, parametry NLP).
- **`database.py`** – Moduł obsługi bazy danych SQL Server (zapisywanie wyników).
- **`extractor.py`** – Moduł odpowiedzialny za wyciąganie tekstu z plików o różnych formatach (PDF, Word, Excel).
- **`processor.py`** – Moduł przetwarzania języka naturalnego (NLP) szukający imion i nazwisk.

## Kluczowe Funkcjonalności
*   **Wielofunkcyjna Ekstrakcja:** Obsługa najpopularniejszych formatów dokumentów biurowych.
*   **Zaawansowana Analiza NLP:** Inteligentne rozpoznawanie osób (NER) przy użyciu biblioteki `spaCy`.
*   **Normalizacja Językowa:** Sprowadzanie nazwisk do formy podstawowej (mianownika).
*   **Łączenie Rekordów:** Dopasowywanie samych nazwisk do pełnych nazwisk znalezionych w dokumencie.
*   **Integracja SQL:** Zautomatyzowany zapis do bazy danych SQL Server.

## Wymagania Techniczne
*   Środowisko Python 3.10+
*   Biblioteki: `spacy`, `pyodbc`, `pdfplumber`, `python-docx`, `openpyxl`.
*   Zainstalowany model języka polskiego: `python -m spacy download pl_core_news_lg`.

## Jak uruchomić
1.  Skonfiguruj dane połączenia w pliku `config.py`.
2.  Zainstaluj wymagane biblioteki: `pip install -r requirements.txt`.
3.  Uruchom główny skrypt: `python zad_1.py`.
