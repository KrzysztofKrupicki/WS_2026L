from config import Config
from database import DatabaseRepository
from extractor import load_texts_parallel
from processor import NLPProcessor

def main():
    """Główna funkcja sterująca procesem wyszukiwania osób w dokumentach."""
    
    # Inicjalizacja konfiguracji i modułów
    cfg = Config()
    repo = DatabaseRepository(cfg)
    processor = NLPProcessor(cfg)

    # 1. Pobieramy listę plików z bazy danych
    print("Łączenie z bazą danych...")
    with repo.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name, file_type FROM dbo.Files WHERE is_directory = 0")
        files = cursor.fetchall()

    print(f"Znaleziono {len(files)} plików do przetworzenia.")

    # 2. Wczytujemy pliki z dysku (równolegle)
    print("Wczytywanie plików z dysku...")
    texts_meta = load_texts_parallel(
        files, cfg.FILE_ROOT, workers=cfg.EXTRACTOR_WORKERS
    )
    print(f"Wczytano {len(texts_meta)} niepustych plików.")

    if not texts_meta:
        print("Brak tekstów do przetworzenia.")
        return

    # 3. Przetwarzamy teksty za pomocą NLP (sztuczna inteligencja)
    print("Przetwarzanie dokumentów przez AI (szukanie osób)...")
    all_found = processor.process_batch(texts_meta)

    # 4. Zapisujemy wyniki w bazie SQL
    if all_found:
        print(f"Znaleziono {len(all_found)} rekordów (osób). Zapisywanie do bazy...")
        repo.save_results(all_found)
    else:
        print("Nie znaleziono żadnych osób w dokumentach.")

    print("--- GOTOWE ---")

if __name__ == "__main__":
    main()
