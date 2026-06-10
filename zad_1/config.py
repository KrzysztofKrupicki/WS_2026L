from dataclasses import dataclass

@dataclass
class Config:
    """Klasa zawierająca wszystkie ustawienia programu - serwer, bazę danych i ścieżki.
    
    SERVER - nazwa serwera bazy danych
    DATABASE - nazwa bazy danych
    DRIVER - sterownik do bazy danych
    FILE_ROOT - ścieżka do folderu z plikami do sprawdzenia
    LONE_THRESHOLD - minimalne podobieństwo nazwiska (0.0 - 1.0)
    EXTRACTOR_WORKERS - liczba wątków do wczytywania plików
    NLP_BATCH_SIZE - ile dokumentów naraz przetwarza AI
    """
    SERVER: str = "DESKTOP-RBULOMS"
    DATABASE: str = "LAB_REPAIR"
    DRIVER: str = "{ODBC Driver 17 for SQL Server}"
    FILE_ROOT: str = r"\\Desktop-rbuloms\mssqlserver\LAB_DATA_Files\Files_Dir"
    LONE_THRESHOLD: float = 0.75 
    EXTRACTOR_WORKERS: int = 4 
    NLP_BATCH_SIZE: int = 16 
