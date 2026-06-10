import pyodbc
from typing import List, Dict
from config import Config

class DatabaseRepository:
    """Klasa odpowiedzialna za całą komunikację z bazą danych SQL Server."""
    
    def __init__(self, config: Config):
        self.conn_str = (
            f"DRIVER={config.DRIVER};SERVER={config.SERVER};"
            f"DATABASE={config.DATABASE};Trusted_Connection=yes;"
        )

    def get_connection(self):
        """Tworzy i zwraca aktywne połączenie z bazą danych."""
        return pyodbc.connect(self.conn_str)

    def save_results(self, results: List[Dict]):
        """Zapisuje listę znalezionych osób do tabeli PersonSearchResults."""
        if not results:
            return

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.fast_executemany = True
            # Usuwamy stare wyniki
            cursor.execute("DELETE FROM dbo.PersonSearchResults")

            query = """
                INSERT INTO dbo.PersonSearchResults
                (file_name, file_type, full_name, first_name, last_name, forms, forms_count, searched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """
            params = [
                (
                    r["file_name"],
                    r["file_type"],
                    r["full_name"],
                    r["first_name"],
                    r["last_name"],
                    r["forms"],
                    r["forms_count"],
                    r["searched_at"],
                )
                for r in results
            ]
            cursor.executemany(query, params)
            conn.commit()
