import logging
import pyodbc
from config.settings import Settings

class SQLConnectionManager:
    """Gestor de conexión con SQL Server."""
    def __init__(self):
        self.driver = "{ODBC Driver 18 for SQL Server}"
        self.conn_str = (
            f"DRIVER={self.driver};"
            f"SERVER={Settings.SQL_SERVER},{Settings.SQL_PORT};"
            f"DATABASE={Settings.SQL_DATABASE};"
            f"UID={Settings.SQL_USER};PWD={Settings.SQL_PASSWORD};"
            f"TrustServerCertificate=yes;"
            f"Encrypt=yes;"
            f"Connection Timeout=60;"
            f"Login Timeout=60;"

        )

    def connect(self):
        try:
            return pyodbc.connect(self.conn_str)
        except Exception as e:
            logging.error(f" Error al conectar con SQL Server: {e}")
            raise
