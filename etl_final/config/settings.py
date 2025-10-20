import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    SQL_SERVER = os.getenv("SQL_SERVER")
    SQL_DATABASE = os.getenv("SQL_DATABASE")
    SQL_USER = os.getenv("SQL_USER")
    SQL_PASSWORD = os.getenv("SQL_PASSWORD")
    SQL_PORT = os.getenv("SQL_PORT", "1433")
    ESIOS_TOKEN = os.getenv("ESIOS_TOKEN")
    TIMEZONE = os.getenv("TIMEZONE", "Europe/Madrid")
    DAG_NAME = os.getenv("DAG_NAME", "manual_run")
