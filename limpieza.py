#imprimir las 10 primeras filas de la tabla Demanda
import sys
import pandas as pd
from sqlalchemy import create_engine, text
SERVER   = "udcserver2025.database.windows.net"
DATABASE = "grupo_1"
USER     = "ugrupo1"
PASSWORD = "HK9WXIJaBp2Q97haePdY"
ENGINE_URL = (
    f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER}:1433/{DATABASE}"
    "?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no"
)   
engine = create_engine(ENGINE_URL, pool_pre_ping=True)
tabla = "dbo.Demanda"
with engine.connect() as con:
    df = pd.read_sql(text(f"SELECT TOP 10 * FROM {tabla}"), con)
print(df)
