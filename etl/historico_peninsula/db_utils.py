import os, pyodbc, logging
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Configuración SQL
server = os.getenv("SQL_SERVER")
database = os.getenv("SQL_DATABASE")
username = os.getenv("SQL_USER")
password = os.getenv("SQL_PASSWORD")
driver = "{ODBC Driver 18 for SQL Server}"
port = os.getenv("SQL_PORT", "1433")

# ----------------- CONEXIÓN -----------------
def get_conn():
    """Devuelve una conexión activa a SQL Server."""
    conn_str = (
        f"DRIVER={driver};"
        f"SERVER={server},{port};"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        f"TrustServerCertificate=yes;"
        f"Connection Timeout=15;"
    )
    return pyodbc.connect(conn_str)

# ----------------- CREAR TABLA -----------------
def create_table():
    """Crea la tabla demanda_peninsula si no existe, con columnas derivadas incluidas."""
    conn = None
    cursor = None
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'demanda_peninsula')
        BEGIN
            CREATE TABLE dbo.demanda_peninsula (
                fecha DATETIME NOT NULL PRIMARY KEY,
                fecha_utc DATETIME NOT NULL,
                valor_real FLOAT NULL,
                valor_previsto FLOAT NULL,
                valor_programado FLOAT NULL,
                hora INT NULL,
                dia_semana NVARCHAR(20) NULL,
                es_fin_semana BIT NULL,
                error_absoluto FLOAT NULL,
                error_relativo_pct FLOAT NULL
            );
        END
        """)
        conn.commit()
        logging.info("Tabla 'demanda_peninsula' lista o ya existente.")
    except Exception as e:
        logging.error(f"Error creando tabla: {e}")
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

# ----------------- CARGAR DATOS -----------------
def load_to_sql(df, table_name="demanda_peninsula"):
    """Carga datos en SQL Server en bloques usando MERGE, con limpieza de tipos."""
    conn = None
    cursor = None
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.fast_executemany = True

        # Limpieza de tipos
        df = df.copy()
        df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
        df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], errors="coerce")
        for c in ["valor_real", "valor_previsto", "valor_programado", "error_absoluto", "error_relativo_pct"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "hora" in df.columns:
            df["hora"] = pd.to_numeric(df["hora"], errors="coerce").fillna(-1).astype(int)
        if "es_fin_semana" in df.columns:
            df["es_fin_semana"] = df["es_fin_semana"].apply(
                lambda x: 1 if str(x).lower() in ["true", "1", "yes"] else
                          0 if str(x).lower() in ["false", "0", "no"] else None
            )

        # Filtramos filas válidas
        df = df[df["fecha"].notna() & df["datetime_utc"].notna()]
        if df.empty:
            logging.warning("No hay filas válidas para insertar.")
            return

        rows_to_insert = [
            (
                row["fecha"],
                row["datetime_utc"],
                row.get("valor_real"),
                row.get("valor_previsto"),
                row.get("valor_programado"),
                int(row.get("hora")) if pd.notna(row.get("hora")) else None,
                row.get("dia_semana"),
                row.get("es_fin_semana"),
                row.get("error_absoluto"),
                row.get("error_relativo_pct"),
            )
            for _, row in df.iterrows()
        ]

        sql = f"""
        MERGE INTO dbo.{table_name} AS target
        USING (VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)) 
        AS source (fecha, fecha_utc, valor_real, valor_previsto, valor_programado,
                   hora, dia_semana, es_fin_semana, error_absoluto, error_relativo_pct)
        ON target.fecha = source.fecha
        WHEN NOT MATCHED THEN
            INSERT (fecha, fecha_utc, valor_real, valor_previsto, valor_programado,
                    hora, dia_semana, es_fin_semana, error_absoluto, error_relativo_pct)
            VALUES (source.fecha, source.fecha_utc, source.valor_real, source.valor_previsto,
                    source.valor_programado, source.hora, source.dia_semana,
                    source.es_fin_semana, source.error_absoluto, source.error_relativo_pct);
        """

        batch_size = 500
        total = len(rows_to_insert)
        for i in range(0, total, batch_size):
            batch = rows_to_insert[i:i + batch_size]
            try:
                cursor.executemany(sql, batch)
                conn.commit()
                logging.info(f"Lote {i // batch_size + 1}: {len(batch)} filas procesadas ({i + len(batch)}/{total})")
            except Exception as e:
                logging.error(f"❌ Error en lote {i // batch_size + 1}: {e}")
                conn.rollback()

        logging.info(f"✅ Procesadas {total} filas en total (solo nuevas insertadas).")

    except Exception as e:
        logging.error(f"Error al cargar datos en SQL: {e}")
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()

