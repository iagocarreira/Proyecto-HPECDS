import os, pyodbc, logging
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from dateutil import tz
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
    return pyodbc.connect(
        f"DRIVER={driver};SERVER={server};PORT={port};DATABASE={database};UID={username};PWD={password}"
    )


# ----------------- CREAR TABLAS -----------------
def create_table():
    """Crea la tabla histórica (A) si no existe."""
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
        cursor.close()
        conn.close()


def create_week_table():
    """Crea la tabla temporal semanal (B) si no existe."""
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'demanda_peninsula_semana')
        BEGIN
            CREATE TABLE dbo.demanda_peninsula_semana (
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
        logging.info("Tabla 'demanda_peninsula_semana' lista o ya existente.")
    except Exception as e:
        logging.error(f"Error creando tabla semanal: {e}")
    finally:
        cursor.close()
        conn.close()


# ----------------- CARGAR DATOS -----------------
def load_to_sql(df, table_name="demanda_peninsula"):
    """Carga datos en SQL Server usando MERGE para evitar duplicados por fecha."""
    try:
        conn = get_conn()
        cursor = conn.cursor()

        rows_to_insert = []
        for _, row in df.iterrows():
            fecha = row.get("fecha")
            if pd.isna(fecha):
                continue

            rows_to_insert.append((
                fecha,
                row.get("datetime_utc"),
                row.get("valor_real"),
                row.get("valor_previsto"),
                row.get("valor_programado"),
                row.get("hora"),
                row.get("dia_semana"),
                int(row.get("es_fin_semana")) if pd.notna(row.get("es_fin_semana")) else None,
                row.get("error_absoluto"),
                row.get("error_relativo_pct")
            ))

        if rows_to_insert:
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
                        source.valor_programado, source.hora, source.dia_semana, source.es_fin_semana,
                        source.error_absoluto, source.error_relativo_pct);
            """
            cursor.executemany(sql, rows_to_insert)
            conn.commit()
            logging.info(f"{len(rows_to_insert)} filas procesadas en '{table_name}'.")
        else:
            logging.info("No hay filas para procesar.")

    except Exception as e:
        logging.error(f"Error al cargar datos en SQL: {e}")
    finally:
        cursor.close()
        conn.close()


# ----------------- LIMPIEZA Y TRANSFERENCIA -----------------
def cleanup_week_table():
    """Elimina registros antiguos (más de 7 días) de la tabla semanal."""
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM dbo.demanda_peninsula_semana
            WHERE fecha < DATEADD(day, -7, GETDATE());
        """)
        conn.commit()
        logging.info("Limpieza completada: solo se conservan los últimos 7 días.")
    except Exception as e:
        logging.error(f"Error limpiando tabla semanal: {e}")
    finally:
        cursor.close()
        conn.close()


def transfer_week_to_main():
    """Transfiere datos de la tabla semanal (B) a la histórica (A) evitando duplicados."""
    try:
        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("""
            MERGE INTO dbo.demanda_peninsula AS target
            USING (
                SELECT * FROM dbo.demanda_peninsula_semana
                WHERE fecha >= DATEADD(day, -7, GETDATE())
            ) AS source
            ON target.fecha = source.fecha
            WHEN NOT MATCHED BY TARGET THEN
                INSERT (fecha, fecha_utc, valor_real, valor_previsto, valor_programado,
                        hora, dia_semana, es_fin_semana, error_absoluto, error_relativo_pct)
                VALUES (source.fecha, source.fecha_utc, source.valor_real, source.valor_previsto,
                        source.valor_programado, source.hora, source.dia_semana,
                        source.es_fin_semana, source.error_absoluto, source.error_relativo_pct);

            DELETE FROM dbo.demanda_peninsula_semana
            WHERE fecha >= DATEADD(day, -7, GETDATE());
        """)
        conn.commit()
        logging.info("Datos transferidos a la tabla histórica sin duplicados.")
    except Exception as e:
        logging.error(f"Error transfiriendo datos: {e}")
    finally:
        cursor.close()
        conn.close()


# ----------------- LOG DE EJECUCIÓN -----------------
def log_etl_execution(start_time, end_time, status, message=None, dag_name=None):
    """Guarda la información de ejecución de la ETL (hora local Madrid)."""
    try:
        tz_madrid = tz.gettz("Europe/Madrid")
        if start_time.tzinfo:
            start_time = start_time.astimezone(tz_madrid).replace(tzinfo=None)
        if end_time and end_time.tzinfo:
            end_time = end_time.astimezone(tz_madrid).replace(tzinfo=None)

        conn = get_conn()
        cursor = conn.cursor()
        cursor.execute("""
        IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'etl_log')
        BEGIN
            CREATE TABLE dbo.etl_log (
                id INT IDENTITY(1,1) PRIMARY KEY,
                dag_name NVARCHAR(100) NULL,
                start_time DATETIME NOT NULL,
                end_time DATETIME NULL,
                duration_seconds FLOAT NULL,
                status NVARCHAR(50) NOT NULL,
                message NVARCHAR(MAX) NULL
            );
        END
        """)
        conn.commit()

        duration = None
        if start_time and end_time:
            duration = (end_time - start_time).total_seconds()

        cursor.execute(
            "INSERT INTO dbo.etl_log (dag_name, start_time, end_time, duration_seconds, status, message) VALUES (?, ?, ?, ?, ?, ?)",
            (dag_name, start_time, end_time, duration, status, message)
        )
        conn.commit()
        logging.info("Registro de ejecución guardado en etl_log (hora local Madrid).")
    except Exception as e:
        logging.error(f"Error al registrar ejecución ETL: {e}")
    finally:
        cursor.close()
        conn.close()

