import os, pyodbc, logging
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from dateutil import tz

load_dotenv()

# ----------------- CONFIGURACIÓN SQL -----------------
server = os.getenv("SQL_SERVER")
database = os.getenv("SQL_DATABASE")
username = os.getenv("SQL_USER")
password = os.getenv("SQL_PASSWORD")
driver = "{ODBC Driver 18 for SQL Server}"
port = os.getenv("SQL_PORT", "1433")


def get_conn():
    return pyodbc.connect(
        f"DRIVER={driver};SERVER={server};PORT={port};DATABASE={database};UID={username};PWD={password}"
    )


# ----------------- CREACIÓN DE TABLAS -----------------
def create_main_and_week_tables():
    """Crea las tablas histórica (A) y semanal (B) si no existen."""
    try:
        conn = get_conn()
        cursor = conn.cursor()

        # Tabla histórica
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

        # Tabla semanal (últimos 7 días)
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
        logging.info("Tablas 'demanda_peninsula' y 'demanda_peninsula_semana' listas o ya existentes.")
    except Exception as e:
        logging.error(f"Error creando tablas: {e}")
    finally:
        cursor.close()
        conn.close()


# ----------------- INSERCIÓN -----------------
def load_to_sql(df, table_name="demanda_peninsula_semana"):
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

        if not rows_to_insert:
            logging.info("No hay filas para insertar.")
            return

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
        logging.info(f"{len(rows_to_insert)} filas nuevas procesadas en '{table_name}'.")

    except Exception as e:
        logging.error(f"Error al cargar datos: {e}")
    finally:
        cursor.close()
        conn.close()


# ----------------- TRANSFERENCIA Y MANTENIMIENTO -----------------
def transfer_oldest_from_week_to_main():
    """
    Transfiere a la tabla histórica (A) todos los registros de la tabla semanal (B)
    que correspondan a días completos anteriores al rango de los últimos 7 días naturales.
    """
    try:
        conn = get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            DECLARE @fecha_max DATE;
            SELECT @fecha_max = CAST(MAX(fecha) AS DATE) FROM dbo.demanda_peninsula_semana;

            IF @fecha_max IS NOT NULL
            BEGIN
                -- Día más antiguo que debe conservarse (para tener 7 días: max, max-1, ..., max-6)
                DECLARE @fecha_min_a_conservar DATE = DATEADD(DAY, -6, @fecha_max);

                -- 1) Insertar en A todo lo ANTERIOR a ese mínimo a conservar (== días que sobran)
                INSERT INTO dbo.demanda_peninsula (
                    fecha, fecha_utc, valor_real, valor_previsto, valor_programado,
                    hora, dia_semana, es_fin_semana, error_absoluto, error_relativo_pct
                )
                SELECT s.fecha, s.fecha_utc, s.valor_real, s.valor_previsto, s.valor_programado,
                       s.hora, s.dia_semana, s.es_fin_semana, s.error_absoluto, s.error_relativo_pct
                FROM dbo.demanda_peninsula_semana s
                WHERE CAST(s.fecha AS DATE) < @fecha_min_a_conservar
                AND NOT EXISTS (
                    SELECT 1 FROM dbo.demanda_peninsula a WHERE a.fecha = s.fecha
                );
            END
        """)

        conn.commit()
        logging.info("Transferencia a histórica (A) realizada para los días anteriores al rango de 7 días.")
    except Exception as e:
        logging.error(f"Error transfiriendo registros antiguos: {e}")
    finally:
        cursor.close()
        conn.close()

def ensure_week_table_has_7_days():
    """Mantiene solo los últimos 7 días naturales completos en la tabla semanal (B)."""
    try:
        conn = get_conn()
        cursor = conn.cursor()

        cursor.execute("""
            DECLARE @fecha_max DATE;
            SELECT @fecha_max = CAST(MAX(fecha) AS DATE) FROM dbo.demanda_peninsula_semana;

            IF @fecha_max IS NOT NULL
            BEGIN
                -- Conservar: @fecha_max, ..., @fecha_max-6
                DELETE FROM dbo.demanda_peninsula_semana
                WHERE CAST(fecha AS DATE) < DATEADD(DAY, -6, @fecha_max);
            END
        """)

        conn.commit()
        logging.info("Tabla semanal (B) ajustada a los últimos 7 días naturales.")
    except Exception as e:
        logging.error(f"Error ajustando días en la tabla semanal: {e}")
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

        duration = (end_time - start_time).total_seconds() if start_time and end_time else None

        cursor.execute("""
            INSERT INTO dbo.etl_log (dag_name, start_time, end_time, duration_seconds, status, message)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (dag_name, start_time, end_time, duration, status, message))

        conn.commit()
        logging.info("Registro de ejecución guardado en etl_log.")
    except Exception as e:
        logging.error(f"Error al registrar ejecución ETL: {e}")
    finally:
        cursor.close()
        conn.close()
