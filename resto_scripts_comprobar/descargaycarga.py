import os, requests, pandas as pd
from datetime import datetime, timedelta
from dateutil import tz
import pyodbc

# Configuración de la API
ESIOS_TOKEN = os.environ.get("ESIOS_TOKEN") or "ddcbd54fa41b494243f3a6094062af3f41a4675956a8f50a2b92b80bd0fbc71a"
HEADERS = {
    "Accept": "application/json; application/vnd.esios-api-v1+json",
    "Content-Type": "application/json",
    "x-api-key": ESIOS_TOKEN
}

# Indicadores esenciales
INDICATORS = {
    "real": 1293,
    "previsto": 544,
    "programado": 2053
}

# Conexión SQL Server
server = 'udcserver2025.database.windows.net'
database = 'grupo_1'
username = 'ugrupo1'
password = 'HK9WXIJaBp2Q97haePdY'
driver = '{ODBC Driver 18 for SQL Server}'

# Conversión fechas
def convert_to_datetime(date_str):
    if isinstance(date_str, str):
        if 'Z' in date_str:
            date_str = date_str.replace('Z', '')
        elif '+' in date_str:
            date_str = date_str.split("+")[0]
        try:
            return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%f')
        except ValueError:
            return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
    return date_str

# Crear tabla
def create_table():
    conn = pyodbc.connect(f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}')
    cursor = conn.cursor()
    cursor.execute("""
    IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'demanda_total')
    BEGIN
        CREATE TABLE dbo.demanda_total (
            fecha DATETIME NOT NULL,
            fecha_utc DATETIME NOT NULL,
            tz_time DATETIME NOT NULL,
            valor_real FLOAT NULL,
            geo_id_real INT NULL,
            geo_name_real NVARCHAR(100) NULL,
            valor_previsto FLOAT NULL,
            geo_id_previsto INT NULL,
            geo_name_previsto NVARCHAR(100) NULL,
            valor_programado FLOAT NULL,
            geo_id_programado INT NULL,
            geo_name_programado NVARCHAR(100) NULL
        );
    END
    """)
    conn.commit()
    cursor.close()
    conn.close()
    print("Tabla 'demanda_total' creada o ya existe.")

# Insertar en SQL
def load_to_sql(df, table_name="demanda_total"):
    # Asegurar floats válidos
    for col in ["valor_real", "valor_previsto", "valor_programado"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").where(pd.notna(df[col]), None)

    conn = pyodbc.connect(f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}')
    cursor = conn.cursor()
    cursor.execute(f"SELECT fecha FROM dbo.{table_name}")
    existing_dates = set([row[0] for row in cursor.fetchall()])

    rows_to_insert = []
    for _, row in df.iterrows():
        fecha = convert_to_datetime(row['datetime'])
        if fecha not in existing_dates:
            fecha_utc = convert_to_datetime(row['datetime_utc'])
            tz_time = convert_to_datetime(row['tz_time'])
            rows_to_insert.append((
                fecha, fecha_utc, tz_time,
                row.get("valor_real"), row.get("geo_id_real"), row.get("geo_name_real"),
                row.get("valor_previsto"), row.get("geo_id_previsto"), row.get("geo_name_previsto"),
                row.get("valor_programado"), row.get("geo_id_programado"), row.get("geo_name_programado"),
            ))

    if rows_to_insert:
        cursor.executemany(f"""
            INSERT INTO dbo.{table_name} 
            (fecha, fecha_utc, tz_time, valor_real, geo_id_real, geo_name_real,
             valor_previsto, geo_id_previsto, geo_name_previsto,
             valor_programado, geo_id_programado, geo_name_programado)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows_to_insert)
        conn.commit()
        print(f"Se insertaron {len(rows_to_insert)} filas nuevas en {table_name}.")
    else:
        print("No hay filas nuevas para insertar.")
    cursor.close()
    conn.close()

# Descargar API
def fetch_indicator(indicator_id, col_prefix, start_dt_iso, end_dt_iso):
    url = f"https://api.esios.ree.es/indicators/{indicator_id}"
    params = {"start_date": start_dt_iso, "end_date": end_dt_iso}
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    print("URL:", r.url, " ->", r.status_code)
    r.raise_for_status()
    values = r.json().get("indicator", {}).get("values", [])
    df = pd.DataFrame(values)
    df = df.rename(columns={
        "value": f"valor_{col_prefix}",
        "geo_id": f"geo_id_{col_prefix}",
        "geo_name": f"geo_name_{col_prefix}"
    })
    return df

if __name__ == "__main__":
    tz_madrid = tz.gettz("Europe/Madrid")
    hoy = datetime.now(tz_madrid).date()
    inicio = (hoy - timedelta(days=3)).isoformat()
    fin = hoy.isoformat()

    dfs = []
    for prefix, indicator_id in INDICATORS.items():
        df = fetch_indicator(indicator_id, prefix, inicio + "T00:00:00+02:00", fin + "T00:00:00+02:00")
        dfs.append(df[["datetime", "datetime_utc", "tz_time",
                       f"valor_{prefix}", f"geo_id_{prefix}", f"geo_name_{prefix}"]])

    # Merge progresivo por datetime
    merged = dfs[0]
    for d in dfs[1:]:
        merged = pd.merge(merged, d, on=["datetime", "datetime_utc", "tz_time"], how="outer")

    # Crear tabla e insertar
    create_table()
    load_to_sql(merged, "demanda_total")

    print("Proceso finalizado. Datos cargados en la BD.")
