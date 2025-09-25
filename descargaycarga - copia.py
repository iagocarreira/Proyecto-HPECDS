import os, requests, pandas as pd
from datetime import datetime , timedelta
from dateutil import tz
import pyodbc

# Configuración de la API
ESIOS_TOKEN = os.environ.get("ESIOS_TOKEN") or "ddcbd54fa41b494243f3a6094062af3f41a4675956a8f50a2b92b80bd0fbc71a"
HEADERS = {
    "Accept": "application/json; application/vnd.esios-api-v1+json",
    "Content-Type": "application/json",
    "x-api-key": ESIOS_TOKEN   
}
INDICATOR_ID = 1293

# Conexión a la base de datos SQL Server
server = 'udcserver2025.database.windows.net'
database = 'grupo_1'
username = 'ugrupo1'
password = 'HK9WXIJaBp2Q97haePdY'
driver = '{ODBC Driver 18 for SQL Server}' #Hay que instalar el driver ODBC 18

# Función para formatear las fechas adecuadamente
def convert_to_datetime(date_str):
    if isinstance(date_str, str):
        if 'Z' in date_str:
            date_str = date_str.replace('Z', '')
        elif '+' in date_str:
            date_str = date_str.split("+")[0]
        try:
            # Primero intentamos con milisegundos
            return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%f')
        except ValueError:
            # Si falla, intentamos sin milisegundos
            return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S')
    return date_str


# Función para crear la tabla si no existe
def create_table():
    conn = pyodbc.connect(f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}')
    cursor = conn.cursor()

    # Crear la tabla si no existe
    cursor.execute("""
    IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'demanda')
    BEGIN
        CREATE TABLE dbo.demanda (
            fecha DATETIME NOT NULL,
            fecha_utc DATETIME NOT NULL,
            tz_time DATETIME NOT NULL,
            valor FLOAT NOT NULL,
            geo_id INT NOT NULL,
            geo_name NVARCHAR(100) NOT NULL
        );
    END
    """)
    conn.commit()
    cursor.close()
    conn.close()
    print("Tabla 'demanda' creada o ya existe.")

# Función para cargar los datos en la base de datos
def load_to_sql(df, table_name):
    conn = pyodbc.connect(f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}')
    cursor = conn.cursor()

    # Paso 1: obtener las fechas existentes
    cursor.execute(f"SELECT fecha FROM {table_name}")
    existing_dates = set([row[0] for row in cursor.fetchall()])

    # Paso 2: filtrar el DataFrame para insertar solo las fechas nuevas
    rows_to_insert = []
    for index, row in df.iterrows():
        fecha = convert_to_datetime(row['datetime'])
        if fecha not in existing_dates:
            fecha_utc = convert_to_datetime(row['datetime_utc'])
            tz_time = convert_to_datetime(row['tz_time'])
            valor = row['value']
            geo_id = row['geo_id']
            geo_name = row['geo_name']
            rows_to_insert.append((fecha, fecha_utc, tz_time, valor, geo_id, geo_name))

    # Paso 3: insertar solo las filas nuevas
    if rows_to_insert:
        cursor.executemany(f"""
            INSERT INTO {table_name} (fecha, fecha_utc, tz_time, valor, geo_id, geo_name)
            VALUES (?, ?, ?, ?, ?, ?)
        """, rows_to_insert)
        conn.commit()
        print(f"Se insertaron {len(rows_to_insert)} filas nuevas en {table_name}.")
    else:
        print("No hay filas nuevas para insertar.")

    cursor.close()
    conn.close()

# Función para obtener los datos de la API
def fetch_demanda(start_dt_iso: str, end_dt_iso: str):
    url = f"https://api.esios.ree.es/indicators/{INDICATOR_ID}"
    params = {"start_date": start_dt_iso, "end_date": end_dt_iso}
    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    print("URL:", r.url, " ->", r.status_code)
    r.raise_for_status()
    values = r.json().get("indicator", {}).get("values", [])
    return pd.DataFrame(values)

if __name__ == "__main__":
    tz_madrid = tz.gettz("Europe/Madrid")
    hoy = datetime.now(tz_madrid).date()
    inicio = (hoy - timedelta(days=3)).isoformat()
    fin = hoy.isoformat()

    # Crear la tabla si no existe
    create_table()

    # Obtener datos de la API
    df = fetch_demanda(inicio + "T00:00:00+02:00", fin + "T00:00:00+02:00")
    print(df.head())

    # Guardar los datos en un archivo CSV
    df.to_csv(f"demanda_real_{inicio}_a_{fin}.csv", index=False)

    # Cargar los datos en la base de datos
    load_to_sql(df, 'dbo.demanda')  # Nombre correcto de la tabla (incluyendo esquema)
