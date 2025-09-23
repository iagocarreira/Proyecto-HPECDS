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
            # Elimina el 'Z' (indicador de UTC) y convierte la cadena a un objeto datetime
            date_str = date_str.replace('Z', '')  
            return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%f')
        elif '+' in date_str:
            # Elimina la parte de la zona horaria, manteniendo la parte de la fecha
            date_str = date_str.split("+")[0]
            return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%f')
        else:
            # Si no tiene zona horaria, intentamos convertirla directamente
            return datetime.strptime(date_str, '%Y-%m-%dT%H:%M:%S.%f')
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
            demanda FLOAT NOT NULL
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

    for index, row in df.iterrows():
        # Asegúrate de convertir la fecha al formato adecuado antes de insertarla
        fecha = convert_to_datetime(row['datetime'])
        demanda = row['value']

        cursor.execute(f"""
            INSERT INTO {table_name} (fecha, demanda)
            VALUES (?, ?)
        """, fecha, demanda)  # Inserta la fecha y la demanda en la base de datos
    
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Datos cargados en la tabla {table_name}.")

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
