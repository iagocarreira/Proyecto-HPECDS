import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from dateutil import tz
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from db_utils import create_table, load_to_sql
from transform_utils import transform  # importamos transformaciones

# ---------------- CONFIGURACIÓN ----------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

ESIOS_TOKEN = os.getenv("ESIOS_TOKEN")
HEADERS = {
    "Accept": "application/json; application/vnd.esios-api-v1+json",
    "Content-Type": "application/json",
    "x-api-key": ESIOS_TOKEN
}

INDICATORS = {
    "real": 1293,
    "previsto": 544,
    "programado": 2053
}

# Configurar sesión con reintentos automáticos
session = requests.Session()
retries = Retry(
    total=3,               # número total de reintentos
    backoff_factor=2,      # tiempo de espera incremental entre intentos
    status_forcelist=[429, 500, 502, 503, 504],  # errores que reintenta
    allowed_methods=["GET"]
)
session.mount("https://", HTTPAdapter(max_retries=retries))


# ---------------- EXTRACT ----------------
def fetch_indicator(indicator_id, col_prefix, start_dt_iso, end_dt_iso):
    """Descarga un indicador desde la API de ESIOS y lo devuelve como DataFrame."""
    url = f"https://api.esios.ree.es/indicators/{indicator_id}"
    params = {"start_date": start_dt_iso, "end_date": end_dt_iso}

    try:
        r = session.get(url, headers=HEADERS, params=params, timeout=90)
        r.raise_for_status()
        values = r.json().get("indicator", {}).get("values", [])
        df = pd.DataFrame(values)
        if df.empty:
            logging.warning(f"Sin registros para {col_prefix} en el rango {start_dt_iso} - {end_dt_iso}")
            return pd.DataFrame()
        df = df.rename(columns={"value": f"valor_{col_prefix}"})
        logging.info(f"Extraídos {len(df)} registros para {col_prefix} ({start_dt_iso} - {end_dt_iso})")
        return df[["datetime", "datetime_utc", f"valor_{col_prefix}"]]
    except Exception as e:
        logging.error(f"Error al extraer indicador {col_prefix} ({start_dt_iso} - {end_dt_iso}): {e}")
        return pd.DataFrame()


def extract_range(start, end):
    """Descarga todos los indicadores definidos en INDICATORS para un rango concreto."""
    dfs = []
    for prefix, indicator_id in INDICATORS.items():
        df = fetch_indicator(indicator_id, prefix, start, end)
        if not df.empty:
            dfs.append(df)
    return dfs


def extract_periodically(start_date, end_date, step_days=30):
    """Descarga los datos por intervalos mensuales y los combina por indicador."""
    all_data = {prefix: [] for prefix in INDICATORS.keys()}
    current = start_date

    while current < end_date:
        next_date = min(current + timedelta(days=step_days), end_date)
        start_iso = current.isoformat() + "T00:00:00+02:00"
        end_iso = next_date.isoformat() + "T00:00:00+02:00"
        logging.info(f"Descargando datos del {start_iso} al {end_iso}")

        for prefix, indicator_id in INDICATORS.items():
            df = fetch_indicator(indicator_id, prefix, start_iso, end_iso)
            if not df.empty:
                all_data[prefix].append(df)

        current = next_date

    # Concatenar todos los fragmentos de cada indicador
    dfs_final = []
    for prefix in INDICATORS.keys():
        if all_data[prefix]:
            concatenated = pd.concat(all_data[prefix], ignore_index=True)
            concatenated = concatenated.drop_duplicates(subset=["datetime_utc"])
            dfs_final.append(concatenated)

    return dfs_final


# ---------------- TRANSFORM ----------------
def merge_and_transform(dfs):
    """Une los DataFrames y aplica transformaciones básicas antes de cargar en SQL."""
    if not dfs:
        logging.warning("No se recibieron datos para transformar")
        return pd.DataFrame()

    merged = dfs[0]
    for d in dfs[1:]:
        merged = pd.merge(merged, d, on=["datetime", "datetime_utc"], how="outer")

    merged = transform(merged)
    return merged


# ---------------- PIPELINE ----------------
def run_etl():
    tz_madrid = tz.gettz(os.getenv("TIMEZONE", "Europe/Madrid"))
    hoy = datetime.now(tz_madrid).date()
    inicio = hoy - timedelta(days=365)
    fin = hoy

    logging.info(f"Ejecutando ETL desde {inicio} hasta {fin}")

    dfs = extract_periodically(inicio, fin, step_days=30)
    data = merge_and_transform(dfs)

    if data.empty:
        logging.warning("No hay datos para cargar en la base de datos")
        return

    create_table()
    load_to_sql(data, "demanda_peninsula")
    logging.info("ETL finalizada correctamente ")


if __name__ == "__main__":
    run_etl()
