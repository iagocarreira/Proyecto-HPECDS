import os, logging, pandas as pd, requests
from datetime import datetime, timedelta
from dateutil import tz
from dotenv import load_dotenv

from db_utils import (
    create_table, create_week_table, load_to_sql,
    cleanup_week_table, transfer_week_to_main, log_etl_execution
)
from transform_utils import transform

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

INDICATORS = {"real": 1293, "previsto": 544, "programado": 2053}


def fetch_indicator(indicator_id, col_prefix, start_dt_iso, end_dt_iso):
    url = f"https://api.esios.ree.es/indicators/{indicator_id}"
    params = {"start_date": start_dt_iso, "end_date": end_dt_iso}
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        r.raise_for_status()
        values = r.json().get("indicator", {}).get("values", [])
        df = pd.DataFrame(values)
        df = df.rename(columns={"value": f"valor_{col_prefix}"})
        logging.info(f"Extraídos {len(df)} registros para {col_prefix}")
        return df
    except Exception as e:
        logging.error(f"Error al extraer indicador {col_prefix}: {e}")
        return pd.DataFrame()


def extract(start, end):
    dfs = []
    for prefix, indicator_id in INDICATORS.items():
        df = fetch_indicator(indicator_id, prefix, start, end)
        if not df.empty:
            dfs.append(df[["datetime", "datetime_utc", f"valor_{prefix}"]])
    return dfs


def merge_and_transform(dfs):
    if not dfs:
        logging.warning("No se recibieron datos para transformar")
        return pd.DataFrame()

    merged = dfs[0]
    for d in dfs[1:]:
        merged = pd.merge(merged, d, on=["datetime", "datetime_utc"], how="outer")
    return transform(merged)


def run_etl():
    tz_madrid = tz.gettz("Europe/Madrid")
    start_time = datetime.now(tz_madrid)
    dag_name = os.getenv("DAG_NAME", "manual_run")
    log_status, log_msg = "INICIO", "Ejecución iniciada."

    try:
        hoy = start_time.date()
        inicio = (hoy - timedelta(days=3)).isoformat() + "T00:00:00+02:00"
        fin = hoy.isoformat() + "T00:00:00+02:00"

        logging.info(f"Ejecutando ETL ({dag_name}) desde {inicio} hasta {fin}")

        dfs = extract(inicio, fin)
        data = merge_and_transform(dfs)

        if data.empty:
            log_status = "SIN_DATOS"
            logging.warning("No hay datos para cargar.")
            return

        create_table()
        create_week_table()
        load_to_sql(data, "demanda_peninsula_semana")
        cleanup_week_table()

        if hoy.weekday() == 6:  # domingo
            transfer_week_to_main()

        log_status, log_msg = "OK", "ETL completada correctamente."

    except Exception as e:
        log_status, log_msg = "ERROR", str(e)
        logging.error(f"ETL fallida: {e}")

    finally:
        end_time = datetime.now(tz_madrid)
        log_etl_execution(start_time, end_time, log_status, log_msg, dag_name=dag_name)


if __name__ == "__main__":
    run_etl()


