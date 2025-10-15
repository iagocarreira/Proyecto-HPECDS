import os, logging, pandas as pd, requests
from datetime import datetime, timedelta
from dateutil import tz
from dotenv import load_dotenv
from db_utils import (
    create_main_and_week_tables, load_to_sql, transfer_oldest_from_week_to_main,
    ensure_week_table_has_7_days, log_etl_execution
)
from transform_utils import transform

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

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
        df = pd.DataFrame(r.json().get("indicator", {}).get("values", []))
        df = df.rename(columns={"value": f"valor_{col_prefix}"})
        return df
    except Exception as e:
        logging.error(f"Error extrayendo {col_prefix}: {e}")
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
        return pd.DataFrame()
    merged = dfs[0]
    for d in dfs[1:]:
        merged = pd.merge(merged, d, on=["datetime", "datetime_utc"], how="outer")
    return transform(merged)


def run_etl():
    tz_madrid = tz.gettz("Europe/Madrid")
    start_time = datetime.now(tz_madrid)
    dag_name = os.getenv("DAG_NAME", "manual_run")

    try:
        hoy = start_time.date()
        inicio = (hoy - timedelta(days=1)).isoformat() + "T00:00:00+02:00"
        fin = hoy.isoformat() + "T00:00:00+02:00"

        logging.info(f"Ejecutando ETL desde {inicio} hasta {fin}")
        dfs = extract(inicio, fin)
        data = merge_and_transform(dfs)

        if data.empty:
            logging.warning("No hay datos nuevos.")
            log_etl_execution(start_time, datetime.now(tz_madrid), "SIN_DATOS", "No se extrajeron datos.", dag_name)
            return

        create_main_and_week_tables()
        load_to_sql(data, "demanda_peninsula_semana")
        transfer_oldest_from_week_to_main()
        ensure_week_table_has_7_days()
        

        log_etl_execution(start_time, datetime.now(tz_madrid), "OK", "ETL completada correctamente.", dag_name)

    except Exception as e:
        logging.error(f"ETL fallida: {e}")
        log_etl_execution(start_time, datetime.now(tz_madrid), "ERROR", str(e), dag_name)


if __name__ == "__main__":
    run_etl()
