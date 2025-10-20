import logging
import pandas as pd
from datetime import timedelta
import pytz

from utils.api_client import ESIOSClient
from transform.demand_transformer import DemandTransformer
from database.connection import SQLConnectionManager
from database.tables import TableManager
from database.loader import SQLLoader
from utils.time_utils import now_madrid
from config.settings import Settings


class ETLHistoric:
    """Pipeline ETL histórico para demanda eléctrica (por tramos)."""

    def __init__(self):
        self.client = ESIOSClient(Settings.ESIOS_TOKEN)
        self.transformer = DemandTransformer()
        self.conn_mgr = SQLConnectionManager()
        self.table_mgr = TableManager(self.conn_mgr)
        self.loader = SQLLoader(self.conn_mgr)
        self.indicators = {"real": 1293, "previsto": 544, "programado": 2053}
        self.dag_name = "etl_peninsula_historico"

    def run(self):
        """Ejecuta la ETL histórica descargando datos por tramos mensuales."""
        start_time = now_madrid()
        fin = start_time.date()
        inicio = fin - timedelta(days=365)
        logging.info(f"ETL Historico: {inicio.isoformat()}T00:00:00+02:00 → {fin.isoformat()}T00:00:00+02:00")

        madrid_tz = pytz.timezone("Europe/Madrid")
        dfs = []

        rango_actual = inicio
        while rango_actual < fin:
            rango_siguiente = min(rango_actual + timedelta(days=30), fin)
            start_iso = f"{rango_actual.isoformat()}T00:00:00+02:00"
            end_iso = f"{rango_siguiente.isoformat()}T00:00:00+02:00"
            logging.info(f"Descargando datos del {start_iso} al {end_iso}")

            tramo_dfs = []

            for prefix, indicator_id in self.indicators.items():
                try:
                    data = self.client.fetch(indicator_id, start_iso, end_iso, prefix)
                    if not data:
                        continue

                    df = pd.DataFrame(data)
                    if "value" in df.columns:
                        df = df.rename(columns={"value": f"valor_{prefix}"})

                    # Conversión de fechas robusta
                    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")

                    # Asegurar columnas sin tz en datetime local y UTC fijo
                    df["datetime_utc"] = df["datetime"]
                    df["datetime"] = df["datetime"].dt.tz_convert(madrid_tz).dt.tz_localize(None)
                    df["datetime_utc"] = df["datetime_utc"].dt.tz_localize(None)

                    df[f"valor_{prefix}"] = pd.to_numeric(df[f"valor_{prefix}"], errors="coerce")
                    df = df.dropna(subset=["datetime", "datetime_utc", f"valor_{prefix}"])

                    tramo_dfs.append(df[["datetime", "datetime_utc", f"valor_{prefix}"]])
                    logging.info(f"Extraídos {len(df)} registros para {prefix} ({start_iso} → {end_iso})")
                except Exception as e:
                    logging.error(f"Error extrayendo {prefix} ({start_iso} → {end_iso}): {e}")

            # Fusionar los tres indicadores del tramo
            if tramo_dfs:
                tramo_merged = tramo_dfs[0]
                for tdf in tramo_dfs[1:]:
                    tramo_merged = pd.merge(tramo_merged, tdf, on=["datetime", "datetime_utc"], how="outer")
                dfs.append(tramo_merged)

            rango_actual = rango_siguiente

        if not dfs:
            logging.warning("No se extrajeron datos del periodo histórico.")
            return

        # Concatenar todos los tramos (sin duplicar columnas)
        merged = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["datetime", "datetime_utc"])

        # Transformar y cargar
        transformed = self.transformer.transform(merged)
        self.table_mgr.ensure_tables_exist()
        

        try:
            self.loader.merge_insert(transformed, "demanda_peninsula")
            self.loader.log_etl_run(self.dag_name, start_time, "OK")
            logging.info(" ETL Histórica completada correctamente.")
        except Exception as e:
            logging.error(f" Error en la ETL histórica: {e}")
            self.loader.log_etl_run(self.dag_name, start_time, "KO")
