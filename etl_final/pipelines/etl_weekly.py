import logging
import pandas as pd
from datetime import timedelta
from utils.api_client import ESIOSClient
from transform.demand_transformer import DemandTransformer
from database.connection import SQLConnectionManager
from database.tables import TableManager
from database.loader import SQLLoader
from utils.time_utils import now_madrid
from config.settings import Settings

class ETLWeekly:
    """Pipeline ETL semanal para demanda eléctrica."""

    def __init__(self):
        self.client = ESIOSClient(Settings.ESIOS_TOKEN)
        self.transformer = DemandTransformer()
        self.conn_mgr = SQLConnectionManager()
        self.table_mgr = TableManager(self.conn_mgr)
        self.loader = SQLLoader(self.conn_mgr)
        self.indicators = {"real": 1293, "previsto": 544, "programado": 2053}
        self.dag_name = "etl_peninsula_weekly"

    def run(self):
        start_time = now_madrid()
        inicio = (start_time.date() - timedelta(days=1)).isoformat() + "T00:00:00+02:00"
        fin = start_time.date().isoformat() + "T00:00:00+02:00"
        logging.info(f"ETL Semanal: {inicio} → {fin}")

        dfs = []

        try:
            for prefix, indicator_id in self.indicators.items():
                try:
                    data = self.client.fetch(indicator_id, inicio, fin, prefix)
                    if not data:
                        logging.warning(f"No se obtuvieron datos para {prefix}.")
                        continue

                    df = pd.DataFrame(data)

                    # Normalizar nombres y tipos
                    if "value" in df.columns:
                        df = df.rename(columns={"value": f"valor_{prefix}"})
                    df["datetime"] = pd.to_datetime(df.get("datetime"), errors="coerce")
                    df["datetime_utc"] = pd.to_datetime(df.get("datetime_utc"), errors="coerce")
                    if f"valor_{prefix}" in df.columns:
                        df[f"valor_{prefix}"] = pd.to_numeric(df[f"valor_{prefix}"], errors="coerce")
                    df = df.dropna(subset=["datetime", "datetime_utc", f"valor_{prefix}"])
                    dfs.append(df[["datetime", "datetime_utc", f"valor_{prefix}"]])
                except Exception as e:
                    logging.error(f"Error procesando indicador '{prefix}': {e}")

            if not dfs:
                logging.warning("No se extrajeron datos de ningún indicador.")
                self.loader.log_etl_run(self.dag_name, start_time, "KO")
                return

            merged = dfs[0]
            for d in dfs[1:]:
                merged = pd.merge(merged, d, on=["datetime", "datetime_utc"], how="outer")

            transformed = self.transformer.transform(merged)
            logging.info(f"Tipos de datos antes de insertar:\n{transformed.dtypes}")

            self.table_mgr.ensure_tables_exist()
            self.loader.merge_insert(transformed, "demanda_peninsula_semana")
            self.loader.roll_week_window(
                week_table="demanda_peninsula_semana",
                hist_table="demanda_peninsula",
                max_days=8
            )

            #  Registrar éxito
            self.loader.log_etl_run(self.dag_name, start_time, "OK")
            logging.info("ETL Semanal completada correctamente.")

        except Exception as e:
            logging.error(f"Error general en ETL semanal: {e}")
            self.loader.log_etl_run(self.dag_name, start_time, "KO")
