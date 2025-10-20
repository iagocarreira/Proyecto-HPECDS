import pandas as pd
import logging
from transform.base_transformer import BaseTransformer

class DemandTransformer(BaseTransformer):
    """Transformaciones específicas para demanda eléctrica."""
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._normalize(df)
        df = self._add_features(df)
        return df

    def _normalize(self, df):
        for c in ["valor_real", "valor_previsto", "valor_programado"]:
            if c in df:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["fecha"] = pd.to_datetime(df["datetime"], errors="coerce")
        return df

    def _add_features(self, df):
        if "fecha" not in df.columns:
            logging.warning("No existe columna 'fecha'.")
            return df
        df["hora"] = df["fecha"].dt.hour
        df["dia_semana"] = df["fecha"].dt.day_name()
        df["es_fin_semana"] = df["fecha"].dt.weekday >= 5
        if {"valor_real", "valor_previsto"}.issubset(df.columns):
            df["error_absoluto"] = (df["valor_real"] - df["valor_previsto"]).abs()
            df["error_relativo_pct"] = (
                (df["valor_real"] - df["valor_previsto"]) /
                df["valor_real"].replace(0, pd.NA)
            ) * 100
        return df
