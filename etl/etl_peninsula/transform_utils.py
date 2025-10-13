import pandas as pd
import logging

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza tipos de datos básicos."""
    for col in ["valor_real", "valor_previsto", "valor_programado"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["fecha"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df

def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade columnas útiles para análisis y dashboards."""
    if "fecha" not in df.columns:
        logging.warning("No existe columna 'fecha' en el dataframe")
        return df

    df["hora"] = df["fecha"].dt.hour
    df["dia_semana"] = df["fecha"].dt.day_name()
    df["es_fin_semana"] = df["fecha"].dt.weekday >= 5

    if "valor_real" in df and "valor_previsto" in df:
        df["error_absoluto"] = (df["valor_real"] - df["valor_previsto"]).abs()
        df["error_relativo_pct"] = (
            (df["valor_real"] - df["valor_previsto"]) / df["valor_real"].replace(0, pd.NA)
        ) * 100

    return df

def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline de transformaciones para ETL."""
    df = normalize_columns(df)
    df = add_basic_features(df)
    return df
