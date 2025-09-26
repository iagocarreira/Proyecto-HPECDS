import os
from typing import Optional, List
from datetime import datetime
import math
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sklearn.ensemble import IsolationForest

# ===== Configuración =====
SERVER   = os.getenv("SQL_SERVER",   "udcserver2025.database.windows.net")
DATABASE = os.getenv("SQL_DATABASE", "grupo_1")
USER     = os.getenv("SQL_USER",     "ugrupo1")
PASSWORD = os.getenv("SQL_PASSWORD", "HK9WXIJaBp2Q97haePdY")  # en prod: variables de entorno
ENGINE_URL = (
    f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER}:1433/{DATABASE}"
    f"?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no"
)
TABLE = "dbo.demanda_total"
FREQ = "5T"  # frecuencia objetivo

MODEL_PATH = os.getenv("MODEL_PATH", "models/model_lgbm.pkl")

# ===== Conexión BD =====
engine = create_engine(ENGINE_URL, pool_pre_ping=True)

# ===== Limpieza y features (idénticas a las usadas al entrenar) =====
def limpiar_para_modelo(df, col="demanda_real", freq=FREQ, max_gap_minutes=60):
    df_ = df.copy().sort_index().asfreq(freq)
    step = pd.Timedelta(freq).seconds // 60
    max_steps = max(1, max_gap_minutes // step)
    mask_nan = df_[col].isna()
    grp = (~mask_nan).cumsum()
    run_len = mask_nan.groupby(grp).transform("sum")
    small = mask_nan & (run_len <= max_steps)
    serie = df_[col].copy()
    serie[small] = serie.interpolate(limit_direction="both")[small]
    df_[col] = serie.clip(lower=0)
    return df_

def limpiar_para_anomalias(df, col="demanda_real", freq=FREQ):
    df_ = df.copy().sort_index().asfreq(freq)
    df_[col] = df_[col].clip(lower=0)
    return df_

def crear_features_avanzadas(df_in, target_col="demanda_real"):
    df_ = df_in.copy()
    idx = df_.index
    df_["hora"]        = idx.hour
    df_["dia_semana"]  = idx.dayofweek
    df_["dia_mes"]     = idx.day
    df_["mes"]         = idx.month
    df_["es_fin_sem"]  = (df_["dia_semana"] >= 5).astype(int)
    df_[f"lag_5min"]   = df_[target_col].shift(1)
    df_[f"lag_1h"]     = df_[target_col].shift(12)
    df_[f"lag_24h"]    = df_[target_col].shift(288)
    df_[f"roll_mean_1h"]  = df_[target_col].rolling(12,  min_periods=12).mean().shift(1)
    df_[f"roll_std_1h"]   = df_[target_col].rolling(12,  min_periods=12).std().shift(1)
    df_[f"roll_mean_6h"]  = df_[target_col].rolling(72,  min_periods=72).mean().shift(1)
    df_[f"roll_mean_24h"] = df_[target_col].rolling(288, min_periods=288).mean().shift(1)
    req = [
        target_col, "lag_5min", "lag_1h", "lag_24h",
        "roll_mean_1h", "roll_std_1h", "roll_mean_6h", "roll_mean_24h"
    ]
    return df_.dropna(subset=req)

FEATURE_COLS = [
    "demanda_prevista","demanda_programada",
    "hora","dia_semana","dia_mes","mes","es_fin_sem",
    "lag_5min","lag_1h","lag_24h",
    "roll_mean_1h","roll_std_1h","roll_mean_6h","roll_mean_24h"
]

# ===== Carga de datos desde SQL =====
def cargar_df_base(desde: Optional[str]=None, hasta: Optional[str]=None) -> pd.DataFrame:
    where = []
    params = {}
    if desde:
        where.append("fecha >= :desde"); params["desde"] = desde
    if hasta:
        where.append("fecha < :hasta");  params["hasta"]  = hasta
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    sql = text(f"""
        SELECT fecha, valor_real, valor_previsto, valor_programado,
               geo_name_real, geo_name_previsto, geo_name_programado
        FROM {TABLE}
        {where_sql}
        ORDER BY fecha;
    """)
    df = pd.read_sql(sql, engine, parse_dates=["fecha"])
    if df.empty:
        return df
    df = df.rename(columns={
        "valor_real":"demanda_real",
        "valor_previsto":"demanda_prevista",
        "valor_programado":"demanda_programada"
    })
    geo_cols = [c for c in ["geo_name_real","geo_name_previsto","geo_name_programado"] if c in df.columns]
    df["geo_name"] = df[geo_cols].bfill(axis=1).iloc[:,0] if geo_cols else "Península"
    return df.dropna(subset=["fecha"]).set_index("fecha").sort_index()

# ===== Model serving: cargar modelo entrenado =====
def cargar_modelo(path: str):
    if not os.path.exists(path):
        raise RuntimeError(
            f"No se encontró el modelo en '{path}'. "
            f"Entrena offline y guarda el artefacto (joblib.dump) antes de arrancar la API."
        )
    model = joblib.load(path)
    return model

MODEL = cargar_modelo(MODEL_PATH)

# ===== Esquemas de respuesta =====
class PuntoDemanda(BaseModel):
    fecha: datetime
    demanda_real: Optional[float] = None
    demanda_prevista: Optional[float] = None
    demanda_programada: Optional[float] = None
    geo_name: Optional[str] = None

class PuntoPred(BaseModel):
    fecha: datetime
    real: Optional[float]
    pred: float

class PuntoAnomalia(BaseModel):
    fecha: datetime
    real: float
    pred: float
    residual: float
    residual_pct: float
    score: float  # IsolationForest decision_function (negativo = más anómalo)

# ===== FastAPI =====
app = FastAPI(title="HPECDS – API de Datos, Predicciones y Anomalías", version="1.0.0")

@app.get("/health")
def health():
    best_iter = getattr(MODEL, "best_iteration_", None)
    return {"status": "ok", "model_loaded": True, "best_iteration": best_iter}

@app.get("/api/v1/demanda_total", response_model=List[PuntoDemanda])
def get_demanda_total(
    limit: int = Query(500, gt=1, le=5000),
    desde: Optional[str] = Query(None, description="ISO-8601, ej. 2025-09-24T00:00:00"),
    hasta: Optional[str] = Query(None, description="ISO-8601"),
):
    df = cargar_df_base(desde, hasta)
    if df.empty:
        return []
    df = df.tail(limit)
    out = df[["demanda_real","demanda_prevista","demanda_programada","geo_name"]].copy()
    out["fecha"] = out.index
    return out.reset_index(drop=True).to_dict(orient="records")

@app.get("/api/v1/predicciones", response_model=List[PuntoPred])
def get_predicciones(
    desde: Optional[str] = Query(None),
    hasta: Optional[str] = Query(None),
    limit: int = Query(1000, gt=1, le=20000)
):
    df_base = cargar_df_base(desde, hasta)
    if df_base.empty:
        return []
    # pipeline igual al de entrenamiento
    df_model = limpiar_para_modelo(df_base)
    df_feat  = crear_features_avanzadas(df_model, "demanda_real")
    if df_feat.empty:
        return []
    X = df_feat[FEATURE_COLS].copy()
    if "geo_name" in df_feat.columns and df_feat["geo_name"].nunique() > 1:
        X["geo_name"] = df_feat["geo_name"].astype("category")
    y_real = df_feat["demanda_real"]
    try:
        y_pred = MODEL.predict(X)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")
    res = pd.DataFrame({"real": y_real, "pred": y_pred}, index=y_real.index).tail(limit)
    res["fecha"] = res.index
    return res.reset_index(drop=True).to_dict(orient="records")

@app.get("/api/v1/anomalias", response_model=List[PuntoAnomalia])
def get_anomalias(
    desde: Optional[str] = Query(None),
    hasta: Optional[str] = Query(None),
    contamination: float = Query(0.02, ge=0.001, le=0.2),
    limit: int = Query(1000, gt=1, le=20000)
):
    # base y ramas
    df_base = cargar_df_base(desde, hasta)
    if df_base.empty:
        return []
    real_crudo = limpiar_para_anomalias(df_base)["demanda_real"]

    df_model = limpiar_para_modelo(df_base)
    df_feat  = crear_features_avanzadas(df_model, "demanda_real")
    if df_feat.empty:
        return []

    X = df_feat[FEATURE_COLS].copy()
    if "geo_name" in df_feat.columns and df_feat["geo_name"].nunique() > 1:
        X["geo_name"] = df_feat["geo_name"].astype("category")
    pred = pd.Series(MODEL.predict(X), index=X.index, name="pred")

    # alineación
    df_join = pd.DataFrame({"real": real_crudo}).join(pred, how="inner").dropna()
    if df_join.empty:
        return []

    # features para IF
    df_join["residual"] = df_join["real"] - df_join["pred"]
    df_join["residual_pct"] = (df_join["residual"] / df_join["real"]).abs()
    X_if = df_join[["residual","residual_pct"]].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    iso = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    labels = iso.fit_predict(X_if)
    scores = iso.decision_function(X_if)

    df_join["score"] = scores
    df_join["is_anom"] = (labels == -1)

    anoms = df_join[df_join["is_anom"]].copy()
    if anoms.empty:
        return []
    anoms = anoms.nsmallest(limit, "score")  # más anómalos primero
    anoms["fecha"] = anoms.index
    cols = ["fecha","real","pred","residual","residual_pct","score"]
    return anoms.reset_index(drop=True)[cols].to_dict(orient="records")
