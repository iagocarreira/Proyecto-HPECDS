import pandas as pd
import numpy as np
import lightgbm as lgb
from sqlalchemy import create_engine, text
from sklearn.metrics import (
    mean_squared_error, mean_absolute_percentage_error,
    mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import math

# ===== 1) CONEXIÓN BD =====
SERVER   = "udcserver2025.database.windows.net"
DATABASE = "grupo_1"
USER     = "ugrupo1"
PASSWORD = "HK9WXIJaBp2Q97haePdY"  # en prod usa variables de entorno

ENGINE_URL = (
    f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER}:1433/{DATABASE}"
    f"?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no"
)
engine = create_engine(ENGINE_URL, pool_pre_ping=True)

# ===== 1.1) FUNCIONES DE LIMPIEZA =====
def limpiar_para_modelo(df, col="demanda_real", freq="5T", max_gap_minutes=60):
    """ Limpieza moderada: pensada para entrenar """
    df_ = df.copy().sort_index()
    df_ = df_.asfreq(freq)
    # Interpolar huecos pequeños
    step = pd.Timedelta(freq).seconds // 60
    max_steps = max(1, max_gap_minutes // step)
    mask_nan = df_[col].isna()
    grp = (~mask_nan).cumsum()
    run_len = mask_nan.groupby(grp).transform("sum")
    small = mask_nan & (run_len <= max_steps)
    serie = df_[col].copy()
    serie[small] = serie.interpolate(limit_direction="both")[small]
    df_[col] = serie
    # Quitar negativos imposibles
    df_[col] = df_[col].clip(lower=0)
    return df_

def limpiar_para_anomalias(df, col="demanda_real", freq="5T"):
    """ Limpieza mínima: solo lo necesario para detectar anomalías """
    df_ = df.copy().sort_index()
    df_ = df_.asfreq(freq)
    df_[col] = df_[col].clip(lower=0)
    return df_

# ===== 2) CARGA DATOS =====
sql = text("""
SELECT
    fecha,
    fecha_utc,
    tz_time,
    valor_real,
    valor_previsto,
    valor_programado,
    geo_name_real,
    geo_name_previsto,
    geo_name_programado
FROM dbo.demanda_total
ORDER BY fecha;
""")

df = pd.read_sql(sql, engine, parse_dates=["fecha", "fecha_utc", "tz_time"])

# ===== 2.1) NORMALIZACIÓN DE NOMBRES =====
df = df.rename(columns={
    "valor_real":       "demanda_real",
    "valor_previsto":   "demanda_prevista",
    "valor_programado": "demanda_programada",
})

geo_cols = [c for c in ["geo_name_real", "geo_name_previsto", "geo_name_programado"] if c in df.columns]
if geo_cols:
    df["geo_name"] = df[geo_cols].bfill(axis=1).iloc[:, 0]
else:
    df["geo_name"] = "Península"

df = df.dropna(subset=["fecha"]).set_index("fecha").sort_index()

# ===== 2.2) APLICAR LIMPIEZAS =====
df_model = limpiar_para_modelo(df, col="demanda_real", freq="5T")
df_anom  = limpiar_para_anomalias(df, col="demanda_real", freq="5T")

# ===== 3) FEATURES AVANZADAS (sobre df_model) =====
def crear_features_avanzadas(df_in, target_col="demanda_real"):
    df_ = df_in.copy()
    df_["hora"]        = df_.index.hour
    df_["dia_semana"]  = df_.index.dayofweek
    df_["dia_mes"]     = df_.index.day
    df_["mes"]         = df_.index.month
    df_["es_fin_sem"]  = (df_["dia_semana"] >= 5).astype(int)
    df_[f"lag_5min"]  = df_[target_col].shift(1)
    df_[f"lag_1h"]    = df_[target_col].shift(12)
    df_[f"lag_24h"]   = df_[target_col].shift(288)
    df_[f"roll_mean_1h"]  = df_[target_col].rolling(12,  min_periods=12).mean().shift(1)
    df_[f"roll_std_1h"]   = df_[target_col].rolling(12,  min_periods=12).std().shift(1)
    df_[f"roll_mean_6h"]  = df_[target_col].rolling(72,  min_periods=72).mean().shift(1)
    df_[f"roll_mean_24h"] = df_[target_col].rolling(288, min_periods=288).mean().shift(1)
    df_ = df_.dropna(subset=[
        target_col,
        f"lag_5min", f"lag_1h", f"lag_24h",
        f"roll_mean_1h", f"roll_std_1h", f"roll_mean_6h", f"roll_mean_24h"
    ])
    return df_

df_feat = crear_features_avanzadas(df_model, target_col="demanda_real")

# ===== 3.1) MATRIZ DE FEATURES =====
feature_cols = [
    "demanda_prevista",
    "demanda_programada",
    "hora", "dia_semana", "dia_mes", "mes", "es_fin_sem",
    "lag_5min", "lag_1h", "lag_24h",
    "roll_mean_1h", "roll_std_1h", "roll_mean_6h", "roll_mean_24h"
]
y = df_feat["demanda_real"]

if df_feat["geo_name"].nunique() > 1:
    df_feat["geo_name"] = df_feat["geo_name"].astype("category")
    feature_cols.append("geo_name")

X = df_feat[feature_cols].dropna()
y = y.loc[X.index]

# ===== 4) SPLIT =====
n = len(X)
if n < 100:
    raise SystemExit("Muy pocos datos para entrenar.")
n_trainval = int(n * 0.8)
trainval_idx = X.index[:n_trainval]
test_idx     = X.index[n_trainval:]

trainval = X.loc[trainval_idx].copy()
test     = X.loc[test_idx].copy()
y_trainval = y.loc[trainval_idx]
y_test     = y.loc[test_idx]

n_val = int(len(trainval) * 0.2)
train = trainval.iloc[:-n_val].copy()
val   = trainval.iloc[-n_val:].copy()
y_train = y_trainval.iloc[:-n_val].copy()
y_val   = y_trainval.iloc[-n_val:].copy()

# ===== 5) ENTRENAMIENTO =====
lgb_reg = lgb.LGBMRegressor(
    objective="regression",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

lgb_reg.fit(
    train, y_train,
    eval_set=[(val, y_val)],
    eval_metric="mse",
    callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
)

# ===== 6) PREDICCIÓN =====
y_pred = lgb_reg.predict(test)

mse  = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"MSE : {mse:.2f} MW^2")
print(f"RMSE: {rmse:.2f} MW")
print(f"MAPE: {mape:.2f}%")
print(f"MAE : {mae:.2f} MW")
print(f"R2  : {r2:.3f}")
print(f"Best iteration (early stopping): {lgb_reg.best_iteration_}")

# ===== 7) IMPORTANCIA DE VARIABLES =====
importancias = pd.Series(lgb_reg.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nImportancia de variables (Top 20):")
print(importancias.head(20))

# ===== 8) PLOTS PREDICCIÓN =====
# Para anomalías, usar los valores casi crudos
real_test = df_anom.loc[y_test.index, "demanda_real"]
res_test = pd.DataFrame({"real": real_test, "pred": y_pred}, index=y_test.index)

plt.figure(figsize=(14, 6))
plt.plot(res_test.index, res_test["real"], label="Real", linewidth=1.2)
plt.plot(res_test.index, res_test["pred"], label="Predicción LGBM", linewidth=1.2, alpha=0.9)
plt.title(f"Demanda real vs predicción | RMSE={rmse:.1f} MW | MAPE={mape:.2f}%")
plt.xlabel("Tiempo"); plt.ylabel("MW"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
importancias.head(20).iloc[::-1].plot(kind="barh")
plt.title("Importancia de variables (Top 20)")
plt.tight_layout()
plt.show()

# ===== 9) DETECCIÓN DE ANOMALÍAS =====
ROLLING_MINUTES = 180
PCT_THRESHOLD   = 0.02   # más sensible
ZSCORE_LIMIT    = 2.5
MIN_PERIODS_STD = 36

anom_df = res_test.copy()
anom_df["residual"]     = anom_df["real"] - anom_df["pred"]
anom_df["residual_pct"] = (anom_df["residual"] / anom_df["real"]).abs()

freq = pd.infer_freq(anom_df.index) or "5T"
step_minutes = pd.Timedelta(freq).seconds / 60
win = max(1, math.ceil(ROLLING_MINUTES / step_minutes))

anom_df["sigma"] = (
    anom_df["residual"]
    .shift(1)
    .rolling(window=win, min_periods=max(MIN_PERIODS_STD, win // 2))
    .std()
)

anom_df["zscore"] = anom_df["residual"] / anom_df["sigma"]
umbral_sigma = (anom_df["sigma"] * ZSCORE_LIMIT).fillna(0.0)
umbral_pct   = (anom_df["real"] * PCT_THRESHOLD).fillna(0.0)
# cambiar a min para disparar si supera cualquiera de los criterios
anom_df["umbral_abs"] = pd.concat([umbral_sigma, umbral_pct], axis=1).min(axis=1)
anom_df["is_anom"] = anom_df["residual"].abs() > anom_df["umbral_abs"]

def _motivo(row):
    m = []
    if pd.notna(row["zscore"]) and abs(row["zscore"]) >= ZSCORE_LIMIT:
        m.append(f"|z|≥{ZSCORE_LIMIT}")
    if row["residual_pct"] >= PCT_THRESHOLD:
        m.append(f"abs(err)≥{int(PCT_THRESHOLD*100)}%")
    return " y ".join(m) or "umbral mixto"

anom_df["motivo"] = anom_df.apply(_motivo, axis=1)

total = len(anom_df)
n_anom = int(anom_df["is_anom"].sum())
print("\n=== Detección de anomalías (sobre TEST) ===")
print(f"Filas analizadas: {total}")
print(f"Anomalías detectadas: {n_anom} ({(n_anom/total*100):.2f}%)")

if n_anom > 0:
    top = (
        anom_df[anom_df["is_anom"]]
        .assign(abs_z=anom_df["zscore"].abs())
        .sort_values("abs_z", ascending=False)
        .drop(columns=["abs_z"])
        .head(10)[["real","pred","residual","residual_pct","sigma","zscore","umbral_abs","is_anom","motivo"]]
    )
    print("\nTop 10 anomalías por |zscore|:")
    print(top)

plt.figure(figsize=(14, 6))
plt.plot(anom_df.index, anom_df["real"], label="Real", linewidth=1.2)
plt.plot(anom_df.index, anom_df["pred"], label="Predicción", linewidth=1.2, alpha=0.9)
if n_anom > 0:
    puntos = anom_df[anom_df["is_anom"]]
    plt.scatter(puntos.index, puntos["real"], marker="o", s=40, label="Anomalía", zorder=3)
plt.title("Detección de anomalías sobre el conjunto de test")
plt.xlabel("Tiempo"); plt.ylabel("MW"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()
