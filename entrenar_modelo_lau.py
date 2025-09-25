import pandas as pd
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

# ===== 2) CARGA DATOS (UNA SOLA TABLA: dbo.demanda_total) =====
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

min_cols = ["fecha", "demanda_real", "demanda_prevista", "demanda_programada", "geo_name"]
missing = [c for c in min_cols if c not in df.columns]
if missing:
    raise SystemExit(f"Faltan columnas obligatorias tras la carga: {missing}")

df = df.dropna(subset=["fecha"]).set_index("fecha").sort_index()

# ===== 3) FEATURES =====
y = df["demanda_real"]
feature_cols = ["demanda_prevista", "demanda_programada"]

if df["geo_name"].nunique() > 1:
    df["geo_name"] = df["geo_name"].astype("category")
    feature_cols.append("geo_name")

X = df[feature_cols].dropna()
y = y.loc[X.index]

# ===== 4) SPLIT TEMPORAL =====
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

# ===== 5) ENTRENAMIENTO LIGHTGBM =====
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

# ===== 6) PREDICCIÓN Y MÉTRICAS =====
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
print("\nImportancia de variables:")
print(importancias)

# ===== 8) PLOTS PREDICCIÓN =====
res_test = pd.DataFrame({"real": y_test, "pred": y_pred}, index=y_test.index)

plt.figure(figsize=(14, 6))
plt.plot(res_test.index, res_test["real"], label="Real", linewidth=1.2)
plt.plot(res_test.index, res_test["pred"], label="Predicción LGBM", linewidth=1.2, alpha=0.9)
plt.title(f"Demanda real vs predicción | RMSE={rmse:.1f} MW | MAPE={mape:.2f}%")
plt.xlabel("Tiempo"); plt.ylabel("MW"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
importancias.iloc[::-1].plot(kind="barh")
plt.title("Importancia de variables")
plt.tight_layout()
plt.show()

# ===== 9) DETECCIÓN DE ANOMALÍAS (integrada) =====
# Parámetros del detector
ROLLING_MINUTES = 180   # 3 horas
PCT_THRESHOLD   = 0.03  # 3% respecto a la demanda real
ZSCORE_LIMIT    = 3.0   # 3 sigmas
MIN_PERIODS_STD = 36    # puntos mínimos para sigma (evita falsos positivos al inicio)

# Construye DF con real y pred sobre el conjunto de test
anom_df = res_test.copy()
anom_df["residual"]     = anom_df["real"] - anom_df["pred"]
anom_df["residual_pct"] = (anom_df["residual"] / anom_df["real"]).abs()

# Ventana para sigma según la frecuencia del índice
freq = pd.infer_freq(anom_df.index) or "5T"
step_minutes = pd.Timedelta(freq).seconds / 60
win = max(1, math.ceil(ROLLING_MINUTES / step_minutes))

# Sigma móvil SIN FUGA (shift antes del rolling)
anom_df["sigma"] = (
    anom_df["residual"]
    .shift(1)
    .rolling(window=win, min_periods=max(MIN_PERIODS_STD, win // 2))
    .std()
)

# Z-score y umbral mixto
anom_df["zscore"] = anom_df["residual"] / anom_df["sigma"]
umbral_sigma = (anom_df["sigma"] * ZSCORE_LIMIT).fillna(0.0)
umbral_pct   = (anom_df["real"] * PCT_THRESHOLD).fillna(0.0)
anom_df["umbral_abs"] = pd.concat([umbral_sigma, umbral_pct], axis=1).max(axis=1)

# Bandera de anomalía y motivo
anom_df["is_anom"] = anom_df["residual"].abs() > anom_df["umbral_abs"]

def _motivo(row):
    m = []
    if pd.notna(row["zscore"]) and abs(row["zscore"]) >= ZSCORE_LIMIT:
        m.append(f"|z|≥{ZSCORE_LIMIT}")
    if row["residual_pct"] >= PCT_THRESHOLD:
        m.append(f"abs(err)≥{int(PCT_THRESHOLD*100)}%")
    return " y ".join(m) or "umbral mixto"

anom_df["motivo"] = anom_df.apply(_motivo, axis=1)

# Resumen por consola
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

# (Opcional) Plot con anomalías marcadas
plt.figure(figsize=(14, 6))
plt.plot(anom_df.index, anom_df["real"], label="Real", linewidth=1.2)
plt.plot(anom_df.index, anom_df["pred"], label="Predicción", linewidth=1.2, alpha=0.9)
if n_anom > 0:
    puntos = anom_df[anom_df["is_anom"]]
    plt.scatter(puntos.index, puntos["real"], marker="o", s=40, label="Anomalía", zorder=3)
plt.title("Detección de anomalías sobre el conjunto de test")
plt.xlabel("Tiempo"); plt.ylabel("MW"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()
