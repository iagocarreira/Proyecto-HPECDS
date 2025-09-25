# lgb_pipeline.py
import pandas as pd
import lightgbm as lgb
from sqlalchemy import create_engine, text
from sklearn.metrics import (
    mean_squared_error, mean_absolute_percentage_error,
    mean_absolute_error, r2_score
)
import matplotlib.pyplot as plt
import math  # añadido para calcular RMSE

# ===== 1) CONEXIÓN BD (ajusta si hiciera falta) =====
SERVER   = "udcserver2025.database.windows.net"
DATABASE = "grupo_1"
USER     = "ugrupo1"
PASSWORD = "HK9WXIJaBp2Q97haePdY"  # ojo: es 'l' minúscula tras HK9WX

ENGINE_URL = (
    f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER}:1433/{DATABASE}"
    f"?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no"
)
engine = create_engine(ENGINE_URL, pool_pre_ping=True)

# ===== 2) CARGA DATOS =====
# Si quieres limitar por rango temporal, añade WHERE fecha >= ...
query = text("SELECT fecha, valor FROM dbo.demanda ORDER BY fecha;")
df = pd.read_sql(query, engine, parse_dates=["fecha"]).dropna()
df = df.set_index("fecha").sort_index()

# (Opcional) asegurar frecuencia 5 min y rellenar pequeños huecos:
# df = df.asfreq("5T")
# df["valor"] = df["valor"].interpolate(limit_direction="both")

# ===== 3) FEATURES =====
def crear_features_avanzadas(s):
    df_ = s.copy()

    # Temporales
    df_["hora"]        = df_.index.hour
    df_["dia_semana"]  = df_.index.dayofweek
    df_["dia_mes"]     = df_.index.day
    df_["mes"]         = df_.index.month
    # (Opcional) fines de semana
    df_["es_fin_sem"]  = (df_["dia_semana"] >= 5).astype(int)

    # Lags (5 min, 1h, 24h)
    df_["lag_5min"]  = df_["valor"].shift(1)
    df_["lag_1h"]    = df_["valor"].shift(12)
    df_["lag_24h"]   = df_["valor"].shift(288)

    # Rolling windows solo con pasado (sin fuga) + min_periods
    df_["roll_mean_1h"]  = df_["valor"].rolling(12,  min_periods=12).mean().shift(1)
    df_["roll_std_1h"]   = df_["valor"].rolling(12,  min_periods=12).std().shift(1)
    df_["roll_mean_6h"]  = df_["valor"].rolling(72,  min_periods=72).mean().shift(1)
    df_["roll_mean_24h"] = df_["valor"].rolling(288, min_periods=288).mean().shift(1)

    # Quita filas con NaN generados por lags/rolling iniciales
    return df_.dropna()

df_feat = crear_features_avanzadas(df)

# ===== 4) SPLIT TEMPORAL (train/val/test) =====
# 80% trainval, 20% test; y dentro del trainval, 80/20 para validación
n = len(df_feat)
n_trainval = int(n * 0.8)
trainval = df_feat.iloc[:n_trainval].copy()
test     = df_feat.iloc[n_trainval:].copy()

n_val = int(len(trainval) * 0.2)
train = trainval.iloc[:-n_val].copy()
val   = trainval.iloc[-n_val:].copy()

X_train, y_train = train.drop(columns=["valor"]), train["valor"]
X_val,   y_val   = val.drop(columns=["valor"]),   val["valor"]
X_test,  y_test  = test.drop(columns=["valor"]),  test["valor"]

# ===== 5) ENTRENAMIENTO LIGHTGBM (early stopping en VALIDACIÓN) =====
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
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="mse",
    callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
)

# ===== 6) PREDICCIÓN Y MÉTRICAS =====
y_pred = lgb_reg.predict(X_test)

mse  = mean_squared_error(y_test, y_pred)
rmse = math.sqrt(mse)  # añadido: RMSE correcto
mape = mean_absolute_percentage_error(y_test, y_pred) * 100
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"MSE : {mse:.2f} MW^2")    # corregida la unidad
print(f"RMSE: {rmse:.2f} MW")     # añadido RMSE
print(f"MAPE: {mape:.2f}%")
print(f"MAE : {mae:.2f} MW")
print(f"R2  : {r2:.3f}")
print(f"Best iteration (early stopping): {lgb_reg.best_iteration_}")

# ===== 7) IMPORTANCIA DE VARIABLES =====
importancias = pd.Series(lgb_reg.feature_importances_, index=X_train.columns)\
                .sort_values(ascending=False)
print("\nTop-12 features:")
print(importancias.head(12))

# ===== 8) PLOTS =====
# Serie real vs predicción en test
res_test = pd.DataFrame({"real": y_test, "pred": y_pred}, index=y_test.index)

plt.figure(figsize=(14, 6))
plt.plot(res_test.index, res_test["real"], label="Real", linewidth=1.2)
plt.plot(res_test.index, res_test["pred"], label="Predicción LGBM", linewidth=1.2, alpha=0.9)
plt.title(f"Demanda real vs predicción | RMSE={rmse:.1f} MW | MAPE={mape:.2f}%")  # usa RMSE real
plt.xlabel("Tiempo"); plt.ylabel("MW"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.show()

# Importancias
plt.figure(figsize=(8, 6))
importancias.head(20).iloc[::-1].plot(kind="barh")
plt.title("Importancia de variables (Top 20)")
plt.tight_layout()
plt.show()
