#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# =========================
# 0) IMPORTS
# =========================
import os
import json
from datetime import datetime
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sqlalchemy import create_engine, text
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

import pyodbc  # asegúrate de tenerlo instalado

# =========================
# A) CONFIG REENTRENO (env)
# =========================
WINDOW_DAYS = int(os.getenv("WINDOW_DAYS", "180"))  # 90/180 recomendado
MODE = os.getenv("MODE", "warm").lower()            # "warm" semanal, "cold" mensual
PROMOTE_IF_BETTER = os.getenv("PROMOTE_IF_BETTER", "1") == "1"
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./artifacts")
RELOAD_URL = os.getenv("RELOAD_URL", "")            # p.ej. http://localhost:5000/admin/reload

# =========================
# 1) CONEXIÓN A LA BASE DE DATOS
# =========================
SERVER   = os.getenv("SQL_SERVER",   "udcserver2025.database.windows.net")
DATABASE = os.getenv("SQL_DB",       "grupo_1")
USER     = os.getenv("SQL_USER",     "ugrupo1")
PASSWORD = os.getenv("SQL_PASSWORD", "HK9WXIJaBp2Q97haePdY")

ENGINE_URL = (
    f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER}:1433/{DATABASE}"
    f"?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no"
)
engine = create_engine(ENGINE_URL, pool_pre_ping=True)

# =========================
# B) UTILIDADES ARTEFACTOS / PROMOCIÓN
# =========================
def _ensure_dirs():
    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)

def _current_paths():
    cur = Path(ARTIFACTS_DIR) / "current"
    return cur, cur / "modelo_lstm_multivariate.keras", cur / "scaler_lstm_multivariate.joblib"

def _maybe_promote(run_dir: Path, new_metrics: dict, baseline: dict | None):
    """
    Promociona si no hay baseline o si RMSE y MAPE mejoran (son menores).
    Crea symlink 'current' -> run_dir (o copia en Windows si symlink falla).
    Llama a RELOAD_URL si está definido.
    """
    improved = False if baseline else True
    if baseline:
        improved = (new_metrics["rmse"] < baseline.get("rmse", 1e9)) and \
                   (new_metrics["mape"] < baseline.get("mape", 1e9))

    if PROMOTE_IF_BETTER and improved:
        cur_link = Path(ARTIFACTS_DIR) / "current"
        try:
            if cur_link.exists() or cur_link.is_symlink():
                # si es carpeta real, borrar; si es symlink, unlink
                if cur_link.is_dir() and not cur_link.is_symlink():
                    import shutil
                    shutil.rmtree(cur_link, ignore_errors=True)
                else:
                    cur_link.unlink()
            # symlink relativo para portabilidad
            cur_link.symlink_to(run_dir.name)
            print(f"[INFO] Promocionado: {run_dir.name} -> {cur_link}")
        except Exception as e:
            # Fallback Windows: copiar si no se puede crear symlink
            print(f"[WARN] Symlink no disponible ({e}). Copiando artefactos a 'current'...")
            import shutil
            if cur_link.exists():
                shutil.rmtree(cur_link, ignore_errors=True)
            shutil.copytree(run_dir, cur_link)

        # Recarga API opcional
        if RELOAD_URL:
            try:
                import requests
                r = requests.post(RELOAD_URL, json={"reason":"auto-promotion","run_dir":str(run_dir)}, timeout=8)
                print(f"[INFO] Reload API status: {r.status_code}")
            except Exception as e:
                print(f"[WARN] No se pudo llamar a reload API: {e}")
    else:
        print("[INFO] Modelo NO promocionado (no mejora baseline o promoción desactivada).")

    return improved

# =========================
# 2) CARGAR Y PREPARAR DATOS
# =========================
sql = text("""
    SELECT
        fecha,
        valor_real,
        valor_previsto,
        valor_programado
    FROM dbo.demanda_peninsula
    ORDER BY fecha;
""")
df = pd.read_sql(sql, engine, parse_dates=["fecha"]).dropna()
df = df.set_index("fecha").sort_index()

# Renombrar a tus nombres usados en el modelo
df = df.rename(columns={
    "valor_real": "demanda_real",
    "valor_previsto": "demanda_prevista",
    "valor_programado": "demanda_programada"
})

# --- C) Ventana deslizante (rolling) ---
if WINDOW_DAYS > 0 and len(df) > 0:
    tmax = df.index.max()
    tmin = tmax - pd.Timedelta(days=WINDOW_DAYS)
    df = df.loc[(df.index >= tmin) & (df.index <= tmax)].copy()

# =========================
# 3) SPLIT + ESCALADO (SIN LEAKAGE)
# =========================
features_to_scale = ["demanda_real", "demanda_prevista", "demanda_programada"]

train_size = int(len(df) * 0.8)
df_train = df.iloc[:train_size]
df_test  = df.iloc[train_size:]

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df_train[features_to_scale])                 # <-- fit SOLO con train
train_data = scaler.transform(df_train[features_to_scale])
test_data  = scaler.transform(df_test[features_to_scale])

# =========================
# 4) INGENIERÍA DE SECUENCIAS
# =========================
LOOK_BACK = 24  # 24 pasos (ajusta a tu frecuencia; p.ej. 24*5min = 2h si es cada 5 min)

def create_sequences_multivariate(data, look_back):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :])   # ventana con 3 features
        Y.append(data[i + look_back, 0])       # objetivo: demanda_real siguiente (columna 0)
    return np.array(X), np.array(Y)

X_train, y_train = create_sequences_multivariate(train_data, LOOK_BACK)
X_test, y_test = create_sequences_multivariate(test_data, LOOK_BACK)

print(f"Forma de X_train (LSTM 3D): {X_train.shape}")
print(f"Datos de entrenamiento: {len(y_train)} | Datos de test: {len(y_test)}")

# =========================
# 5) MODELO LSTM
# =========================
print("\nDefiniendo y entrenando el modelo LSTM...")
num_features = X_train.shape[2]

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(LOOK_BACK, num_features)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# --- D) Warm-start antes de entrenar ---
_ensure_dirs()
cur_dir, cur_model_p, cur_scaler_p = _current_paths()
if MODE == "warm" and cur_model_p.exists():
    try:
        model.load_weights(cur_model_p.as_posix())
        print("[INFO] Warm-start: pesos previos cargados.")
    except Exception as e:
        print(f"[WARN] No se pudieron cargar pesos previos: {e}")

history = model.fit(
    X_train, y_train,
    epochs=100,                      # puedes reducir en warm si quieres; EarlyStopping corta solo
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
    shuffle=False,   # <-- importante en series temporales
    verbose=0
)

# =========================
# 6) PREDICCIÓN Y MÉTRICAS
# =========================
y_pred_escalada = model.predict(X_test, verbose=0).flatten()

# Invertir el escalado (colocando valores en la columna 0 del scaler)
dummy_array_test = np.zeros((len(y_test), len(features_to_scale)))
dummy_array_test[:, 0] = y_test
y_test_original = scaler.inverse_transform(dummy_array_test)[:, 0]

dummy_array_pred = np.zeros((len(y_pred_escalada), len(features_to_scale)))
dummy_array_pred[:, 0] = y_pred_escalada
y_pred_original = scaler.inverse_transform(dummy_array_pred)[:, 0]

# Métricas en MW
rmse = root_mean_squared_error(y_test_original, y_pred_original)
mape = mean_absolute_percentage_error(y_test_original, y_pred_original) * 100
mae  = mean_absolute_error(y_test_original, y_pred_original)

# Métricas normalizadas (en %)
nrmse_mean  = rmse / np.mean(y_test_original) * 100
nrmse_range = rmse / (np.max(y_test_original) - np.min(y_test_original)) * 100

print(f"\nError (RMSE) del modelo LSTM en test: {rmse:.2f} MW")
print(f"Error (MAE)  del modelo LSTM en test: {mae:.2f} MW")
print(f"Error (MAPE) del modelo LSTM en test: {mape:.2f}%")
print(f"nRMSE (sobre media): {nrmse_mean:.2f}%")
print(f"nRMSE (sobre rango): {nrmse_range:.2f}%")

# Diagnóstico de error máximo (valor y timestamp)
indices_test = df_test.index[LOOK_BACK:]  # coherente con el nuevo split
err = y_pred_original - y_test_original
imax = int(np.argmax(np.abs(err)))
if len(indices_test) > 0:
    print(f"Error máximo: {err[imax]:.2f} MW en {indices_test[imax]}")

# === Evaluación baseline (modelo 'current') para decidir promoción ===
baseline = None
if cur_model_p.exists():
    try:
        base_m = tf.keras.models.load_model(cur_model_p.as_posix())
        base_pred_scaled = base_m.predict(X_test, verbose=0).flatten()
        dummy_base = np.zeros((len(base_pred_scaled), len(features_to_scale)))
        dummy_base[:, 0] = base_pred_scaled
        base_pred_original = scaler.inverse_transform(dummy_base)[:, 0]
        base_rmse = root_mean_squared_error(y_test_original, base_pred_original)
        base_mape = mean_absolute_percentage_error(y_test_original, base_pred_original) * 100
        baseline = {"rmse": float(base_rmse), "mape": float(base_mape)}
        print(f"[BASELINE] RMSE={base_rmse:.2f} MW | MAPE={base_mape:.2f}%")
    except Exception as e:
        print(f"[WARN] No se pudo evaluar baseline: {e}")

# =========================
# 7) VISUALIZACIÓN PREDICCIÓN
# =========================
resultados_test = pd.DataFrame({
    'real': y_test_original,
    'prediccion': y_pred_original
}, index=indices_test)

plt.figure(figsize=(16, 8))
plt.style.use('seaborn-v0_8-whitegrid')
plt.plot(resultados_test.index, resultados_test['real'], label='Demanda Real (MW)', linewidth=1.5)
plt.plot(resultados_test.index, resultados_test['prediccion'], label='Predicción LSTM (MW)', linewidth=2.5, alpha=0.8)
plt.title(
    f'Predicción de Demanda con LSTM | RMSE: {rmse:.2f} MW | MAE: {mae:.2f} MW | '
    f'MAPE: {mape:.2f}% | nRMSE(media): {nrmse_mean:.2f}%',
    fontsize=18
)
plt.xlabel("Fecha y Hora", fontsize=14)
plt.ylabel("Potencia (MW)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# =========================
# 8) ANOMALÍAS (híbrido σ-rodante afinado)
# =========================
# Parámetros
K = 2.2                 # sensibilidad base (se recalibra si AUTO_CALIB=True)
PCT_THRESHOLD = 0.006   # 0.6% del valor real
FLOOR_MW = 80.0         # suelo absoluto
EWMA_ALPHA = 0.0        # sin suavizado para no aplastar picos
MIN_STREAK = 1          # nº mínimo de puntos consecutivos
WIN_MIN = 180           # ~3 h
MIN_FRAC = 0.5
AUTO_CALIB = True
TARGET_RATE = 0.015     # objetivo ~1.5%
CAP_SIGMA_RATIO = 1.3   # tope frente a σ inflada

# 8.1 Residuos
df_an = resultados_test.copy()
df_an["residuos"] = df_an["real"] - df_an["prediccion"]
df_an["res_smooth"] = df_an["residuos"] if EWMA_ALPHA == 0 else df_an["residuos"].ewm(alpha=EWMA_ALPHA, adjust=False).mean()

# util: ventana en puntos
def _infer_window_points(index, minutes=WIN_MIN, min_frac=MIN_FRAC):
    try:
        freq = pd.infer_freq(index)
    except Exception:
        freq = None
    if not freq:
        delta = (pd.Series(index).diff().median()
                 if len(index) >= 2 else pd.Timedelta(minutes=5))
        if pd.isna(delta): 
            delta = pd.Timedelta(minutes=5)
    else:
        delta = pd.to_timedelta(freq)
    step_min = max(1, int(delta.total_seconds() // 60))
    win = max(1, int(math.ceil(minutes / step_min)))
    min_periods = max(1, int(math.ceil(win * min_frac)))
    return win, min_periods

win, min_per = _infer_window_points(df_an.index)

# 8.2 σ rodante con warm-up suave + robusto MAD + CAP
res_s = df_an["res_smooth"].shift(1)
sigma_roll = res_s.rolling(win, min_periods=min_per).std()
sigma_exp  = res_s.expanding(min_periods=max(10, min_per//2)).std()  # warm-up
sigma_loc  = sigma_roll.fillna(sigma_exp)

mad_loc = (res_s.rolling(win, min_periods=min_per)
           .apply(lambda s: np.median(np.abs(s - np.median(s))), raw=False))
sigma_robusta = 1.4826 * mad_loc

sigma_eff = sigma_loc.copy()
sigma_eff = np.where(
    (pd.isna(sigma_eff)) | (sigma_eff > CAP_SIGMA_RATIO * sigma_robusta),
    sigma_robusta,
    sigma_eff
)
sigma_eff = pd.Series(sigma_eff, index=df_an.index).replace(0, np.nan)

# 8.3 Umbral híbrido y auto-calibración
umbral_sigma = K * sigma_eff
umbral_pct   = PCT_THRESHOLD * df_an["real"].abs()
umbral_abs   = pd.Series(FLOOR_MW, index=df_an.index)
df_an["umbral_MW"] = pd.concat([umbral_sigma, umbral_pct, umbral_abs], axis=1).max(axis=1)

if AUTO_CALIB:
    val = (df_an["res_smooth"].abs() / sigma_eff).replace([np.inf, -np.inf], np.nan).dropna()
    if not val.empty:
        K = float(np.nanpercentile(val, 100*(1 - TARGET_RATE)))
        umbral_sigma = K * sigma_eff
        df_an["umbral_MW"] = pd.concat([umbral_sigma, umbral_pct, umbral_abs], axis=1).max(axis=1)

# 8.4 Etiquetado (con streak opcional)
is_anom_point = (df_an["res_smooth"].abs() > df_an["umbral_MW"].fillna(np.inf))
if MIN_STREAK > 1:
    streak = (is_anom_point.groupby((~is_anom_point).cumsum())
              .cumcount() + 1) * is_anom_point.astype(int)
    is_anom = streak >= MIN_STREAK
else:
    is_anom = is_anom_point
df_an["anomalia"] = is_anom.astype(int)

# 8.5 Resumen
n = len(df_an); n_anom = int(df_an["anomalia"].sum())
print("\n=== Anomalías (híbrido σ-rodante afinado) ===")
print(f"Ventana≈{win} pts | K≈{K:.2f} | PCT={PCT_THRESHOLD*100:.2f}% | FLOOR={FLOOR_MW} MW")
print(f"Filas: {n} | Anómalos: {n_anom} ({100*n_anom/n:.2f}%)")
print("pct |res| > umbral:", float((df_an['residuos'].abs() > df_an['umbral_MW']).mean()*100), "%")

# 8.6 Visualización
plt.figure(figsize=(16,8))
plt.style.use('seaborn-v0_8-whitegrid')
plt.plot(df_an.index, df_an["real"], label='Demanda Real (MW)', linewidth=1.5)
plt.plot(df_an.index, df_an["prediccion"], label='Predicción LSTM (MW)', linewidth=2.2, alpha=0.9)
plt.fill_between(df_an.index,
                 df_an["prediccion"] - df_an["umbral_MW"],
                 df_an["prediccion"] + df_an["umbral_MW"],
                 alpha=0.18, label='Banda ± umbral híbrido')
if n_anom > 0:
    pts = df_an[df_an["anomalia"]==1]
    plt.scatter(pts.index, pts["real"], s=70, zorder=5, label="Anomalía")
plt.title(f"Detección de Anomalías con LSTM | híbrido σ-rodante (K≈{K:.2f}) | RMSE: {rmse:.2f} MW", fontsize=16)
plt.xlabel("Fecha y Hora"); plt.ylabel("Potencia (MW)")
plt.legend(fontsize=12); plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(); plt.show()

# =========================
# 9) GUARDAR ARTEFACTOS VERSIONADOS + PROMOCIÓN
# =========================
_ensure_dirs()
ts_run = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
run_dir = Path(ARTIFACTS_DIR) / f"run_{ts_run}"
run_dir.mkdir(parents=True, exist_ok=True)

# Modelo Keras con el nombre que espera la API
model.save((run_dir / "modelo_lstm_multivariate.keras").as_posix())

# Guardar el scaler
from joblib import dump
dump(scaler, run_dir / "scaler_lstm_multivariate.joblib")

# Guardar métricas y anomalías
metrics_out = {
    "rmse": float(rmse),
    "mae": float(mae),
    "mape": float(mape),
    "nrmse_mean": float(nrmse_mean),
    "nrmse_range": float(nrmse_range),
    "baseline_rmse": float(baseline["rmse"]) if baseline else None,
    "baseline_mape": float(baseline["mape"]) if baseline else None,
    "mode": MODE,
    "window_days": WINDOW_DAYS,
    "seq_len": LOOK_BACK,
    "epochs": int(len(history.history.get("loss", []))),  # aprox. efectivas
    "n_obs_train": int(len(df_train)),
    "n_obs_test": int(len(df_test)),
    "features": features_to_scale,
    "targets": ["demanda_real"],
    "run_dir": str(run_dir),
}
with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_out, f, ensure_ascii=False, indent=2)

df_an.to_csv(run_dir / "anomalies.csv", index=True)

# Imprime JSON final (útil para Airflow/XCom o inspección manual)
print(json.dumps(metrics_out, ensure_ascii=False))

# Promoción condicionada (+ recarga API opcional)
_promoted = _maybe_promote(run_dir, {"rmse": float(rmse), "mape": float(mape)}, baseline)
if _promoted:
    print(f"[INFO] Modelo PROMOCIONADO: {run_dir.name}")

# =========================
# 10) INFERENCIA "HOY" (one-step-ahead) + fallback opcional
# =========================
# Nota:
# - Entrenamiento genera (X[t-LOOK_BACK..t-1] -> y[t]).
# - Para predecir el siguiente instante (t_next), NO necesitas covariables de t_next.

def _infer_freq(index: pd.DatetimeIndex):
    """Intenta inferir la frecuencia para proponer el timestamp siguiente."""
    try:
        freq = pd.infer_freq(index)
        if freq:
            return pd.tseries.frequencies.to_offset(freq)
    except Exception:
        pass
    if len(index) >= 2:
        delta = pd.Series(index).diff().median()
        if pd.isna(delta):
            delta = pd.Timedelta(minutes=60)
        return pd.tseries.frequencies.to_offset(delta)
    return pd.tseries.frequencies.to_offset(pd.Timedelta(minutes=60))

def predict_next_step(model, scaler, df_full, look_back=LOOK_BACK):
    """
    Predice el siguiente punto y[t_next] usando las ÚLTIMAS 'look_back' filas (hasta t).
    No requiere 'valor_previsto'/'valor_programado' del futuro.
    """
    df_full = df_full.sort_index()
    if len(df_full) < look_back:
        raise ValueError("No hay suficientes filas para construir la ventana de inferencia.")
    feats = ["demanda_real","demanda_prevista","demanda_programada"]
    X_win = df_full[feats].values[-look_back:, :]
    X_scaled = scaler.transform(X_win)
    X_scaled = np.expand_dims(X_scaled, axis=0)  # (1, look_back, num_features)
    y_scaled = model.predict(X_scaled, verbose=0).flatten()[0]
    dummy = np.zeros((1, len(feats))); dummy[0, 0] = y_scaled  # desescalar como en test
    y_pred = scaler.inverse_transform(dummy)[0, 0]

    # timestamp objetivo = último índice + frecuencia inferida
    off = _infer_freq(df_full.index)
    t_next = df_full.index[-1] + off
    return t_next, float(y_pred)

# --- OPCIONAL: si quieres predecir t+1 usando covariables en t (con fallback) ---
def weekly_seasonal_fill(hist_series: pd.Series, t, k_weeks=4):
    """Media últimas k semanas para (dow, hora, minuto); plan B: media últimos 28 días por hora."""
    s = hist_series.dropna().copy()
    if s.empty:
        return float("nan")
    dfh = s.to_frame("v")
    dfh["dow"] = dfh.index.dayofweek
    dfh["hod"] = dfh.index.hour
    dfh["min"] = dfh.index.minute
    mask = (dfh.index < t) & (dfh.index >= t - pd.Timedelta(weeks=k_weeks)) \
           & (dfh["dow"] == t.dayofweek) & (dfh["hod"] == t.hour) & (dfh["min"] == t.minute)
    vals = dfh.loc[mask, "v"]
    if len(vals):
        return float(vals.mean())
    mask2 = (dfh.index < t) & (dfh["hod"] == t.hour) & (dfh["min"] == t.minute)
    vals2 = dfh.loc[mask2].tail(28)["v"]
    return float(vals2.mean()) if len(vals2) else float(dfh["v"].iloc[-1])

def predict_t_plus1_with_fallback(model, scaler, df_full, look_back=LOOK_BACK):
    """
    Predice y[t+1] usando la ventana que termina en t (requiere covariables en t).
    Si faltan 'demanda_prevista'/'demanda_programada' en t, las imputa con estacionalidad semanal.
    """
    df_full = df_full.sort_index()
    if len(df_full) < look_back + 1:
        raise ValueError("No hay suficientes filas para construir la ventana.")
    feats = ["demanda_real","demanda_prevista","demanda_programada"]

    # t = último índice disponible
    t = df_full.index[-1]
    row_t = df_full.iloc[-1].copy()
    if pd.isna(row_t["demanda_prevista"]):
        row_t["demanda_prevista"] = weekly_seasonal_fill(df_full["demanda_prevista"], t)
    if pd.isna(row_t["demanda_programada"]):
        row_t["demanda_programada"] = weekly_seasonal_fill(df_full["demanda_programada"], t)

    # ventana: (t-look_back+1 ... t)
    tail = df_full.iloc[-look_back-1:-1].copy()
    tail.loc[t] = row_t  # garantizamos covariables en t
    X_win = tail[feats].values[-look_back:, :]

    X_scaled = scaler.transform(X_win)
    X_scaled = np.expand_dims(X_scaled, axis=0)
    y_scaled = model.predict(X_scaled, verbose=0).flatten()[0]
    dummy = np.zeros((1, len(feats))); dummy[0, 0] = y_scaled
    y_pred = scaler.inverse_transform(dummy)[0, 0]

    # objetivo de esta ventana es t+1
    off = _infer_freq(df_full.index)
    t_next = t + off
    return t_next, float(y_pred)

# ------- Ejecución de ejemplo (elige el modo que prefieras) -------
try:
    # One-step-ahead (coherente con el entrenamiento): no necesita exógenas futuras
    t_pred, y_hat = predict_next_step(model, scaler, df, look_back=LOOK_BACK)
    print(f"[INFERENCIA] Predicción one-step-ahead para {t_pred}: {y_hat:.2f} MW")

    # Si quieres forzar usar covariables “de ahora” y prever el siguiente:
    # t_pred2, y_hat2 = predict_t_plus1_with_fallback(model, scaler, df, look_back=LOOK_BACK)
    # print(f"[INFERENCIA] Predicción (t+1 con fallback) para {t_pred2}: {y_hat2:.2f} MW")

except Exception as e:
    print(f"[WARN] No se pudo ejecutar la inferencia: {e}")
