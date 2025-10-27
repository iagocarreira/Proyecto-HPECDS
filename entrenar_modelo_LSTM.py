#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script híbrido LSTM para demanda eléctrica
- Une lo mejor de tus dos scripts:
  * Mejor esquema de escalado: X con MinMaxScaler y y con y_scaler (salida lineal → sin saturación)
  * Pronóstico multi-step (closed-loop) + one-step-ahead funcional
  * Carga cálida (warm start) compatible con artefactos antiguos y nuevos
  * Promoción automática si mejora baseline y/o modelo "current"
  * Anomalías basadas en el modelo (LSTM) con submuestreo de chequeo
  * Ventana por SQL opcional, artefactos versionados y figuras guardadas

Variables de entorno más relevantes (con defaults razonables):
  WINDOW_DAYS=180 | MODE=warm | LOOK_BACK=24 | HORIZON_MIN=1440 | VAL_FRAC=0.15
  LR=0.001 | EPOCHS=80 | BATCH=128 | DROPOUT=0.2 | UNITS1=64 | UNITS2=32 | PATIENCE=10
  SQL_WINDOW_IN_QUERY=0 | ARTIFACTS_DIR=./artifacts | PROMOTE_IF_BETTER=1
  ENABLE_MULTISTEP=1 | FALLBACK=persist | VERBOSE=1
  ANOM_CHECK_EVERY_MIN=15 | ANOM_EWMA_ALPHA=0.1 | ANOM_WIN_MIN=1440 | ANOM_TARGET=0.002
"""

import os
import json
from pathlib import Path
from datetime import datetime
import math
import random
import numpy as np
import pandas as pd
import time

# Matplotlib no interactivo (para servidores)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Opcional: .env si existe
try:
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv(usecwd=True), override=False)
except Exception:
    pass

from sqlalchemy import create_engine, text
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    root_mean_squared_error,
    mean_absolute_percentage_error,
    mean_absolute_error,
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

from joblib import dump, load
import pyodbc  # asegurar instalado si se usa ODBC

# =========================
# Config (env)
# =========================
WINDOW_DAYS = int(os.getenv("WINDOW_DAYS", "180"))
SQL_WINDOW_IN_QUERY = os.getenv("SQL_WINDOW_IN_QUERY", "0") == "1"
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./artifacts")
PROMOTE_IF_BETTER = os.getenv("PROMOTE_IF_BETTER", "1") == "1"
MODE = os.getenv("MODE", "warm").lower()  # 'cold' o 'warm'
LOOK_BACK = int(os.getenv("LOOK_BACK", "24"))
HORIZON_MIN = int(os.getenv("HORIZON_MIN", "1440"))
VAL_FRAC = float(os.getenv("VAL_FRAC", "0.15"))
LR = float(os.getenv("LR", "0.001"))
EPOCHS = int(os.getenv("EPOCHS", "80"))
BATCH = int(os.getenv("BATCH", os.getenv("BATCH_SIZE", "128")))
DROPOUT = float(os.getenv("DROPOUT", "0.2"))
UNITS1 = int(os.getenv("UNITS1", "64"))
UNITS2 = int(os.getenv("UNITS2", "32"))
PATIENCE = int(os.getenv("PATIENCE", "10"))
FALLBACK = os.getenv("FALLBACK", "persist").lower()  # 'persist' o 'zero'
ENABLE_MULTISTEP = os.getenv("ENABLE_MULTISTEP", "1") == "1"
VERBOSE = int(os.getenv("VERBOSE", "1"))
SEED = int(os.getenv("SEED", "42"))
RELOAD_URL = os.getenv("RELOAD_URL", "")

# Reproducibilidad
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

# =========================
# SQL
# =========================
SERVER   = os.getenv("SQL_SERVER",   "udcserver2025.database.windows.net")
DATABASE = os.getenv("SQL_DB",       "grupo_1")
USER     = os.getenv("SQL_USER",     "ugrupo1")
PASSWORD = os.getenv("SQL_PASSWORD", "HK9WXIJaBp2Q97haePdY")
SQL_TABLE = os.getenv("SQL_TABLE", "dbo.demanda_peninsula_semana")

ENGINE_URL = (
    f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER}:1433/{DATABASE}"
    f"?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no"
)
engine = create_engine(ENGINE_URL, pool_pre_ping=True)

# =========================
# Utilidades de artefactos
# =========================

def _ensure_dirs():
    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    cur = Path(ARTIFACTS_DIR) / "current"
    if not cur.exists():
        cur.mkdir(parents=True, exist_ok=True)

def _current_paths():
    cur = Path(ARTIFACTS_DIR) / "current"
    # Compatibilidad nombres antiguos y nuevos
    keras_new = cur / "model.keras"
    keras_old = cur / "modelo_lstm_multivariate.keras"
    xsc_p = cur / "x_scaler.joblib"
    ysc_p = cur / "y_scaler.joblib"
    mono_scaler_old = cur / "scaler_lstm_multivariate.joblib"  # compat
    return cur, keras_new, keras_old, xsc_p, ysc_p, mono_scaler_old


def _maybe_promote(run_dir: Path, new_metrics: dict, baseline: dict | None):
    improved = False if baseline else True
    if baseline:
        improved = (new_metrics["rmse"] < baseline.get("rmse", 1e9)) and \
                   (new_metrics["mape"] < baseline.get("mape", 1e9))
    if PROMOTE_IF_BETTER and improved:
        cur_link = Path(ARTIFACTS_DIR) / "current"
        try:
            if cur_link.exists() or cur_link.is_symlink():
                if cur_link.is_dir() and not cur_link.is_symlink():
                    import shutil; shutil.rmtree(cur_link, ignore_errors=True)
                else:
                    cur_link.unlink()
        except Exception:
            pass
        try:
            cur_link.symlink_to(run_dir.resolve(), target_is_directory=True)
        except Exception:
            import shutil; shutil.copytree(run_dir, cur_link, dirs_exist_ok=True)
        if RELOAD_URL:
            try:
                import requests
                _ = requests.post(RELOAD_URL, json={"reason":"auto-promotion","run_dir":str(run_dir)}, timeout=8)
            except Exception:
                pass
        print(f"[INFO] Promocionado a current: {run_dir.name}")
    else:
        print("[INFO] Modelo NO promocionado (no mejora baseline o promoción desactivada).")
    return improved

# =========================
# Carga de datos
# =========================
if SQL_WINDOW_IN_QUERY and WINDOW_DAYS > 0:
    sql = text(f"""
        WITH mx AS (SELECT MAX(fecha) AS fmax FROM {SQL_TABLE})
        SELECT fecha, valor_real, valor_previsto, valor_programado
        FROM {SQL_TABLE}, mx
        WHERE fecha >= DATEADD(day, -{WINDOW_DAYS}, mx.fmax)
        ORDER BY fecha;
    """)
else:
    sql = text(f"""
        SELECT fecha, valor_real, valor_previsto, valor_programado
        FROM {SQL_TABLE}
        ORDER BY fecha;
    """)

df = pd.read_sql(sql, engine, parse_dates=["fecha"]).dropna()

df = (
    df.set_index("fecha").sort_index()
      .rename(columns={
          "valor_real": "demanda_real",
          "valor_previsto": "demanda_prevista",
          "valor_programado": "demanda_programada",
      })
)

if not SQL_WINDOW_IN_QUERY and WINDOW_DAYS > 0 and len(df) > 0:
    tmax = df.index.max(); tmin = tmax - pd.Timedelta(days=WINDOW_DAYS)
    df = df.loc[(df.index >= tmin) & (df.index <= tmax)].copy()

features = ["demanda_real", "demanda_prevista", "demanda_programada"]

n = len(df)
if n < 100:
    raise ValueError("Muy pocos datos para un split 70/15/15.")

i_train = int(n * 0.70)
rem = n - i_train
i_val = i_train + int(rem * VAL_FRAC)

df_train = df.iloc[:i_train].copy()
df_val   = df.iloc[i_train:i_val].copy()
df_test  = df.iloc[i_val:].copy()

# Escalado: X con x_scaler; y con y_scaler independiente
x_scaler = MinMaxScaler().fit(df_train[features])  # solo train
y_scaler = MinMaxScaler().fit(df_train[["demanda_real"]])


def _build_sequences(X, y, look_back):
    Xs, ys = [], []
    for i in range(len(X) - look_back):
        Xs.append(X[i:i+look_back, :])
        ys.append(y[i+look_back])
    return np.array(Xs), np.array(ys)

X_train = x_scaler.transform(df_train[features])
X_val   = x_scaler.transform(df_val[features])
X_test  = x_scaler.transform(df_test[features])

y_train_all = y_scaler.transform(df_train[["demanda_real"]])[:, 0]
y_val_all   = y_scaler.transform(df_val[["demanda_real"]])[:, 0]
y_test_all  = y_scaler.transform(df_test[["demanda_real"]])[:, 0]

X_train, y_train = _build_sequences(X_train, y_train_all, LOOK_BACK)
X_val,   y_val   = _build_sequences(X_val,   y_val_all,   LOOK_BACK)
X_test,  y_test  = _build_sequences(X_test,  y_test_all,  LOOK_BACK)

print(f"Formas — X_train:{X_train.shape} | X_val:{X_val.shape} | X_test:{X_test.shape}")

# =========================
# Modelo
# =========================

def make_model():
    model = Sequential([
        LSTM(UNITS1, return_sequences=True, input_shape=(LOOK_BACK, len(features))),
        Dropout(DROPOUT),
        LSTM(UNITS2),
        Dropout(DROPOUT),
        Dense(1, activation='sigmoid')  # salida en [0,1] para y escalada con MinMaxScaler
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss="mse",
        metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")]
    )
    return model

model = make_model()

# Warm start compatible con artefactos anteriores
_ensure_dirs()
cur_dir, keras_new, keras_old, xsc_p_cur, ysc_p_cur, mono_scaler_old = _current_paths()

if MODE == "warm":
    loaded = False
    for path in [keras_new, keras_old]:
        if path.exists():
            try:
                print(f"[WARM] Cargando pesos de: {path}")
                # Cargamos weights (no arquitectura) para evitar incompatibilidades menores
                prev = tf.keras.models.load_model(path.as_posix())
                model.set_weights(prev.get_weights())
                loaded = True
                break
            except Exception as e:
                print(f"[WARM] No se pudo cargar: {e}")
    if not loaded:
        print("[WARM] Sin pesos previos compatibles, entrenamiento desde cero.")

cb = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
print("\nDefiniendo y entrenando el modelo LSTM.")
t0 = time.time()
hist = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH,
    callbacks=[cb],
    verbose=VERBOSE
)
train_time_s = time.time() - t0
print(f"[INFO] Tiempo de entrenamiento: {train_time_s:.2f} s")
# =========================
# Evaluación (TEST) + métricas
# =========================

y_pred_scaled = model.predict(X_test, verbose=0).flatten()
# Forzamos al rango [0,1] para evitar extrapolación al desescalar
y_pred_scaled = np.clip(y_pred_scaled, 0.0, 1.0)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()

y_test_orig = y_scaler.inverse_transform(y_test.reshape(-1,1)).ravel()

mape = mean_absolute_percentage_error(y_test_orig, y_pred) * 100
rmse = root_mean_squared_error(y_test_orig, y_pred)
mae  = mean_absolute_error(y_test_orig, y_pred)
rng = float(np.max(y_test_orig) - np.min(y_test_orig))
nrmse_mean  = rmse / float(np.mean(y_test_orig)) * 100
nrmse_range = (rmse / rng * 100) if rng > 0 else float("nan")

print(f"\n[LSTM TEST] RMSE={rmse:.2f} MW | MAE={mae:.2f} MW | MAPE={mape:.2f}% | nRMSE(mean)={nrmse_mean:.2f}%")

indices_test = df_test.index[LOOK_BACK:]

# Baselines

y_real = y_test_orig
previsto    = df_test['demanda_prevista'].iloc[LOOK_BACK:].to_numpy()
programado  = df_test['demanda_programada'].iloc[LOOK_BACK:].to_numpy()
persistence = df_test['demanda_real'].shift(1).iloc[LOOK_BACK:].to_numpy()

def _metrics(name, yhat):
    _rmse = root_mean_squared_error(y_real, yhat)
    _mae  = mean_absolute_error(y_real, yhat)
    _mape = mean_absolute_percentage_error(y_real, yhat) * 100
    print(f"[{name}] RMSE={_rmse:.2f} | MAE={_mae:.2f} | MAPE={_mape:.2f}%")
    return {"rmse": float(_rmse), "mae": float(_mae), "mape": float(_mape)}

base_prev = _metrics("Previsto", previsto)
base_prog = _metrics("Programado", programado)
base_pers = _metrics("Persistencia t-1", persistence)

# =========================
# Comparativa con 'current' (si existe) usando SUS scalers
# =========================

baseline_model_metrics = None
try:
    if (xsc_p_cur.exists() and ysc_p_cur.exists() and (keras_new.exists() or keras_old.exists())):
        base_m = tf.keras.models.load_model((keras_new if keras_new.exists() else keras_old).as_posix())
        xsc_cur = load(xsc_p_cur.as_posix()); ysc_cur = load(ysc_p_cur.as_posix())
        X_test_cur = xsc_cur.transform(df_test[features])
        X_test_cur, _ = _build_sequences(X_test_cur, y_test_all, LOOK_BACK)
        base_pred_scaled = base_m.predict(X_test_cur, verbose=0).flatten()
        base_pred = ysc_cur.inverse_transform(base_pred_scaled.reshape(-1,1)).ravel()
    elif mono_scaler_old.exists() and (keras_old.exists() or keras_new.exists()):
        base_m = tf.keras.models.load_model((keras_new if keras_new.exists() else keras_old).as_posix())
        mono_sc = load(mono_scaler_old.as_posix())
        test_data_mono = mono_sc.transform(df_test[features])
        def _seq_mono(data, lb):
            Xs = []
            for i in range(len(data) - lb):
                Xs.append(data[i:i+lb, :])
            return np.array(Xs)
        X_test_mono = _seq_mono(test_data_mono, LOOK_BACK)
        base_pred_scaled = base_m.predict(X_test_mono, verbose=0).flatten()
        dummy = np.zeros((len(base_pred_scaled), len(features)))
        dummy[:,0] = base_pred_scaled
        base_pred = load(mono_scaler_old.as_posix()).inverse_transform(dummy)[:,0]
    else:
        base_pred = None

    if base_pred is not None:
        base_rmse = root_mean_squared_error(y_real, base_pred)
        base_mape = mean_absolute_percentage_error(y_real, base_pred) * 100
        baseline_model_metrics = {"rmse": float(base_rmse), "mape": float(base_mape)}
        print(f"[BASELINE MODEL current] RMSE={base_rmse:.2f} MW | MAPE={base_mape:.2f}%")
except Exception as e:
    print(f"[WARN] Comparación con current fallida: {e}")

# =========================
# Overlay de test
# =========================

resultados_test = pd.DataFrame({
    "real": y_real,
    "lstm": y_pred,
    "previsto": previsto,
    "programado": programado,
    "persistencia": persistence,
}, index=indices_test)

plt.figure(figsize=(16, 8))
plt.style.use('seaborn-v0_8-whitegrid')
plt.plot(resultados_test.index, resultados_test['real'], label='Demanda Real (MW)', linewidth=1.8)
plt.plot(resultados_test.index, resultados_test['lstm'], label='Predicción LSTM (MW)', linewidth=2.0, alpha=0.95)
plt.plot(resultados_test.index, resultados_test['previsto'], label='Previsto', linestyle='--', alpha=0.8)
plt.plot(resultados_test.index, resultados_test['persistencia'], label='Persistencia t-1', linestyle=':')
plt.title(
    f'Predicción de Demanda | LSTM TEST RMSE: {rmse:.2f} MW | MAE: {mae:.2f} MW | '
    f'MAPE: {mape:.2f}% | nRMSE(media): {nrmse_mean:.2f}%', fontsize=16
)
plt.xlabel("Fecha y Hora"); plt.ylabel("Potencia (MW)")
plt.legend(); plt.tight_layout()
_ensure_dirs()
plt.savefig(Path(ARTIFACTS_DIR) / "lstm_test_overlay.png", dpi=150, bbox_inches='tight')
plt.close()

# =========================
# Anomalías (modelo LSTM como referencia) con submuestreo
# =========================

# Referencia fijada al LSTM (requisito del PDF)
ref_name = "lstm"
ref_series = resultados_test["lstm"]
print(f"[INFO] Referencia de anomalías fijada en: {ref_name}")

df_an = pd.DataFrame({
    "real": resultados_test["real"],
    "referencia": ref_series
}, index=resultados_test.index)
df_an["residuo"] = df_an["real"] - df_an["referencia"]

# Hiper-parámetros (vía entorno)
EWMA_ALPHA = float(os.getenv("ANOM_EWMA_ALPHA", "0.10"))   # suaviza residuo
WIN_MIN    = int(os.getenv("ANOM_WIN_MIN", "1440"))        # ~24h
MIN_FRAC   = float(os.getenv("ANOM_MIN_FRAC", "0.8"))
K_INIT     = float(os.getenv("ANOM_K_INIT", "3.0"))
PCT_THRESHOLD = float(os.getenv("ANOM_PCT", "0.015"))
FLOOR_MW      = float(os.getenv("ANOM_FLOOR", "300.0"))
TARGET_RATE   = float(os.getenv("ANOM_TARGET", "0.002"))   # 0.2%
MAX_ITERS     = int(os.getenv("ANOM_MAX_ITERS", "8"))
ANOM_CHECK_EVERY_MIN = int(os.getenv("ANOM_CHECK_EVERY_MIN", "15"))  # 15 min por defecto

# Suavizado del residuo (para umbral robusto)
res_s = df_an["residuo"].ewm(alpha=EWMA_ALPHA, adjust=False).mean().shift(1)

def _infer_window_points(index: pd.DatetimeIndex, minutes=WIN_MIN, min_frac=MIN_FRAC):
    if len(index) < 3:
        return 12, 6
    try:
        step = pd.Series(index).diff().median()
    except Exception:
        step = pd.Timedelta(minutes=5)
    if pd.isna(step):
        step = pd.Timedelta(minutes=5)
    step_min = max(1, int(step.total_seconds() // 60))
    win = max(12, int(math.ceil(minutes / step_min)))
    min_periods = max(6, int(math.ceil(win * min_frac)))
    return win, min_periods

win, min_per = _infer_window_points(df_an.index)
mad = res_s.rolling(win, min_periods=min_per).apply(
    lambda s: np.median(np.abs(s - np.median(s))), raw=False
)
sigma_rob = 1.4826 * mad.replace(0, np.nan)

K = K_INIT

def _calc_umbral(K):
    u_sigma = K * sigma_rob
    u_pct   = PCT_THRESHOLD * df_an["real"].abs()
    u_abs   = pd.Series(FLOOR_MW, index=df_an.index)
    return pd.concat([u_sigma, u_pct, u_abs], axis=1).max(axis=1)

umbral = _calc_umbral(K)

# --- FRECUENCIA DE CHEQUEO (submuestreo) ---
# tamaño del paso real (en min)
try:
    _step = pd.Series(df_an.index).diff().median()
except Exception:
    _step = pd.Timedelta(minutes=5)
if pd.isna(_step):
    _step = pd.Timedelta(minutes=5)
step_min = max(1, int(_step.total_seconds() // 60))

# cada cuántos puntos chequear
stride = max(1, ANOM_CHECK_EVERY_MIN // step_min) if ANOM_CHECK_EVERY_MIN else 1
check_idx = df_an.index[::stride]

# --- Calibración de K usando SOLO los puntos muestreados ---
for _ in range(MAX_ITERS):
    flags_sub = (df_an.loc[check_idx, "residuo"].abs() > umbral.loc[check_idx]).astype(int)
    rate = float(flags_sub.mean())
    if not np.isfinite(rate):
        break
    if abs(rate - TARGET_RATE) < 0.0005:
        break
    factor = np.clip(TARGET_RATE / max(rate, 1e-9), 0.85, 1.18)
    K *= float(factor)
    umbral = _calc_umbral(K)

# Guardar umbral final y marcar anomalías SOLO en los timestamps muestreados
df_an["umbral_MW"] = umbral
df_an["anomalia"] = 0
df_an.loc[check_idx, "anomalia"] = (
    df_an.loc[check_idx, "residuo"].abs() > df_an.loc[check_idx, "umbral_MW"]
).astype(int)

# --- Plot ---
plt.figure(figsize=(16, 7))
plt.style.use('seaborn-v0_8-whitegrid')
plt.plot(df_an.index, df_an["real"], label="Real", linewidth=1.7)
plt.plot(df_an.index, df_an["referencia"], label=f"Referencia ({ref_name})", alpha=0.95)
plt.fill_between(
    df_an.index,
    (df_an["referencia"] - df_an["umbral_MW"]).values,
    (df_an["referencia"] + df_an["umbral_MW"]).values,
    alpha=0.15, label='Banda umbral híbrida'
)
if df_an["anomalia"].sum() > 0:
    pts = df_an[df_an["anomalia"]==1]
    plt.scatter(pts.index, pts["real"], s=50, zorder=5, label="Anomalía")
plt.title(
    f"Detección de Anomalías | ref: {ref_name} | check cada {ANOM_CHECK_EVERY_MIN} min | RMSE test LSTM: {rmse:.2f} MW",
    fontsize=14
)
plt.xlabel("Fecha y Hora"); plt.ylabel("Potencia (MW)")
plt.legend(); plt.tight_layout()
plt.savefig(Path(ARTIFACTS_DIR) / "anomalies_overlay.png", dpi=150, bbox_inches='tight')
plt.close()

# =========================
# Guardado artefactos + promoción
# =========================
_ensure_dirs()
ts_run = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
run_dir = Path(ARTIFACTS_DIR) / f"run_{ts_run}"
run_dir.mkdir(parents=True, exist_ok=True)

(model.save(run_dir / "model.keras"))
dump(x_scaler, run_dir / "x_scaler.joblib")
dump(y_scaler, run_dir / "y_scaler.joblib")

metrics_out = {
    "rmse": float(rmse),
    "mae": float(mae),
    "mape": float(mape),
    "nrmse_mean": float(nrmse_mean),
    "nrmse_range": float(nrmse_range),
    "baseline_previsto": base_prev,
    "baseline_programado": base_prog,
    "baseline_persistencia": base_pers,
    "mode": MODE,
    "window_days": WINDOW_DAYS,
    "seq_len": LOOK_BACK,
    "epochs_effective": int(len(hist.history.get("loss", []))),
    "n_obs_train": int(len(df_train)),
    "n_obs_val": int(len(df_val)),
    "n_obs_test": int(len(df_test)),
    "features": features,
    "targets": ["demanda_real"],
    "run_dir": str(run_dir),
    "seed": SEED,
}
with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_out, f, ensure_ascii=False, indent=2)

df_an.to_csv(run_dir / "anomalies.csv", index=True)

print(json.dumps(metrics_out, ensure_ascii=False))

best_base_rmse = min(base_prev["rmse"], base_prog["rmse"], base_pers["rmse"])
should_promote = (
    (rmse < best_base_rmse) and
    (baseline_model_metrics is None or (rmse < baseline_model_metrics.get("rmse", 1e9) and mape < baseline_model_metrics.get("mape", 1e9)))
)
if should_promote:
    _ = _maybe_promote(run_dir, {"rmse": float(rmse), "mape": float(mape)}, baseline_model_metrics)
else:
    print("[INFO] Modelo NO promocionado: no mejora baselines y/o 'current'.")

# =========================
# Inferencia one-step-ahead y multi-step
# =========================

def _infer_freq(index: pd.DatetimeIndex):
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

def _step_minutes(index: pd.DatetimeIndex) -> int:
    try:
        freq = pd.infer_freq(index)
    except Exception:
        freq = None
    delta = pd.to_timedelta(freq) if freq else pd.Series(index).diff().median()
    if pd.isna(delta):
        delta = pd.Timedelta(minutes=5)
    return max(1, int(delta.total_seconds() // 60))

def predict_next_step(model, x_scaler, y_scaler, df_full, look_back=LOOK_BACK):
    df_full = df_full.sort_index()
    if len(df_full) < look_back:
        raise ValueError("No hay suficientes filas para inferencia.")
    feats = ["demanda_real","demanda_prevista","demanda_programada"]
    X_win_df = df_full[feats].iloc[-look_back:]
    X_scaled = x_scaler.transform(X_win_df)
    X_scaled = np.expand_dims(X_scaled, axis=0)

    y_scaled = float(model.predict(X_scaled, verbose=0).flatten()[0])
    y_scaled = float(np.clip(y_scaled, 0.0, 1.0))
    y_pred = float(y_scaler.inverse_transform([[y_scaled]])[0,0])

    off = _infer_freq(df_full.index)
    t_next = df_full.index[-1] + off
    return t_next, y_pred

def forecast_multi_step(model, x_scaler, y_scaler, df_hist, look_back, horizon_steps,
                        future_cov: pd.DataFrame | None = None,
                        fallback: str = FALLBACK,
                        feats=("demanda_real","demanda_prevista","demanda_programada")):
    hist = df_hist.copy(); cols = list(feats)
    preds = []
    cur_hist = hist.copy()
    step = _step_minutes(hist.index)

    for _ in range(horizon_steps):
        win = cur_hist.iloc[-look_back:].copy()
        X = x_scaler.transform(win.loc[:, cols])
        X = np.expand_dims(X, axis=0)

        y_scaled = float(model.predict(X, verbose=0).flatten()[0])
        y_scaled = float(np.clip(y_scaled, 0.0, 1.0))
        y_next = float(y_scaler.inverse_transform([[y_scaled]])[0,0])

        t_next = cur_hist.index[-1] + pd.Timedelta(minutes=step)
        preds.append((t_next, y_next))

        row = win.iloc[-1].copy()
        row["demanda_real"] = y_next
        if (future_cov is not None) and (t_next in future_cov.index):
            _cov = future_cov.loc[t_next]
            for c in ("demanda_prevista", "demanda_programada"):
                if c in _cov.index and not pd.isna(_cov[c]):
                    row[c] = _cov[c]
        else:
            if fallback == "persist":
                # mantener últimas covariables
                pass
            elif fallback == "zero":
                row["demanda_prevista"] = 0.0
                row["demanda_programada"] = 0.0
        cur_hist.loc[t_next] = row

    return pd.Series([v for _, v in preds], index=[t for t, _ in preds], name="forecast")

# One-step (seguro)
try:
    t_pred, y_hat = predict_next_step(model, x_scaler, y_scaler, df, look_back=LOOK_BACK)
    print(f"[INFERENCIA] Siguiente punto {t_pred}: {y_hat:.2f} MW")
except Exception as e:
    print(f"[WARN] One-step falló: {e}")

# Multi-step (opcional)
print(f"[CFG] Multi-step: HORIZON_MIN={HORIZON_MIN} | fallback={FALLBACK} | ENABLE_MULTISTEP={ENABLE_MULTISTEP}")
if ENABLE_MULTISTEP:
    try:
        steps = max(1, HORIZON_MIN // _step_minutes(df.index))
        _freq = _infer_freq(df.index)
        future_index = pd.date_range(df.index[-1] + _freq, periods=steps, freq=_freq)
        # Por defecto, mantener última covariable
        future_cov = pd.DataFrame(index=future_index)
        future_cov["demanda_prevista"] = df["demanda_prevista"].iloc[-1]
        future_cov["demanda_programada"] = df["demanda_programada"].iloc[-1]

        yhat_ms = forecast_multi_step(
            model, x_scaler, y_scaler, df, LOOK_BACK, steps,
            future_cov=future_cov, fallback=FALLBACK
        )
        out_path = run_dir / f"forecast_{HORIZON_MIN}min.csv"
        yhat_ms.to_csv(out_path, header=True)
        print(f"[FORECAST] Guardado multi-step ({HORIZON_MIN} min, {steps} pasos): {out_path}")
        print(yhat_ms.head(3).to_string()); print(yhat_ms.tail(3).to_string())
    except Exception as e:
        print(f"[ERROR] Multi-step: {e}")
else:
    print("[FORECAST] Desactivado por ENABLE_MULTISTEP=0")

# =========================
# Guardar métricas en BD (solo columnas principales)
# =========================
def save_metrics_db(engine, df_an, rmse, mae, mape, nrmse_mean, train_time_s):
    n_anomalias = int(df_an["anomalia"].sum())
    pct_anomalias = float(n_anomalias / len(df_an) * 100) if len(df_an) > 0 else 0

    query = text("""
        INSERT INTO metricas_lstm (
            fecha_ejecucion, rmse, mae, mape, nrmse_mean,
            n_anomalias, pct_anomalias, tiempo_entrenamiento_s
        )
        VALUES (
            GETDATE(), :rmse, :mae, :mape, :nrmse_mean,
            :n_anomalias, :pct_anomalias, :tiempo_entrenamiento_s
        )
    """)

    with engine.begin() as conn:
        conn.execute(query, {
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "nrmse_mean": float(nrmse_mean),
            "n_anomalias": n_anomalias,
            "pct_anomalias": pct_anomalias,
            "tiempo_entrenamiento_s": float(train_time_s)
        })

    print("[SQL] Métricas guardadas correctamente en metricas_lstm.")

# Guardar métricas en la BD
save_metrics_db(
    engine,
    df_an,
    rmse, mae, mape, nrmse_mean,
    train_time_s
)

