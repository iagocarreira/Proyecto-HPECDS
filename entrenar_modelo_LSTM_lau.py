#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Entrenamiento LSTM (corregido):
- Split cronológico 70/15/15 → train/val/test (val para EarlyStopping; test ciego)
- Escalado fit SOLO con train (sin leakage)
- Baselines: Previsto, Programado y Persistencia (t-1)
- Opción de empujar la ventana temporal a SQL (SQL_WINDOW_IN_QUERY=1)
- Mantiene artefactos y promoción como en el script original
- Warm start desde modelo .keras (set_weights) y comparación justa con scaler del 'current'
"""

# =========================
# 0) IMPORTS
# =========================
import os
import json
from datetime import datetime
from pathlib import Path
import math
import random  # reproducibilidad
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

from joblib import dump, load  # dump y load de scaler
import pyodbc  # asegúrate de tenerlo instalado

# =========================
# A) CONFIG REENTRENO (env)
# =========================
WINDOW_DAYS = int(os.getenv("WINDOW_DAYS", "180"))  # 90/180 recomendado
MODE = os.getenv("MODE", "warm").lower()            # "warm" semanal, "cold" mensual
PROMOTE_IF_BETTER = os.getenv("PROMOTE_IF_BETTER", "1") == "1"
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "./artifacts")
RELOAD_URL = os.getenv("RELOAD_URL", "")            # p.ej. http://localhost:5000/admin/reload
SQL_WINDOW_IN_QUERY = os.getenv("SQL_WINDOW_IN_QUERY", "0") == "1"  # empuja WHERE a SQL
VAL_FRAC = float(os.getenv("VAL_FRAC", "0.15"))      # parte de los datos (tras train) para validación
EPOCHS = int(os.getenv("EPOCHS", "100"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
LOOK_BACK = int(os.getenv("LOOK_BACK", "24"))       # longitud de secuencia

# Reproducibilidad
SEED = int(os.getenv("SEED", "42"))
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

# === MODO PRUEBAS controlado por env =========================
TEST_MODE = os.getenv("TEST_MODE", "0") == "1"  # por defecto desactivado
if TEST_MODE:
    MODE = "cold"                 # arranque desde cero
    PROMOTE_IF_BETTER = False     # no tocar 'current'
    ARTIFACTS_DIR = "./artifacts_pruebas"  # sandbox de artefactos
# =============================================================

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
# 2) CARGA Y PREPARACIÓN DE DATOS
# =========================

# a) Lectura SQL (opcionalmente con WHERE para ventana)
if SQL_WINDOW_IN_QUERY and WINDOW_DAYS > 0:
    sql = text(
        f"""
        WITH mx AS (SELECT MAX(fecha) AS fmax FROM dbo.demanda_peninsula)
        SELECT fecha, valor_real, valor_previsto, valor_programado
        FROM dbo.demanda_peninsula, mx
        WHERE fecha >= DATEADD(day, -{WINDOW_DAYS}, mx.fmax)
        ORDER BY fecha;
        """
    )
else:
    sql = text(
        """
        SELECT fecha, valor_real, valor_previsto, valor_programado
        FROM dbo.demanda_peninsula_semana
        ORDER BY fecha;
        """
    )

df = pd.read_sql(sql, engine, parse_dates=["fecha"]).dropna()

# b) Index/rename/sort
df = (df.set_index("fecha").sort_index()
        .rename(columns={
            "valor_real": "demanda_real",
            "valor_previsto": "demanda_prevista",
            "valor_programado": "demanda_programada",
        }))

# c) Ventana en pandas si no se empujó a SQL
if not SQL_WINDOW_IN_QUERY and WINDOW_DAYS > 0 and len(df) > 0:
    tmax = df.index.max()
    tmin = tmax - pd.Timedelta(days=WINDOW_DAYS)
    df = df.loc[(df.index >= tmin) & (df.index <= tmax)].copy()

# =========================
# 3) SPLIT + ESCALADO (SIN LEAKAGE)
# =========================
features = ["demanda_real", "demanda_prevista", "demanda_programada"]

n = len(df)
if n < 100:
    raise ValueError("Muy pocos datos para un split 70/15/15.")

i_train = int(n * 0.70)
# de lo restante, usar VAL_FRAC para validación
rem = n - i_train
i_val = i_train + int(rem * VAL_FRAC)

# cortes cronológicos
df_train = df.iloc[:i_train]
df_val   = df.iloc[i_train:i_val]
df_test  = df.iloc[i_val:]

# escalado fit SOLO con train
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df_train[features])

train_data = scaler.transform(df_train[features])
val_data   = scaler.transform(df_val[features])
test_data  = scaler.transform(df_test[features])

# =========================
# 4) INGENIERÍA DE SECUENCIAS
# =========================

def create_sequences_multivariate(data: np.ndarray, look_back: int):
    X, Y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back), :])
        Y.append(data[i + look_back, 0])  # objetivo: demanda_real siguiente (col 0)
    return np.array(X), np.array(Y)

X_train, y_train = create_sequences_multivariate(train_data, LOOK_BACK)
X_val,   y_val   = create_sequences_multivariate(val_data,   LOOK_BACK)
X_test,  y_test  = create_sequences_multivariate(test_data,  LOOK_BACK)

print(f"Formas — X_train:{X_train.shape} | X_val:{X_val.shape} | X_test:{X_test.shape}")

# =========================
# 5) MODELO LSTM
# =========================
print("\nDefiniendo y entrenando el modelo LSTM.")
num_features = X_train.shape[2]

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(LOOK_BACK, num_features)))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# --- Warm-start antes de entrenar (desde .keras) ---
_ensure_dirs()
cur_dir, cur_model_p, cur_scaler_p = _current_paths()
if MODE == "warm" and cur_model_p.exists():
    try:
        prev_model = tf.keras.models.load_model(cur_model_p.as_posix())
        model.set_weights(prev_model.get_weights())
        print("[INFO] Warm-start: pesos previos cargados desde modelo .keras.")
    except Exception as e:
        print(f"[WARN] No se pudieron cargar pesos previos: {e}")

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val, y_val),   # ← ahora usa VALIDACIÓN, no test
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
    shuffle=False,
    verbose=0
)

# =========================
# 6) PREDICCIÓN Y MÉTRICAS (en TEST ciego)
# =========================

y_pred_scaled = model.predict(X_test, verbose=0).flatten()

# Invertir escalado (col 0 = demanda_real)
dummy_test = np.zeros((len(y_test), len(features)))
dummy_test[:, 0] = y_test
dummy_test_df = pd.DataFrame(dummy_test, columns=features)
y_test_original = scaler.inverse_transform(dummy_test_df)[:, 0]


dummy_pred = np.zeros((len(y_pred_scaled), len(features)))
dummy_pred[:, 0] = y_pred_scaled
dummy_pred_df = pd.DataFrame(dummy_pred, columns=features)
y_pred_original = scaler.inverse_transform(dummy_pred_df)[:, 0]

# Métricas en MW
rmse = root_mean_squared_error(y_test_original, y_pred_original)
mape = mean_absolute_percentage_error(y_test_original, y_pred_original) * 100
mae  = mean_absolute_error(y_test_original, y_pred_original)

# Métricas normalizadas (con guardia de rango)
rng = float(np.max(y_test_original) - np.min(y_test_original))
nrmse_mean  = rmse / float(np.mean(y_test_original)) * 100
nrmse_range = (rmse / rng * 100) if rng > 0 else float("nan")

print(f"\n[LSTM TEST] RMSE={rmse:.2f} MW | MAE={mae:.2f} MW | MAPE={mape:.2f}% | nRMSE(mean)={nrmse_mean:.2f}%")

# Índices de test alineados con y_test
indices_test = df_test.index[LOOK_BACK:]

# =========================
# 6.1 Baselines: Previsto, Programado y Persistencia
# =========================

y_real = y_test_original
previsto   = df_test['demanda_prevista'].iloc[LOOK_BACK:].to_numpy()
programado = df_test['demanda_programada'].iloc[LOOK_BACK:].to_numpy()
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
# 6.2 Comparativa con baseline de 'current' para promoción (con su scaler)
# =========================

baseline_model_metrics = None
if cur_model_p.exists() and cur_scaler_p.exists():
    try:
        base_m = tf.keras.models.load_model(cur_model_p.as_posix())
        cur_scaler = load(cur_scaler_p.as_posix())

        # Transformar df_test con el scaler del modelo current
        test_data_cur = cur_scaler.transform(df_test[features])
        X_test_cur, _ = create_sequences_multivariate(test_data_cur, LOOK_BACK)

        base_pred_scaled = base_m.predict(X_test_cur, verbose=0).flatten()
        dummy_base = np.zeros((len(base_pred_scaled), len(features)))
        dummy_base[:, 0] = base_pred_scaled
        dummy_base_df = pd.DataFrame(dummy_base, columns=features)
        base_pred_original = cur_scaler.inverse_transform(dummy_base_df)[:, 0]


        base_rmse = root_mean_squared_error(y_test_original, base_pred_original)
        base_mape = mean_absolute_percentage_error(y_test_original, base_pred_original) * 100
        baseline_model_metrics = {"rmse": float(base_rmse), "mape": float(base_mape)}
        print(f"[BASELINE MODEL current] RMSE={base_rmse:.2f} MW | MAPE={base_mape:.2f}%")
    except Exception as e:
        print(f"[WARN] No se pudo evaluar baseline del modelo current (posible incompatibilidad de LOOK_BACK/features): {e}")
elif cur_model_p.exists() and not cur_scaler_p.exists():
    print("[WARN] Modelo current encontrado pero scaler ausente; omitiendo comparación justa.")

# =========================
# 7) VISUALIZACIÓN PREDICCIÓN (overlay con baselines)
# =========================

resultados_test = pd.DataFrame({
    'real': y_real,
    'pred_lstm': y_pred_original,
    'previsto': previsto,
    'programado': programado,
    'persistencia': persistence,
}, index=indices_test)

plt.figure(figsize=(16, 8))
plt.style.use('seaborn-v0_8-whitegrid')
plt.plot(resultados_test.index, resultados_test['real'], label='Demanda Real (MW)', linewidth=1.5)
plt.plot(resultados_test.index, resultados_test['pred_lstm'], label='Predicción LSTM (MW)', linewidth=2.2, alpha=0.9)
plt.plot(resultados_test.index, resultados_test['previsto'], label='Previsto', linestyle='--', alpha=0.8)
plt.plot(resultados_test.index, resultados_test['persistencia'], label='Persistencia t-1', linestyle=':')
plt.title(
    f'Predicción de Demanda | LSTM TEST RMSE: {rmse:.2f} MW | MAE: {mae:.2f} MW | '
    f'MAPE: {mape:.2f}% | nRMSE(media): {nrmse_mean:.2f}%', fontsize=16
)
plt.xlabel("Fecha y Hora"); plt.ylabel("Potencia (MW)")
plt.legend(); plt.tight_layout(); plt.show()

# =========================
# 8) ANOMALÍAS (igual que original, sobre residuo LSTM)
# =========================
K = 2.2; PCT_THRESHOLD = 0.006; FLOOR_MW = 80.0; EWMA_ALPHA = 0.0; MIN_STREAK = 1
WIN_MIN = 180; MIN_FRAC = 0.5; AUTO_CALIB = True; TARGET_RATE = 0.015; CAP_SIGMA_RATIO = 1.3

df_an = resultados_test[["real", "pred_lstm"]].rename(columns={"pred_lstm":"prediccion"}).copy()
df_an["residuos"] = df_an["real"] - df_an["prediccion"]
df_an["res_smooth"] = df_an["residuos"] if EWMA_ALPHA == 0 else df_an["residuos"].ewm(alpha=EWMA_ALPHA, adjust=False).mean()

# util: ventana en puntos
def _infer_window_points(index, minutes=WIN_MIN, min_frac=MIN_FRAC):
    try:
        freq = pd.infer_freq(index)
    except Exception:
        freq = None
    if not freq:
        delta = (pd.Series(index).diff().median() if len(index) >= 2 else pd.Timedelta(minutes=5))
        if pd.isna(delta):
            delta = pd.Timedelta(minutes=5)
    else:
        delta = pd.to_timedelta(freq)
    step_min = max(1, int(delta.total_seconds() // 60))
    win = max(1, int(math.ceil(minutes / step_min)))
    min_periods = max(1, int(math.ceil(win * min_frac)))
    return win, min_periods

win, min_per = _infer_window_points(df_an.index)

# σ rodante + robusto MAD + CAP
res_s = df_an["res_smooth"].shift(1)
sigma_roll = res_s.rolling(win, min_periods=min_per).std()
sigma_exp  = res_s.expanding(min_periods=max(10, min_per//2)).std()
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

# Umbral híbrido y auto-calibración
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

# Etiquetado
is_anom_point = (df_an["res_smooth"].abs() > df_an["umbral_MW"].fillna(np.inf))
if MIN_STREAK > 1:
    streak = (is_anom_point.groupby((~is_anom_point).cumsum())
              .cumcount() + 1) * is_anom_point.astype(int)
    is_anom = streak >= MIN_STREAK
else:
    is_anom = is_anom_point
df_an["anomalia"] = is_anom.astype(int)

n = len(df_an); n_anom = int(df_an["anomalia"].sum())
print("\n=== Anomalías (híbrido σ-rodante) ===")
print(f"Filas: {n} | Anómalos: {n_anom} ({100*n_anom/n:.2f}%) | K≈{K:.2f}")

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
plt.title(f"Detección de Anomalías con LSTM | RMSE test: {rmse:.2f} MW", fontsize=14)
plt.xlabel("Fecha y Hora"); plt.ylabel("Potencia (MW)")
plt.legend(); plt.tight_layout(); plt.show()

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
dump(scaler, run_dir / "scaler_lstm_multivariate.joblib")

# Guardar métricas y anomalías
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
    "epochs": int(len(history.history.get("loss", []))),  # efectivas
    "n_obs_train": int(len(df_train)),
    "n_obs_val": int(len(df_val)),
    "n_obs_test": int(len(df_test)),
    "features": features,
    "targets": ["demanda_real"],
    "run_dir": str(run_dir),
    "seed": SEED,
    "test_mode": TEST_MODE,
}
with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics_out, f, ensure_ascii=False, indent=2)

df_an.to_csv(run_dir / "anomalies.csv", index=True)

# Imprime JSON final (útil para Airflow/XCom o inspección manual)
print(json.dumps(metrics_out, ensure_ascii=False))

best_base_rmse = min(base_prev["rmse"], base_prog["rmse"], base_pers["rmse"])
should_promote = (
    (rmse < best_base_rmse) and
    (baseline_model_metrics is None or (rmse < baseline_model_metrics["rmse"] and mape < baseline_model_metrics["mape"]))
)

if should_promote:
    _promoted = _maybe_promote(run_dir, {"rmse": float(rmse), "mape": float(mape)}, baseline_model_metrics)
    if _promoted:
        print(f"[INFO] Modelo PROMOCIONADO: {run_dir.name}")
else:
    print("[INFO] Modelo NO promocionado: no mejora baselines y/o 'current'.")

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
    X_win_df = df_full[feats].iloc[-look_back:]      
    X_scaled = scaler.transform(X_win_df)           
    X_scaled = np.expand_dims(X_scaled, axis=0)

    y_scaled = model.predict(X_scaled, verbose=0).flatten()[0]
    dummy = np.zeros((1, len(feats))); dummy[0, 0] = y_scaled
    dummy_df = pd.DataFrame(dummy, columns=feats)           # ← NUEVO
    y_pred = scaler.inverse_transform(dummy_df)[0, 0]       # ← NUEVO

    off = _infer_freq(df_full.index)
    t_next = df_full.index[-1] + off
    return t_next, float(y_pred)

def _step_minutes(index: pd.DatetimeIndex) -> int:
    """Devuelve el tamaño de paso en minutos según el índice."""
    try:
        freq = pd.infer_freq(index)
    except Exception:
        freq = None
    if freq:
        delta = pd.to_timedelta(freq)
    else:
        delta = pd.Series(index).diff().median()
    if pd.isna(delta):
        delta = pd.Timedelta(minutes=5)
    return max(1, int(delta.total_seconds() // 60))


def forecast_multi_step(model, scaler, df_hist, look_back, horizon_steps,
                        future_cov: pd.DataFrame | None = None,
                        fallback: str = "persist"):
    """
    Predice varios pasos hacia delante encadenando (closed-loop).
    - df_hist: DataFrame histórico con columnas ['demanda_real','demanda_prevista','demanda_programada'].
    - future_cov: DataFrame opcional indexado por timestamps futuros con columnas
      ['demanda_prevista','demanda_programada'] para cada paso. Si None o faltan
      valores, se aplica 'fallback' (persistencia del último valor).
    - fallback: 'persist' (por defecto) o 'zero'.
    Devuelve un pd.Series con las predicciones en MW indexadas por fecha futura.
    """
    feats = ["demanda_real", "demanda_prevista", "demanda_programada"]
    df_hist = df_hist.sort_index()

    # Ventana de trabajo en unidades REALES (no escaladas)
    window = df_hist[feats].values[-look_back:, :].copy()
    off = _infer_freq(df_hist.index)
    t = df_hist.index[-1]

    preds, times = [], []
    for step in range(1, horizon_steps + 1):
        # Escalar entrada y predecir
        win_df = pd.DataFrame(window, columns=feats)     
        x_scaled = scaler.transform(win_df)             
        y_scaled = model.predict(np.expand_dims(x_scaled, 0), verbose=0).flatten()[0]
        dummy = np.zeros((1, len(feats))); dummy[0, 0] = y_scaled
        dummy_df = pd.DataFrame(dummy, columns=feats)        # ← NUEVO
        y_hat = scaler.inverse_transform(dummy_df)[0, 0]     # ← NUEVO


        # Timestamp futuro
        t = t + off

        # Covariables para t
        if (future_cov is not None) and (t in future_cov.index):
            prev_hat = float(future_cov.loc[t, "demanda_prevista"])
            prog_hat = float(future_cov.loc[t, "demanda_programada"])
            if np.isnan(prev_hat) or np.isnan(prog_hat):
                prev_hat = window[-1, 1] if fallback == "persist" else 0.0
                prog_hat = window[-1, 2] if fallback == "persist" else 0.0
        else:
            # Fallback si no hay covariables futuras
            prev_hat = window[-1, 1] if fallback == "persist" else 0.0
            prog_hat = window[-1, 2] if fallback == "persist" else 0.0

        # Desplazar ventana: añadimos la fila futura (y_hat, prev_hat, prog_hat)
        new_row = np.array([y_hat, prev_hat, prog_hat])
        window = np.vstack([window[1:], new_row])

        preds.append(y_hat); times.append(t)

    return pd.Series(preds, index=pd.DatetimeIndex(times, name="fecha"), name="pred_lstm")


# Ejemplo de inferencia (atrapado para no romper si faltan deps)
try:
    t_pred, y_hat = predict_next_step(model, scaler, df, look_back=LOOK_BACK)
    print(f"[INFERENCIA] Predicción one-step-ahead para {t_pred}: {y_hat:.2f} MW")
except Exception as e:
    print(f"[WARN] No se pudo ejecutar la inferencia: {e}")

# =========================
# 10.1) PRONÓSTICO MULTI-STEP (FORZADO)
# =========================
HORIZON_MIN = 1440          # ← 24h fijo
FALLBACK = "persist"        # ← usa último valor si faltan covariables
print(f"[CFG] Multi-step hardcoded: {HORIZON_MIN} min, fallback={FALLBACK}")

try:
    step_min = _step_minutes(df.index)
    steps = max(1, HORIZON_MIN // step_min)

    future_index = pd.date_range(
        df.index[-1] + _infer_freq(df.index),
        periods=steps, freq=_infer_freq(df.index)
    )
    future_cov = None
    if set(["demanda_prevista","demanda_programada"]).issubset(df.columns):
        _cov = df.reindex(future_index)[["demanda_prevista","demanda_programada"]]
        if not _cov.isna().all().all():
            future_cov = _cov

    yhat_ms = forecast_multi_step(
        model, scaler, df, LOOK_BACK, steps,
        future_cov=future_cov, fallback=FALLBACK
    )

    out_path = run_dir / f"forecast_{HORIZON_MIN}min.csv"
    yhat_ms.to_csv(out_path, header=True)
    print(f"[FORECAST] Guardado multi-step ({HORIZON_MIN} min, {steps} pasos): {out_path}")
    print(yhat_ms.head(3).to_string())
    print(yhat_ms.tail(3).to_string())
except Exception as e:
    print(f"[ERROR] Multi-step: {e}")
