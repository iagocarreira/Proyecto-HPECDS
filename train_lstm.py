#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Entrenamiento LSTM multivariante con opción warm/cold, ventana deslizante,
artefactos versionados, evaluación y promoción opcional.

Uso típico:
  python train_lstm_timeseries.py \
      --mode warm \
      --window-days 180 \
      --timestamp-col fecha_hora \
      --feature-cols "f1,f2,f3" \
      --target-cols "f1" \
      --table dbo.tu_tabla \
      --artifacts /artifacts \
      --seq-len 48 \
      --epochs 30 \
      --batch-size 128 \
      --promote-if-better \
      --reload-url http://api_model:5000/admin/reload

Variables de entorno para SQL (Azure SQL / SQL Server):
  SQL_SERVER, SQL_DB, SQL_USER, SQL_PASSWORD
(Alternativamente puedes pasar --sql-query "SELECT ..." en vez de --table)
"""

import os
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from joblib import dump, load

import sqlalchemy as sa

# TensorFlow/Keras
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers


# ------------------------
# Utilidades de datos
# ------------------------
def make_engine():
    server = os.getenv("SQL_SERVER")
    db = os.getenv("SQL_DB")
    user = os.getenv("SQL_USER")
    pwd = os.getenv("SQL_PASSWORD")

    if not all([server, db, user, pwd]):
        raise RuntimeError("Faltan variables de entorno SQL_SERVER/SQL_DB/SQL_USER/SQL_PASSWORD")

    # Driver ODBC 18
    params = (
        "Driver={{ODBC Driver 18 for SQL Server}};"
        "Server={server};Database={db};Uid={user};Pwd={pwd};Encrypt=yes;TrustServerCertificate=no;Connection Timeout=30;"
    ).format(server=server, db=db, user=user, pwd=pwd)
    engine = sa.create_engine(f"mssql+pyodbc:///?odbc_connect={sa.engine.URL.quote(params)}")
    return engine


def read_data(args) -> pd.DataFrame:
    eng = make_engine()
    if args.sql_query:
        df = pd.read_sql(args.sql_query, eng)
    else:
        if not args.table:
            raise ValueError("Debes indicar --table o --sql-query")
        df = pd.read_sql(f"SELECT * FROM {args.table}", eng)

    if args.timestamp_col not in df.columns:
        raise ValueError(f"No se encontró la columna de timestamp '{args.timestamp_col}'")

    # Orden temporal
    df[args.timestamp_col] = pd.to_datetime(df[args.timestamp_col], utc=False, errors="coerce")
    df = df.dropna(subset=[args.timestamp_col]).sort_values(args.timestamp_col)

    # Filtro por ventana deslizante (últimos N días)
    if args.window_days > 0:
        tmax = df[args.timestamp_col].max()
        tmin = tmax - pd.Timedelta(days=args.window_days)
        df = df.loc[df[args.timestamp_col].between(tmin, tmax)].copy()

    # Selección de columnas
    feats = [c.strip() for c in args.feature_cols.split(",") if c.strip()]
    targs = [c.strip() for c in args.target_cols.split(",") if c.strip()]
    for c in feats + targs:
        if c not in df.columns:
            raise ValueError(f"La columna '{c}' no existe en el dataset")

    # Garantizar numéricos
    df[feats + targs] = df[feats + targs].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=feats + targs)
    df = df.reset_index(drop=True)
    return df, feats, targs


def make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len, :])
        ys.append(y[i + seq_len, :])  # predicción one-step-ahead
    return np.array(Xs), np.array(ys)


# ------------------------
# Modelo
# ------------------------
def build_model(n_features: int, seq_len: int, lr: float = 1e-3) -> tf.keras.Model:
    model = models.Sequential(
        [
            layers.Input(shape=(seq_len, n_features)),
            layers.LSTM(64, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(n_features),  # salida multivariante (para targets⊆features se indexa luego)
        ]
    )
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss="mse")
    return model


# ------------------------
# Métricas y anomalías
# ------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    # y_* shape: [n, d]
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(mean_absolute_error(y_true, y_pred))
    # Evita div/0 en MAPE
    denom = np.where(np.abs(y_true) < 1e-8, 1.0, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    # nRMSE-mean y nRMSE-range
    y_flat = y_true.flatten()
    nrmse_mean = float(rmse / (np.mean(np.abs(y_flat)) + 1e-9))
    y_range = float(np.max(y_flat) - np.min(y_flat) + 1e-9)
    nrmse_range = float(rmse / y_range)
    return dict(rmse=rmse, mae=mae, mape=mape, nrmse_mean=nrmse_mean, nrmse_range=nrmse_range)


def detect_anomalies(errors: pd.Series, k_sigma: float = 3.0, min_pct: float = 5.0, floor: float = 0.0):
    """
    Umbral híbrido: max(k·σ_rodante, pct% del valor, floor)
    """
    roll = errors.rolling(window=288, min_periods=30)  # ~2 días si datos 10min (ajusta a tu frecuencia)
    sigma = roll.std().fillna(errors.std())
    thresh_sigma = k_sigma * sigma

    thresh_pct = (min_pct / 100.0) * errors.abs().rolling(window=288, min_periods=30).mean().fillna(errors.abs().mean())
    thr = np.maximum.reduce([thresh_sigma.values, thresh_pct.values, np.full_like(thresh_sigma.values, floor)])
    flags = (errors.abs().values > thr).astype(int)
    return pd.Series(thr, index=errors.index), pd.Series(flags, index=errors.index)


# ------------------------
# Promoción y utilidades de artefactos
# ------------------------
def ensure_dirs(artifacts_dir: Path):
    artifacts_dir.mkdir(parents=True, exist_ok=True)


def current_paths(artifacts_dir: Path):
    cur = artifacts_dir / "current"
    model_p = cur / "modelo_lstm_multivariate.keras"
    scaler_p = cur / "scaler_lstm_multivariate.joblib"
    return cur, model_p, scaler_p


def evaluate_baseline(cur_model_path: Path, X_test: np.ndarray, y_test_targets: np.ndarray, target_idx: np.ndarray):
    if not cur_model_path.exists():
        return None
    try:
        m = tf.keras.models.load_model(cur_model_path.as_posix())
        y_pred_full = m.predict(X_test, verbose=0)
        y_pred = y_pred_full[:, target_idx]
        return compute_metrics(y_test_targets, y_pred)
    except Exception as e:
        print(f"[WARN] No se pudo evaluar baseline: {e}")
        return None


def maybe_promote(args, run_dir: Path, metrics: dict, baseline: dict | None):
    # Condición de mejora: RMSE y MAPE menores (o baseline ausente)
    improved = False
    if baseline is None:
        improved = True
    else:
        improved = (metrics["rmse"] < baseline.get("rmse", 1e9)) and (metrics["mape"] < baseline.get("mape", 1e9))

    if args.promote_if_better and improved:
        artifacts_dir = Path(args.artifacts)
        cur_link = artifacts_dir / "current"
        # symlink relativo para portabilidad
        if cur_link.exists() or cur_link.is_symlink():
            cur_link.unlink()
        cur_link.symlink_to(run_dir.name)
        print(f"[INFO] Promocionado: {run_dir.name} -> {cur_link}")

        # Llamada a API para recarga (opcional)
        if args.reload_url:
            try:
                import requests

                r = requests.post(args.reload_url, json={"reason": "auto-promotion", "run_dir": str(run_dir)}, timeout=8)
                print(f"[INFO] Reload API status: {r.status_code}")
            except Exception as e:
                print(f"[WARN] No se pudo llamar a reload API: {e}")

    return improved


# ------------------------
# MAIN
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    # Datos
    parser.add_argument("--table", type=str, default="")
    parser.add_argument("--sql-query", type=str, default="")
    parser.add_argument("--timestamp-col", type=str, required=True)
    parser.add_argument("--feature-cols", type=str, required=True, help="coma-separadas")
    parser.add_argument("--target-cols", type=str, required=True, help="coma-separadas (⊆ features)")

    parser.add_argument("--window-days", type=int, default=180)
    parser.add_argument("--seq-len", type=int, default=48)
    # Entrenamiento
    parser.add_argument("--mode", choices=["warm", "cold"], default="warm")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-split", type=float, default=0.2)
    # Artefactos y salida
    parser.add_argument("--artifacts", type=str, default="./artifacts")
    parser.add_argument("--emit-metrics", type=str, default="", help="ruta JSON de métricas (también se imprime)")
    parser.add_argument("--promote-if-better", action="store_true")
    parser.add_argument("--reload-url", type=str, default="")  # p.ej. http://api_model:5000/admin/reload
    # Anomalías
    parser.add_argument("--k-sigma", type=float, default=3.0)
    parser.add_argument("--min-pct", type=float, default=5.0)
    parser.add_argument("--floor", type=float, default=0.0)

    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts)
    ensure_dirs(artifacts_dir)

    # Carga datos
    df, feats, targs = read_data(args)
    ts_col = args.timestamp_col

    # Split temporal: 80/20 final para test (o usa val_split)
    n = len(df)
    if n < (args.seq_len + 10):
        raise ValueError("Muy pocos datos tras aplicar la ventana; reduce --seq-len o aumenta --window-days")

    split_idx = int(n * (1.0 - 0.2))
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()

    # Escalado con MinMax (ajustado SOLO en train)
    scaler = MinMaxScaler()
    scaler.fit(df_train[feats].values)

    X_train_full = scaler.transform(df_train[feats].values)
    X_test_full = scaler.transform(df_test[feats].values)

    y_train_full = scaler.transform(df_train[feats].values)  # salida densa multivar
    y_test_full = scaler.transform(df_test[feats].values)

    # Índices de targets dentro de features
    target_idx = np.array([feats.index(t) for t in targs], dtype=int)

    # Secuencias
    X_train, y_train_fullstep = make_sequences(X_train_full, y_train_full, args.seq_len)
    X_test, y_test_fullstep = make_sequences(X_test_full, y_test_full, args.seq_len)

    # Recorta y aísla targets para métricas
    y_train_targets = y_train_fullstep[:, target_idx]
    y_test_targets = y_test_fullstep[:, target_idx]

    # Modelo
    model = build_model(n_features=len(feats), seq_len=args.seq_len, lr=args.learning_rate)

    # Warm start si procede
    cur_dir, cur_model_p, cur_scaler_p = current_paths(artifacts_dir)
    if args.mode == "warm" and cur_model_p.exists():
        try:
            model.load_weights(cur_model_p.as_posix())
            print("[INFO] Warm-start: pesos previos cargados.")
        except Exception as e:
            print(f"[WARN] No se pudieron cargar pesos previos: {e}")

    # Entrenamiento
    es = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    hist = model.fit(
        X_train,
        y_train_fullstep,  # salida multivar
        validation_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        verbose=1,
        callbacks=[es],
        shuffle=False,
    )

    # Predicción y métricas en test
    y_pred_full = model.predict(X_test, verbose=0)
    y_pred_targets = y_pred_full[:, target_idx]
    metrics = compute_metrics(y_test_targets, y_pred_targets)

    # Reconstrucción de índices temporales alineados al final de secuencias
    idx_test = df_test[ts_col].iloc[args.seq_len:].reset_index(drop=True)

    # Errores por timestamp (norma L2 sobre targets)
    err_vec = np.sqrt(np.mean((y_test_targets - y_pred_targets) ** 2, axis=1))
    s_err = pd.Series(err_vec, index=idx_test, name="error")

    # Umbral y flags de anomalía
    thr, flags = detect_anomalies(s_err, k_sigma=args.k_sigma, min_pct=args.min_pct, floor=args.floor)
    anom = pd.DataFrame(
        {
            ts_col: idx_test,
            "error": s_err.values,
            "threshold": thr.values,
            "is_anomaly": flags.values.astype(int),
        }
    )

    # --------- Evalúa baseline si existe ----------
    baseline = evaluate_baseline(cur_model_p, X_test, y_test_targets, target_idx)
    if baseline:
        metrics["baseline_rmse"] = baseline["rmse"]
        metrics["baseline_mape"] = baseline["mape"]

    # --------- Guardado de artefactos versionados ----------
    ts_run = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = artifacts_dir / f"run_{ts_run}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Guardar modelo y scaler
    model.save((run_dir / "modelo_lstm_multivariate.keras").as_posix())
    dump(scaler, run_dir / "scaler_lstm_multivariate.joblib")

    # Guardar métricas y anomalías
    metrics_out = {
        **metrics,
        "run_dir": str(run_dir),
        "mode": args.mode,
        "window_days": args.window_days,
        "seq_len": args.seq_len,
        "epochs": int(es.stopped_epoch or args.epochs),
        "n_obs_train": int(len(df_train)),
        "n_obs_test": int(len(df_test)),
        "features": feats,
        "targets": targs,
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, ensure_ascii=False, indent=2)

    anom_path = run_dir / "anomalies.csv"
    anom.to_csv(anom_path, index=False)

    # Imprime JSON de métricas al stdout (útil para Airflow XCom)
    print(json.dumps(metrics_out, ensure_ascii=False))

    # --------- Promoción opcional + recarga de API ----------
    if args.promote_if_better:
        improved = maybe_promote(args, run_dir, metrics, baseline)
        metrics_out["promoted"] = bool(improved)
        with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics_out, f, ensure_ascii=False, indent=2)

    # Fin OK
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
