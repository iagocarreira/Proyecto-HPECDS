#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Principal (Flask) v5 - MERGED
- Compatible con modelo v2 (x_scaler, y_scaler) [Iago]
- Cache de "warm-up" para la predicción "latest" [Iago]
- Integración de anomalías en la vista de predicción [Iago]
- Corregidas advertencias de Pandas (FutureWarning) y Sklearn (UserWarning) [Iago]
- Endpoints API para chatbot [Tu código]
- Configuración de rutas con Path y .env [Tu código]
- (NUEVO) Unión semanal + histórica para consultas y predicción
- (NUEVO) Endpoint /api/anomalies_range para rangos largos
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import io
import pandas as pd
import numpy as np
from flask import jsonify
from pathlib import Path
from dotenv import load_dotenv

# Configurar el backend de Matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from datetime import datetime, timedelta

from flask import Flask, request, render_template
from flask_cors import CORS

# Importaciones de Modelado y BD
from sqlalchemy import create_engine, MetaData, Table, select, and_
from tensorflow.keras.models import load_model
import joblib

# ---------- Base dir del fichero (no del CWD) ----------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent  # Raíz del proyecto (un nivel arriba)

# Cargar .env desde la raíz del proyecto
load_dotenv(PROJECT_ROOT / ".env")

def _resolve_path(p: str) -> Path:
    """Si p es relativo, lo resuelve respecto a la RAÍZ del proyecto."""
    pp = Path(p)
    return (pp if pp.is_absolute() else (PROJECT_ROOT / pp)).resolve()

# --- CONFIGURACIÓN GLOBAL Y PARÁMETROS CRÍTICOS ---
app = Flask(__name__)
CORS(app)  # Para el chatbot

MODELS = {}
X_SCALER, Y_SCALER = None, None
LOOK_BACK = int(os.getenv("LOOK_BACK", "24"))
NUM_FEATURES = 3 # real, previsto, programado
PREDICTION_STEPS = 288 # 24h * 12 (intervalos de 5 min)
PREDICTION_DATA_WINDOW_DAYS, API_DATA_URL = 3, "http://127.0.0.1:5002"

# Configuración de rutas (tu código)
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/current/model.keras")
X_SCALER_PATH = os.getenv("X_SCALER_PATH", "artifacts/current/x_scaler.joblib")
Y_SCALER_PATH = os.getenv("Y_SCALER_PATH", "artifacts/current/y_scaler.joblib")

MODEL_PATH = _resolve_path(MODEL_PATH)
X_SCALER_PATH = _resolve_path(X_SCALER_PATH)
Y_SCALER_PATH = _resolve_path(Y_SCALER_PATH)

print("[CWD]           ", Path.cwd())
print("[BASE_DIR]      ", BASE_DIR)
print("[PROJECT_ROOT]  ", PROJECT_ROOT)
print("[MODEL_PATH abs]", MODEL_PATH)
print("[X_SCALER_PATH abs]", X_SCALER_PATH)
print("[Y_SCALER_PATH abs]", Y_SCALER_PATH)

LATEST_PRED_CACHE = {
    "data_json": None,
    "img_base64": None,
    "anomaly_results": None,
    "error": "Cache no inicializado."
}

# --- Conexión a Azure SQL ---
SERVER, DATABASE, USER, PASSWORD = "udcserver2025.database.windows.net", "grupo_1", "ugrupo1", "HK9WXIJaBp2Q97haePdY"
ENGINE_URL = (f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER}:1433/{DATABASE}?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=yes")
engine, tabla_semanal, tabla_historica = None, None, None
try:
    engine = create_engine(ENGINE_URL, pool_pre_ping=True)
    meta = MetaData()
    tabla_semanal = Table("demanda_peninsula_semana", meta, autoload_with=engine, schema="dbo")
    tabla_historica = Table("demanda_peninsula", meta, autoload_with=engine, schema="dbo")
    print("INFO: Conexión a la BD establecida.")
except Exception as e:
    print(f"ERROR CRÍTICO: No se pudo conectar a la base de datos. {e}")


# --- Utilidades de inferencia (Iago) ---

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
            delta = pd.Timedelta(minutes=5)
        return pd.tseries.frequencies.to_offset(delta)
    return pd.tseries.frequencies.to_offset(pd.Timedelta(minutes=5))

def _step_minutes(index: pd.DatetimeIndex) -> int:
    try:
        freq = pd.infer_freq(index)
    except Exception:
        freq = None
    delta = pd.to_timedelta(freq) if freq else pd.Series(index).diff().median()
    if pd.isna(delta):
        delta = pd.Timedelta(minutes=5)
    return max(1, int(delta.total_seconds() // 60))

def forecast_multi_step(model, x_scaler, y_scaler, df_hist, look_back, horizon_steps,
                        future_cov: pd.DataFrame | None = None,
                        fallback: str = "persist",
                        feats=("valor_real","valor_previsto","valor_programado")):
    hist = df_hist.copy(); cols = list(feats)
    preds = []
    cur_hist = hist.copy()
    step = _step_minutes(hist.index)
    if step == 0: step = 5

    # --- Mapa de renombrado para el scaler (Iago) ---
    RENAME_MAP_SCALER = {
        "valor_real": "demanda_real",
        "valor_previsto": "demanda_prevista",
        "valor_programado": "demanda_programada"
    }

    for _ in range(horizon_steps):
        win = cur_hist.iloc[-look_back:].copy()
        
        for f in feats:
            if f not in win.columns:
                win[f] = 0.0
        
        # --- Renombrar temporalmente antes de escalar (Iago) ---
        win_renamed = win.rename(columns=RENAME_MAP_SCALER)
        X = x_scaler.transform(win_renamed[x_scaler.feature_names_in_])
        
        X = np.expand_dims(X, axis=0)

        y_scaled = float(model.predict(X, verbose=0).flatten()[0])
        y_scaled = float(np.clip(y_scaled, 0.0, 1.0))
        y_next = float(y_scaler.inverse_transform([[y_scaled]])[0,0])

        t_next = cur_hist.index[-1] + pd.Timedelta(minutes=step)
        preds.append((t_next, y_next))

        row_data = {
            "valor_real": y_next,
            "valor_previsto": win.iloc[-1]["valor_previsto"],
            "valor_programado": win.iloc[-1]["valor_programado"]
        }

        if (future_cov is not None) and (t_next in future_cov.index):
            _cov = future_cov.loc[t_next]
            for c in ("valor_previsto", "valor_programado"):
                if c in _cov.index and not pd.isna(_cov[c]):
                    row_data[c] = _cov[c]
        else:
            if fallback == "zero":
                row_data["valor_previsto"] = 0.0
                row_data["valor_programado"] = 0.0
        
        # --- Usar pd.concat para evitar FutureWarning (Iago) ---
        new_df_row = pd.DataFrame(row_data, index=[t_next])
        cur_hist = pd.concat([cur_hist, new_df_row])

    return pd.Series([v for _, v in preds], index=[t for t, _ in preds], name="forecast")

# --- Carga de Modelos v2 (Iago con tus rutas) ---
def load_models_at_startup():
    global MODELS, X_SCALER, Y_SCALER
    try:
        X_SCALER = joblib.load(X_SCALER_PATH)
        Y_SCALER = joblib.load(Y_SCALER_PATH)
        print("INFO: Scalers X/Y cargados correctamente.")
    except Exception as e:
        print(f"ERROR CRÍTICO al cargar scalers (x_scaler/y_scaler): {e}")
    
    try:
        MODELS['LSTM'] = load_model(MODEL_PATH)
        print(f"INFO: Modelo LSTM cargado desde {MODEL_PATH}.")
    except Exception as e:
        print(f"ERROR al cargar el modelo LSTM: {e}")

# --- fetch_data_from_db ---
def fetch_data_from_db(table_to_query: Table, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    if not engine or table_to_query is None: return pd.DataFrame()
    query = select(table_to_query).where(and_(table_to_query.c.fecha >= start_date, table_to_query.c.fecha <= end_date)).order_by(table_to_query.c.fecha)
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            if not df.empty and 'fecha' in df.columns:
                df['fecha'] = pd.to_datetime(df['fecha'])
                df = df.set_index('fecha').sort_index()
            print(f"DEBUG: Se obtuvieron {len(df)} registros de '{table_to_query.name}'")
            return df
    except Exception as e:
        print(f"ERROR al consultar '{table_to_query.name}': {e}")
        return pd.DataFrame()

# --- NUEVO: unión semanal + histórica, priorizando semanal en solapes ---
def fetch_data_union(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Devuelve datos combinando semanal + histórica, priorizando semanal cuando hay solapes.
    """
    if (tabla_semanal is None) and (tabla_historica is None):
        return pd.DataFrame()

    df_w = fetch_data_from_db(tabla_semanal, start_date, end_date) if tabla_semanal is not None else pd.DataFrame()
    df_h = fetch_data_from_db(tabla_historica, start_date, end_date) if tabla_historica is not None else pd.DataFrame()

    if df_w.empty and df_h.empty:
        return pd.DataFrame()
    if df_w.empty:
        return df_h
    if df_h.empty:
        return df_w

    combined = df_w.combine_first(df_h)
    combined = combined.sort_index()
    return combined

# --- Lógica de predicción (Iago) ---
def generate_prediction_series(target_date: str | None = None):
    
    if 'LSTM' not in MODELS:
        return None, None, "Error: Modelo LSTM no cargado."
    if X_SCALER is None or Y_SCALER is None:
        return None, None, "Error crítico: Scalers X/Y no cargados."

    features = ['valor_real', 'valor_previsto', 'valor_programado']
    
    df_history = pd.DataFrame()
    df_future_covariates = pd.DataFrame()
    df_actuals_for_plot = pd.DataFrame()
    prediction_start_time = None
    freq_str = '5min'

    try:
        if target_date:
            # MODO 1: Predecir un día específico (pasado)
            prediction_day_start = datetime.strptime(target_date, "%Y-%m-%d")
            prediction_start_time = prediction_day_start

            end_history = prediction_day_start - timedelta(microseconds=1)
            start_history = end_history - timedelta(days=PREDICTION_DATA_WINDOW_DAYS)
            end_prediction_day = prediction_day_start + timedelta(days=1) - timedelta(microseconds=1)

            # Usa ambas tablas unificadas (prioriza semanal)
            df_history = fetch_data_union(start_history, end_history)
            df_future_covariates = fetch_data_union(prediction_day_start, end_prediction_day)

            if df_future_covariates.empty:
                return None, None, f"No hay datos de 'previsto/programado' para {target_date}."

            if df_history.empty or len(df_history) < LOOK_BACK:
                return None, None, f"Histórico insuficiente para {target_date}. Necesitas ~{LOOK_BACK} puntos previos."

            df_actuals_for_plot = df_future_covariates.copy()
        
        else:
            # MODO 2: Predecir "a partir de ahora"
            end_dt, start_dt = datetime.now(), datetime.now() - timedelta(days=PREDICTION_DATA_WINDOW_DAYS)

            # Historia reciente uniendo ambas tablas
            df_history = fetch_data_union(start_dt, end_dt)
            
            if df_history.empty:
                return None, None, "No se encontraron datos históricos recientes."
            if len(df_history) < LOOK_BACK:
                return None, None, f"Datos históricos insuficientes ({len(df_history)}). Se necesitan {LOOK_BACK}."
            
            freq_obj = _infer_freq(df_history.index)
            freq_str = freq_obj.freqstr
            prediction_start_time = df_history.index[-1] + freq_obj
            df_actuals_for_plot = df_history.copy()
            
            # Futuros de respaldo: mismo tramo hace 1 año, usando la unión
            start_proxy = prediction_start_time - timedelta(days=365)
            end_proxy = start_proxy + timedelta(minutes=PREDICTION_STEPS * _step_minutes(df_history.index))
            df_proxy_futures = fetch_data_union(start_proxy, end_proxy)
            
            if len(df_proxy_futures) < PREDICTION_STEPS:
                return None, None, "No hay datos de 'previsto/programado' suficientes para construir los próximos 5 min × 24h."
            
            proxy_index = pd.date_range(start=prediction_start_time, periods=len(df_proxy_futures), freq=freq_obj)
            df_proxy_futures.index = proxy_index
            df_future_covariates = df_proxy_futures.copy()

        if df_history.empty:
            return None, None, "df_history está vacío, no se puede predecir."
        
        df_history[features] = df_history[features].ffill().fillna(0)
        df_future_covariates[features] = df_future_covariates[features].ffill().fillna(0)
        
        predictions_series = forecast_multi_step(
            MODELS['LSTM'], X_SCALER, Y_SCALER,
            df_hist=df_history,
            look_back=LOOK_BACK,
            horizon_steps=PREDICTION_STEPS,
            future_cov=df_future_covariates,
            fallback="persist",
            feats=['valor_real', 'valor_previsto', 'valor_programado']
        )
        
        return predictions_series, df_actuals_for_plot, None

    except Exception as e:
        print(f"ERROR en generate_prediction_series: {e}")
        import traceback
        traceback.print_exc()
        return None, None, f"Error interno durante la predicción: {e}"

# --- Función de ploteo (Iago) ---
def generate_prediction_plot_image(predictions_series, df_actuals, target_date: str = None):
    try:
        plt.figure(figsize=(14, 7)); plt.style.use('seaborn-v0_8-whitegrid')
        
        real_demand, dates_real = pd.Series([], dtype=float), pd.Series([], dtype='datetime64[ns]')
        if df_actuals is not None and not df_actuals.empty and 'valor_real' in df_actuals.columns:
            real_demand = df_actuals['valor_real']
            dates_real = pd.to_datetime(df_actuals.index)
            plt.plot(dates_real, real_demand, label='Demanda Real (MW)', color='steelblue', marker='.', markersize=4)

        dates_pred = pd.to_datetime(predictions_series.index)
        predictions_real_mw = predictions_series.values
        plt.plot(dates_pred, predictions_real_mw, label='Predicción LSTM', color='darkorange', linestyle='--', linewidth=2)

        if target_date:
            title_date = f"para el día {target_date}"
        else:
            title_date = f"a partir de {predictions_series.index[0].strftime('%Y-%m-%d %H:%M')}"
            
        plt.title(f'Predicción de Demanda con LSTM {title_date}', fontsize=16)
        plt.xlabel("Tiempo"); plt.ylabel("Potencia (MW)"); plt.legend()
        
        all_values = pd.concat([pd.Series(predictions_real_mw), real_demand])
        if not all_values.dropna().empty:
            plt.ylim(all_values.min()*0.9, all_values.max()*1.1)
            
        plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
        
        img_data = io.BytesIO(); plt.savefig(img_data, format='png'); plt.close()
        img_data.seek(0)
        return img_data, None
    except Exception as e:
        print(f"Error generando plot: {e}")
        return None, f"Error al generar la imagen: {e}"

# --- Helper para Anomalías (Iago) ---
def _get_anomaly_results(predictions_series, df_actuals, thresh=2.5):
    if df_actuals is None or 'valor_real' not in df_actuals.columns:
        return {"anomalies": [], "error": "No hay datos reales para analizar."}

    common_index = predictions_series.index.intersection(df_actuals.index)
    if common_index.empty:
        return {"anomalies": [], "error": "No hay solapamiento entre predicción y reales."}

    y_pred = predictions_series[common_index]
    y_true = df_actuals.loc[common_index, 'valor_real']
    
    valid_idx = y_true.notna() & y_pred.notna()
    residuals = (y_true[valid_idx] - y_pred[valid_idx])
    
    if residuals.empty:
        return {"anomalies": []}

    rs = residuals.values
    mu, sd = float(np.mean(rs)), float(np.std(rs) + 1e-9)
    
    anomalies = []
    for ts, residual in residuals.items():
        z = (residual - mu) / sd
        if abs(z) >= thresh:
            anomalies.append({
                "ts": ts.isoformat(), 
                "residual": float(residual), 
                "z": float(z), 
                "severity": min(abs(z)/5.0, 1.0)
            })
    return {"mu": mu, "sd": sd, "threshold": thresh, "anomalies": anomalies}


# --- Función de "Warm-Up" (Iago) ---
def warm_up_prediction_cache():
    global LATEST_PRED_CACHE
    print("INFO: [WARM-UP] Iniciando predicción de 'últimos datos'...")

    predictions_series, df_actuals, err = generate_prediction_series(target_date=None)
    
    if err:
        print(f"ERROR: [WARM-UP] Falló la predicción: {err}")
        LATEST_PRED_CACHE = {"data_json": None, "img_base64": None, "anomaly_results": None, "error": err}
        return

    try:
        data_json = {
            "model": "LSTM",
            "interval_min": _step_minutes(predictions_series.index),
            "start": predictions_series.index[0].isoformat(),
            "end": predictions_series.index[-1].isoformat(),
            "points": [{"ts": d.isoformat(), "y": float(v)} for d, v in predictions_series.items()],
        }
    except Exception as e:
        LATEST_PRED_CACHE["error"] = f"Error al crear JSON: {e}"
        return

    img_data_buffer, img_err = generate_prediction_plot_image(predictions_series, df_actuals, target_date=None)
    if img_err:
        LATEST_PRED_CACHE["error"] = img_err
        return
    img_base64 = base64.b64encode(img_data_buffer.read()).decode('utf-8')

    anomaly_results = _get_anomaly_results(predictions_series, df_actuals)

    LATEST_PRED_CACHE = {
        "data_json": data_json,
        "img_base64": img_base64,
        "anomaly_results": anomaly_results,
        "error": None
    }
    print("INFO: [WARM-UP] Predicción (JSON, Imagen y Anomalías) cacheada correctamente.")


# --- INICIALIZACIÓN Y ENDPOINTS (HTML) ---
load_models_at_startup() 
warm_up_prediction_cache()

@app.route("/")
def home(): 
    return render_template("index.html", api_data_url=API_DATA_URL)

@app.route("/predict_latest/<model_name>", methods=["GET"])
def predict_latest(model_name):
    cache = LATEST_PRED_CACHE
    if cache["error"]:
        return render_template("plot_view.html", error=cache["error"], model='LSTM')
    
    return render_template("plot_view.html", 
                           model='LSTM', 
                           img_data=cache["img_base64"],
                           anomaly_results=cache["anomaly_results"],
                           date="Últimos datos disponibles")

@app.route("/predict_custom", methods=["POST"])
def predict_custom():
    prediction_date = request.form.get("desde_fecha")
    if not prediction_date:
        return render_template("plot_view.html", error="No se seleccionó ninguna fecha.", model='LSTM')
        
    predictions_series, df_actuals, error = generate_prediction_series(target_date=prediction_date)
    if error:
        return render_template("plot_view.html", error=error, model='LSTM')

    img_data_buffer, img_error = generate_prediction_plot_image(predictions_series, df_actuals, target_date=prediction_date)
    if img_error:
        return render_template("plot_view.html", error=img_error, model='LSTM')
    img_base64 = base64.b64encode(img_data_buffer.read()).decode('utf-8')

    anomaly_results = _get_anomaly_results(predictions_series, df_actuals)
    
    return render_template("plot_view.html", 
                           model='LSTM', 
                           img_data=img_base64, 
                           date=prediction_date,
                           anomaly_results=anomaly_results)

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/documentacion")
def documentation():
    return render_template("documentacion.html")

@app.route("/health")
def health_check():
    return jsonify({"status": "ok"}), 200

# --- ENDPOINTS (API) para chatbot ---

@app.get("/api/predictions_latest")
def api_predictions_latest():
    cache = LATEST_PRED_CACHE
    if cache["error"]:
        return jsonify({"error": cache["error"]}), 400
    return jsonify(cache["data_json"])

@app.get("/api/predictions")
def api_predictions_for_date():
    date = request.args.get("date")
    if not date:
        return jsonify({"error": "faltan parámetros: date=YYYY-MM-DD"}), 400
        
    predictions_series, df_actuals, err = generate_prediction_series(target_date=date)
    if err: return jsonify({"error": err}), 400

    data = {
        "model": "LSTM",
        "interval_min": _step_minutes(predictions_series.index),
        "start": predictions_series.index[0].isoformat(),
        "end": predictions_series.index[-1].isoformat(),
        "points": [{"ts": d.isoformat(), "y": float(v)} for d, v in predictions_series.items()],
    }
    
    if df_actuals is not None and not df_actuals.empty:
        data["actuals"] = [
            {"ts": d.isoformat(), "y": (None if pd.isna(v) else float(v))}
            for d, v in df_actuals['valor_real'].items()
        ]
    return jsonify(data)

@app.get("/api/kpis")
def api_kpis_for_date():
    date = request.args.get("date")
    predictions_series, df_actuals, err = generate_prediction_series(target_date=date)
    
    if err: return jsonify({"error": err}), 400
    if df_actuals is None or 'valor_real' not in df_actuals.columns:
        return jsonify({"window_points": 0, "mape": None, "rmse": None, "error": "No hay datos reales para comparar."})

    common_index = predictions_series.index.intersection(df_actuals.index)
    if common_index.empty:
        return jsonify({"window_points": 0, "mape": None, "rmse": None, "error": "No hay solapamiento entre predicción y reales."})

    y_pred = predictions_series[common_index]
    y_true = df_actuals.loc[common_index, 'valor_real']
    
    valid_idx = y_true.notna() & y_pred.notna()
    y_true = y_true[valid_idx].values
    y_pred = y_pred[valid_idx].values

    if len(y_true) == 0:
         return jsonify({"window_points": 0, "mape": None, "rmse": None})

    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100.0)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return jsonify({"window_points": len(y_true), "mape": mape, "rmse": rmse})

@app.get("/api/anomalies")
def api_anomalies_for_date():
    date = request.args.get("date")
    thresh = float(request.args.get("z_threshold", 2.5))
    
    # Si no hay fecha, obtener última fecha disponible
    if not date:
        try:
            if tabla_semanal is not None:
                with engine.connect() as conn:
                    query = select(tabla_semanal.c.fecha).order_by(tabla_semanal.c.fecha.desc()).limit(1)
                    result = conn.execute(query).fetchone()
                    if result:
                        latest_date = result[0]
                        date = latest_date.strftime("%Y-%m-%d")
                        print(f"INFO: Usando última fecha disponible para anomalías: {date}")
        except Exception as e:
            print(f"ERROR obteniendo última fecha: {e}")
            return jsonify({"error": "No se pudo determinar la fecha de referencia"}), 400
    
    predictions_series, df_actuals, err = generate_prediction_series(target_date=date)
    if err: 
        return jsonify({"error": err}), 400
    
    results = _get_anomaly_results(predictions_series, df_actuals, thresh)
    
    # Enriquecer con valores reales y predichos
    if "anomalies" in results and df_actuals is not None:
        for anom in results["anomalies"]:
            ts = pd.Timestamp(anom["ts"])
            if ts in df_actuals.index and "valor_real" in df_actuals.columns:
                anom["real_mw"] = float(df_actuals.loc[ts, "valor_real"])
            if ts in predictions_series.index:
                anom["pred_mw"] = float(predictions_series.loc[ts])
    
    if "error" in results:
        return jsonify(results), 400
    
    # Formato esperado por el chatbot
    return jsonify({
        "mu": results.get("mu"),
        "sd": results.get("sd"),
        "threshold": results.get("threshold"),
        "items": results.get("anomalies", [])
    })

# --- NUEVO: anomalías en rango (todo histórico si quieres) ---
@app.get("/api/anomalies_range")
def api_anomalies_range():
    """
    Detecta anomalías en un rango de fechas [start, end] (ambos YYYY-MM-DD).
    Si no se especifica end, usa start. Devuelve lista agregada de anomalías.
    Advertencia: puede tardar si el rango es muy grande.
    """
    start = request.args.get("start")
    end = request.args.get("end")
    thresh = float(request.args.get("z_threshold", 2.5))

    if not start:
        return jsonify({"error": "Parámetro 'start' es requerido (YYYY-MM-DD)."}), 400

    try:
        start_dt = datetime.strptime(start, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end, "%Y-%m-%d").date() if end else start_dt
        if end_dt < start_dt:
            return jsonify({"error": "end < start"}), 400
    except Exception:
        return jsonify({"error": "Formato de fecha inválido. Usa YYYY-MM-DD"}), 400

    all_items = []
    total_days = (end_dt - start_dt).days + 1

    for i in range(total_days):
        day = (start_dt + timedelta(days=i)).strftime("%Y-%m-%d")
        predictions_series, df_actuals, err = generate_prediction_series(target_date=day)
        if err:
            # Continúa, pero anota el error del día
            all_items.append({"date": day, "error": err})
            continue

        results = _get_anomaly_results(predictions_series, df_actuals, thresh)

        # Enriquecer y añadir
        for anom in results.get("anomalies", []):
            ts = pd.Timestamp(anom["ts"])
            if ts in df_actuals.index and "valor_real" in df_actuals.columns:
                anom["real_mw"] = float(df_actuals.loc[ts, "valor_real"])
            if ts in predictions_series.index:
                anom["pred_mw"] = float(predictions_series.loc[ts])
            anom["date"] = day
            all_items.append(anom)

    out = {
        "start": start_dt.isoformat(),
        "end": end_dt.isoformat(),
        "z_threshold": thresh,
        "total_items": len(all_items),
        "items": all_items[:500]  # protección por tamaño
    }
    if len(all_items) > 500:
        out["truncated"] = True

    return jsonify(out)

@app.get("/api/series")
def api_series():
    """
    Endpoint para consultar series históricas de demanda.
    Parámetros:
    - metric: tipo de métrica (demanda, prediccion, programado)
    - start: fecha/hora de inicio (ISO8601 o YYYY-MM-DD) - OBLIGATORIO
    - end: fecha/hora fin (opcional)
    - agg: agregación temporal (5min, 15min, hour, day)
    """
    metric = request.args.get("metric", "demanda")
    start = request.args.get("start")
    end = request.args.get("end")
    agg = request.args.get("agg", "5min")
    
    if not start:
        return jsonify({"error": "Parámetro 'start' requerido"}), 400
    
    try:
        # Parsear fechas
        start_dt = pd.to_datetime(start)
        
        # Si no hay end, usar el mismo día
        if end:
            end_dt = pd.to_datetime(end)
        else:
            # Si solo especificó fecha, buscar todo el día
            if len(start) == 10:  # formato YYYY-MM-DD
                end_dt = start_dt + timedelta(days=1) - timedelta(microseconds=1)
            else:
                end_dt = start_dt  # Punto específico
        
        # Unifica semanal + histórica
        df = fetch_data_union(start_dt, end_dt)
        
        # Si sigue vacío, devolver error amigable
        if df.empty:
            date_range_str = f"{start_dt.date()}" if start_dt.date() == end_dt.date() else f"{start_dt.date()} a {end_dt.date()}"
            return jsonify({
                "error": f"No hay datos disponibles para {date_range_str}",
                "info": "Verifica que la fecha solicitada exista en la base de datos.",
                "points": []
            }), 404
        
        # Determinar columna según métrica
        if metric == "demanda":
            column = "valor_real"
        elif metric == "prediccion":
            column = "valor_previsto"
        elif metric == "programado":
            column = "valor_programado"
        else:
            return jsonify({"error": f"Métrica desconocida: {metric}"}), 400
        
        if column not in df.columns:
            return jsonify({"error": f"Columna {column} no encontrada"}), 404
        
        # Aplicar agregación si se solicita
        if agg and agg != "5min":
            agg_map = {"15min": "15min", "hour": "1H", "day": "1D"}
            if agg in agg_map:
                df = df.resample(agg_map[agg]).mean()
        
        # Detectar si es consulta de punto único (hora específica)
        is_time_specific = len(start) > 10 and "T" in start
        single_point = is_time_specific and (not end or start == end)
        
        # Preparar respuesta
        points = [
            {"ts": idx.isoformat(), "value": float(val) if not pd.isna(val) else None}
            for idx, val in df[column].items()
        ]
        
        # Si es punto único y hay múltiples registros, buscar el más cercano
        if single_point and len(points) > 1:
            target_time = pd.to_datetime(start)
            closest_idx = min(range(len(df)), key=lambda i: abs(df.index[i] - target_time))
            points = [points[closest_idx]]
        
        result = {
            "metric": metric,
            "start": start,
            "end": end or start,
            "agg": agg,
            "points": points,
            "single_point": single_point
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"ERROR en /api/series: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error interno: {e}"}), 500

# --- Bloque final ---
if __name__ == "__main__":
    app.run(debug=True, port=5001)
