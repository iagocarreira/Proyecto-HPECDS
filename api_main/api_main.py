#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Principal (Flask) v5
- Compatible con modelo v2 (x_scaler, y_scaler).
- Cache de "warm-up" para la predicción "latest".
- Integración de anomalías en la vista de predicción.
- ¡NUEVO! Corregidas advertencias de Pandas (FutureWarning) y
  Sklearn (UserWarning) mediante renombrado temporal.
"""

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

# ---------- Base dir del fichero (no del CWD) ----------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent  # Raíz del proyecto (un nivel arriba)

# Cargar .env desde la raíz del proyecto
load_dotenv(PROJECT_ROOT / ".env")

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

def _resolve_path(p: str) -> Path:
    """Si p es relativo, lo resuelve respecto a la RAÍZ del proyecto."""
    pp = Path(p)
    return (pp if pp.is_absolute() else (PROJECT_ROOT / pp)).resolve()

# ----------------- Config -----------------
MODEL_PATH  = os.getenv("MODEL_PATH",  "artifacts/current/modelo_lstm_multivariate.keras")
SCALER_PATH = os.getenv("SCALER_PATH", "artifacts/current/scaler_lstm_multivariate.joblib")

# Resuelve a rutas absolutas robustas:
MODEL_PATH  = _resolve_path(MODEL_PATH)
SCALER_PATH = _resolve_path(SCALER_PATH)

print("[CWD]           ", Path.cwd())
print("[BASE_DIR]      ", BASE_DIR)
print("[PROJECT_ROOT]  ", PROJECT_ROOT)
print("[MODEL_PATH abs]", MODEL_PATH)
print("[SCALER_PATH abs]", SCALER_PATH)

# ... resto del código sin cambios ...

app = Flask(__name__)

MODELS = {}
X_SCALER, Y_SCALER = None, None
LOOK_BACK = int(os.getenv("LOOK_BACK", "24"))
NUM_FEATURES = 3 # real, previsto, programado
PREDICTION_STEPS = 288 # 24h * 12 (intervalos de 5 min)
PREDICTION_DATA_WINDOW_DAYS, API_DATA_URL = 3, "http://127.0.0.1:5002"

LATEST_PRED_CACHE = {
    "data_json": None,
    "img_base64": None,
    "anomaly_results": None,
    "error": "Cache no inicializado."
}

# --- Conexión a Azure SQL ---
SERVER, DATABASE, USER, PASSWORD = "udcserver2025.database.windows.net", "grupo_1", "ugrupo1", "HK9WXIJaBp2Q97haePdY"
ENGINE_URL = (f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER}:1433/{DATABASE}?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=no&TrustServerCertificate=yes")
engine, tabla_semanal, tabla_historica = None, None, None
try:
    engine = create_engine(ENGINE_URL, pool_pre_ping=True)
    meta = MetaData()
    tabla_semanal = Table("demanda_peninsula_semana", meta, autoload_with=engine, schema="dbo")
    tabla_historica = Table("demanda_peninsula", meta, autoload_with=engine, schema="dbo")
    print("INFO: Conexión a la BD establecida.")
except Exception as e:
    print(f"ERROR CRÍTICO: No se pudo conectar a la base de datos. {e}")


# --- Utilidades de inferencia ---

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

    # --- ¡NUEVO! Mapa de renombrado para el scaler ---
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
        
        # --- ¡CORREGIDO! Renombrar temporalmente antes de escalar ---
        win_renamed = win.rename(columns=RENAME_MAP_SCALER)
        # Usar .feature_names_in_ para asegurar el orden correcto que espera el scaler
        X = x_scaler.transform(win_renamed[x_scaler.feature_names_in_])
        # --- Fin de la corrección de Sklearn ---
        
        X = np.expand_dims(X, axis=0)

        y_scaled = float(model.predict(X, verbose=0).flatten()[0])
        y_scaled = float(np.clip(y_scaled, 0.0, 1.0))
        y_next = float(y_scaler.inverse_transform([[y_scaled]])[0,0])

        t_next = cur_hist.index[-1] + pd.Timedelta(minutes=step)
        preds.append((t_next, y_next))

        # Los keys aquí ('valor_real', etc.) deben coincidir con 'feats'
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
        
        # --- ¡CORREGIDO! Usar pd.concat para evitar FutureWarning ---
        new_df_row = pd.DataFrame(row_data, index=[t_next])
        cur_hist = pd.concat([cur_hist, new_df_row])
        # --- Fin de la corrección de Pandas ---

    return pd.Series([v for _, v in preds], index=[t for t, _ in preds], name="forecast")

# --- Carga de Modelos v2 ---
def load_models_at_startup():
    global MODELS, X_SCALER, Y_SCALER
    try:
        X_SCALER = joblib.load("artifacts/current/x_scaler.joblib")
        Y_SCALER = joblib.load("artifacts/current/y_scaler.joblib")
        print("INFO: Scalers X/Y cargados correctamente.")
    except Exception as e:
        print(f"ERROR CRÍTICO al cargar scalers (x_scaler/y_scaler): {e}")
    
    try:
        model_path = "artifacts/current/model.keras"
        if not os.path.exists(model_path):
            model_path = "artifacts/current/modelo_lstm_multivariate.keras"
        
        MODELS['LSTM'] = load_model(model_path)
        print(f"INFO: Modelo LSTM cargado desde {model_path}.")
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


# --- Lógica de predicción ---
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
            seven_days_ago = datetime.now() - timedelta(days=7)
            table_to_use = tabla_semanal if prediction_day_start.date() >= seven_days_ago.date() else tabla_historica
            if table_to_use is None: return None, None, "Error: Conexión a BD no disponible."

            end_history = prediction_day_start - timedelta(microseconds=1)
            start_history = end_history - timedelta(days=PREDICTION_DATA_WINDOW_DAYS)
            df_history = fetch_data_from_db(table_to_use, start_history, end_history)
            
            end_prediction_day = prediction_day_start + timedelta(days=1) - timedelta(microseconds=1)
            df_future_covariates = fetch_data_from_db(table_to_use, prediction_day_start, end_prediction_day)
            
            if len(df_future_covariates) < PREDICTION_STEPS:
                print(f"WARN: Faltan datos de 'previsto'/'programado' para el día {target_date}. Se obtuvieron {len(df_future_covariates)}.")
            
            df_actuals_for_plot = df_future_covariates.copy()
        
        else:
            # MODO 2: Predecir "a partir de ahora"
            table_to_use = tabla_semanal
            end_dt, start_dt = datetime.now(), datetime.now() - timedelta(days=PREDICTION_DATA_WINDOW_DAYS)
            df_history = fetch_data_from_db(table_to_use, start_dt, end_dt)
            
            if df_history.empty:
                return None, None, "No se encontraron datos históricos recientes en la tabla semanal."
            if len(df_history) < LOOK_BACK:
                return None, None, f"Datos históricos insuficientes ({len(df_history)}). Se necesitan {LOOK_BACK}."
            
            freq_obj = _infer_freq(df_history.index)
            freq_str = freq_obj.freqstr
            prediction_start_time = df_history.index[-1] + freq_obj
            df_actuals_for_plot = df_history.copy()
            
            start_proxy = prediction_start_time - timedelta(days=365)
            end_proxy = start_proxy + timedelta(minutes=PREDICTION_STEPS * _step_minutes(df_history.index))
            df_proxy_futures = fetch_data_from_db(tabla_historica, start_proxy, end_proxy)
            
            if len(df_proxy_futures) < PREDICTION_STEPS:
                return None, None, "No se encontraron datos de 'previsto'/'programado' para hoy, y tampoco hay datos de respaldo de hace un año."
            
            proxy_index = pd.date_range(start=prediction_start_time, periods=len(df_proxy_futures), freq=freq_obj)
            df_proxy_futures.index = proxy_index
            df_future_covariates = df_proxy_futures.copy()

        if df_history.empty:
            return None, None, "df_history está vacío, no se puede predecir."
        
        df_history[features] = df_history[features].fillna(method='ffill').fillna(0)
        df_future_covariates[features] = df_future_covariates[features].fillna(method='ffill').fillna(0)
        
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


# --- Función de ploteo ---
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

# --- Helper para Anomalías ---
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


# --- Función de "Warm-Up" ---
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

# --- ENDPOINTS (API) ---

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
    
    predictions_series, df_actuals, err = generate_prediction_series(target_date=date)
    if err: return jsonify({"error": err}), 400
    
    results = _get_anomaly_results(predictions_series, df_actuals, thresh)
    if "error" in results:
        return jsonify(results), 400
    return jsonify(results)


# --- Bloque final ---
if __name__ == "__main__":
    app.run(debug=True, port=5001)