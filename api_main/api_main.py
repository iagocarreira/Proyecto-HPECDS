#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API Principal Unificada - GreenEnergy Insights
Combina funcionalidades de:
- Predicción con modelos ML (LSTM)
- Consulta de demanda histórica
- Detección de anomalías
- Dashboard y visualizaciones
"""

import os
import io
import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
from sqlalchemy import create_engine, MetaData, Table, select, and_
from datetime import datetime, timedelta
import base64
import math
import random

# Configurar el backend de Matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Importaciones de ML
from tensorflow.keras.models import load_model
import joblib

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- CONFIGURACIÓN GLOBAL ---
app = Flask(__name__)

# Configuración de Azure SQL
SERVER = "udcserver2025.database.windows.net"
DATABASE = "grupo_1"
USER = "ugrupo1"
PASSWORD = "HK9WXIJaBp2Q97haePdY"

ENGINE_URL = (
    f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER}:1433/{DATABASE}"
    "?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=no&TrustServerCertificate=yes"
)

# Variables globales de BD
engine = None
tabla_semanal = None
tabla_historica = None

# Variables globales de ML
MODELS = {}
X_SCALER, Y_SCALER = None, None
LOOK_BACK = int(os.getenv("LOOK_BACK", "24"))
NUM_FEATURES = 3  # real, previsto, programado
PREDICTION_STEPS = 288  # 24h * 12 (intervalos de 5 min)
PREDICTION_DATA_WINDOW_DAYS = 3
API_DATA_URL = "http://127.0.0.1:5002"

# Cache de predicciones
LATEST_PRED_CACHE = {
    "data_json": None,
    "img_base64": None,
    "anomaly_results": None,
    "error": "Cache no inicializado."
}

# Parámetros de detección de anomalías
EWMA_ALPHA = float(os.getenv("ANOM_EWMA_ALPHA", "0.10"))
WIN_MIN = int(os.getenv("ANOM_WIN_MIN", "1440"))
MIN_FRAC = float(os.getenv("ANOM_MIN_FRAC", "0.8"))
PCT_THRESHOLD = float(os.getenv("ANOM_PCT", "0.015"))
FLOOR_MW = float(os.getenv("ANOM_FLOOR", "300.0"))
ANOM_CHECK_EVERY_MIN = int(os.getenv("ANOM_CHECK_EVERY_MIN", "15"))


# --- INICIALIZACIÓN DE BASE DE DATOS ---
def init_database():
    """Inicializa la conexión a la base de datos."""
    global engine, tabla_semanal, tabla_historica
    
    try:
        engine = create_engine(ENGINE_URL, pool_pre_ping=True)
        meta = MetaData()
        
        tabla_semanal = Table(
            "demanda_peninsula_semana",
            meta,
            autoload_with=engine,
            schema="dbo"
        )
        
        tabla_historica = Table(
            "demanda_peninsula",
            meta,
            autoload_with=engine,
            schema="dbo"
        )
        
        print("✓ Conexión a la BD establecida correctamente")
        return True
        
    except Exception as e:
        print(f"✗ ERROR al conectar a la base de datos: {e}")
        engine = None
        tabla_semanal = None
        tabla_historica = None
        return False


# --- UTILIDADES DE INFERENCIA ---
def _infer_freq(index: pd.DatetimeIndex):
    """Infiere la frecuencia de un índice temporal."""
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
    """Calcula los minutos entre pasos consecutivos."""
    try:
        freq = pd.infer_freq(index)
    except Exception:
        freq = None
    delta = pd.to_timedelta(freq) if freq else pd.Series(index).diff().median()
    if pd.isna(delta):
        delta = pd.Timedelta(minutes=5)
    return max(1, int(delta.total_seconds() // 60))


def _parse_datetime(date_str: str) -> datetime:
    """Parse fecha en formato ISO con soporte para fecha sola o datetime."""
    try:
        if "T" in date_str:
            return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception as e:
        raise ValueError(f"Formato de fecha inválido: {date_str}. Usa YYYY-MM-DD o YYYY-MM-DDTHH:MM:SS")


def _select_table(target_date: datetime) -> Table:
    """Selecciona la tabla apropiada según la fecha."""
    seven_days_ago = datetime.now() - timedelta(days=7)
    
    if target_date.date() >= seven_days_ago.date():
        return tabla_semanal
    else:
        return tabla_historica


def _apply_aggregation(df: pd.DataFrame, agg: str) -> pd.DataFrame:
    """Aplica agregación temporal a los datos."""
    agg_map = {
        "15min": "15min",
        "hour": "1H",
        "day": "1D"
    }
    
    freq = agg_map.get(agg, "5min")
    
    if freq == "5min":
        return df
    
    df_agg = df.resample(freq).mean()
    return df_agg


# --- CARGA DE MODELOS ML ---
def load_models_at_startup():
    """Carga los modelos y scalers al iniciar la aplicación."""
    global MODELS, X_SCALER, Y_SCALER
    
    try:
        X_SCALER = joblib.load("artifacts/current/x_scaler.joblib")
        Y_SCALER = joblib.load("artifacts/current/y_scaler.joblib")
        print("✓ Scalers X/Y cargados correctamente")
    except Exception as e:
        print(f"✗ ERROR al cargar scalers: {e}")
    
    try:
        model_path = "artifacts/current/model.keras"
        if not os.path.exists(model_path):
            model_path = "artifacts/current/modelo_lstm_multivariate.keras"
        
        MODELS['LSTM'] = load_model(model_path)
        print(f"✓ Modelo LSTM cargado desde {model_path}")
    except Exception as e:
        print(f"✗ ERROR al cargar modelo LSTM: {e}")


# --- CONSULTA DE DATOS ---
def fetch_data_from_db(table_to_query: Table, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Consulta datos de la base de datos en un rango de fechas."""
    if not engine or table_to_query is None:
        return pd.DataFrame()
    
    query = select(table_to_query).where(
        and_(
            table_to_query.c.fecha >= start_date,
            table_to_query.c.fecha <= end_date
        )
    ).order_by(table_to_query.c.fecha)
    
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            if not df.empty and 'fecha' in df.columns:
                df['fecha'] = pd.to_datetime(df['fecha'])
                df = df.set_index('fecha').sort_index()
            print(f"✓ Obtenidos {len(df)} registros de '{table_to_query.name}'")
            return df
    except Exception as e:
        print(f"✗ ERROR al consultar '{table_to_query.name}': {e}")
        return pd.DataFrame()


# --- PREDICCIÓN MULTI-STEP ---
def forecast_multi_step(model, x_scaler, y_scaler, df_hist, look_back, horizon_steps,
                        future_cov: pd.DataFrame | None = None,
                        fallback: str = "persist",
                        feats=("valor_real", "valor_previsto", "valor_programado")):
    """Genera predicciones multi-step con el modelo LSTM."""
    hist = df_hist.copy()
    cols = list(feats)
    preds = []
    cur_hist = hist.copy()
    step = _step_minutes(hist.index)
    if step == 0:
        step = 5

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
        
        win_renamed = win.rename(columns=RENAME_MAP_SCALER)
        X = x_scaler.transform(win_renamed[x_scaler.feature_names_in_])
        X = np.expand_dims(X, axis=0)

        y_scaled = float(model.predict(X, verbose=0).flatten()[0])
        y_scaled = float(np.clip(y_scaled, 0.0, 1.0))
        y_next = float(y_scaler.inverse_transform([[y_scaled]])[0, 0])

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
        
        new_df_row = pd.DataFrame(row_data, index=[t_next])
        cur_hist = pd.concat([cur_hist, new_df_row])

    return pd.Series([v for _, v in preds], index=[t for t, _ in preds], name="forecast")



# --- GENERACIÓN DE GRÁFICOS ---
def generate_prediction_plot_image(predictions_series, df_actuals, target_date: str = None):
    """Genera una imagen de la predicción."""
    try:
        plt.figure(figsize=(14, 7))
        plt.style.use('seaborn-v0_8-whitegrid')
        
        real_demand = pd.Series([], dtype=float)
        dates_real = pd.Series([], dtype='datetime64[ns]')
        
        if df_actuals is not None and not df_actuals.empty and 'valor_real' in df_actuals.columns:
            real_demand = df_actuals['valor_real']
            dates_real = pd.to_datetime(df_actuals.index)
            plt.plot(dates_real, real_demand, label='Demanda Real (MW)', 
                    color='steelblue', marker='.', markersize=4)

        dates_pred = pd.to_datetime(predictions_series.index)
        predictions_real_mw = predictions_series.values
        plt.plot(dates_pred, predictions_real_mw, label='Predicción LSTM', 
                color='darkorange', linestyle='--', linewidth=2)

        if target_date:
            title_date = f"para el día {target_date}"
        else:
            title_date = f"a partir de {predictions_series.index[0].strftime('%Y-%m-%d %H:%M')}"
            
        plt.title(f'Predicción de Demanda con LSTM {title_date}', fontsize=16)
        plt.xlabel("Tiempo")
        plt.ylabel("Potencia (MW)")
        plt.legend()
        
        all_values = pd.concat([pd.Series(predictions_real_mw), real_demand])
        if not all_values.dropna().empty:
            plt.ylim(all_values.min() * 0.9, all_values.max() * 1.1)
            
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        img_data = io.BytesIO()
        plt.savefig(img_data, format='png')
        plt.close()
        img_data.seek(0)
        return img_data, None
    except Exception as e:
        logger.error(f"✗ Error generando plot: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, f"Error al generar la imagen: {e}"


# --- DETECCIÓN DE ANOMALÍAS ---
def _calculate_hybrid_thresholds(df_an: pd.DataFrame, K: float):
    """Calcula umbrales híbridos para detección de anomalías."""
    res_s = df_an["residuo"].ewm(alpha=EWMA_ALPHA, adjust=False).mean().shift(1)
    
    mad_scalar = (res_s - res_s.median()).abs().median()
    sigma_rob_scalar = 1.4826 * mad_scalar

    if not pd.notna(sigma_rob_scalar) or sigma_rob_scalar < 0.1:
        sigma_rob_scalar = df_an["residuo"].std()
        if not pd.notna(sigma_rob_scalar) or sigma_rob_scalar < 0.1:
            sigma_rob_scalar = 1.0 
    
    sigma_rob = pd.Series(sigma_rob_scalar, index=df_an.index)

    u_sigma = K * sigma_rob
    u_pct = PCT_THRESHOLD * df_an["real"].abs()
    u_abs = pd.Series(FLOOR_MW, index=df_an.index)
    
    umbral = pd.concat([u_sigma, u_pct, u_abs], axis=1).max(axis=1)
    step_min = _step_minutes(df_an.index)
    
    return umbral, sigma_rob, step_min


def _get_anomaly_results(predictions_series, df_actuals, thresh=2.5):
    """Detecta anomalías usando lógica robusta."""
    if df_actuals is None or 'valor_real' not in df_actuals.columns:
        return {"anomalies": [], "error": "No hay datos reales para analizar."}

    common_index = predictions_series.index.intersection(df_actuals.index)
    if common_index.empty:
        return {"anomalies": [], "error": "No hay solapamiento entre predicción y reales."}

    y_pred = predictions_series[common_index]
    y_true = df_actuals.loc[common_index, 'valor_real']
    
    valid_idx = y_true.notna() & y_pred.notna()
    if not valid_idx.any():
        return {"anomalies": [], "error": "No hay datos válidos solapados."}

    y_true = y_true[valid_idx]
    y_pred = y_pred[valid_idx]
    
    df_an = pd.DataFrame({"real": y_true, "referencia": y_pred})
    df_an["residuo"] = df_an["real"] - df_an["referencia"]

    try:
        umbral, sigma_rob, step_min = _calculate_hybrid_thresholds(df_an, K=thresh)
    except Exception as e:
        logger.error(f"✗ ERROR en _calculate_hybrid_thresholds: {e}")
        return {"anomalies": [], "error": f"Error al calcular umbrales: {e}"}

    stride = max(1, ANOM_CHECK_EVERY_MIN // step_min) if ANOM_CHECK_EVERY_MIN > 0 else 1
    check_idx = df_an.index[::stride]
    
    anomalies = []
    
    for ts in check_idx:
        if ts not in df_an.index:
            continue
            
        residuo = df_an.loc[ts, "residuo"]
        umbral_mw = umbral.loc[ts]
        
        if pd.isna(residuo) or pd.isna(umbral_mw):
            continue

        if abs(residuo) > umbral_mw:
            sigma = sigma_rob.loc[ts]
            z_robust = (residuo / sigma) if pd.notna(sigma) and sigma > 0.1 else 0.0
            
            anomalies.append({
                "ts": ts.isoformat(),
                "residual": float(residuo),
                "z": float(z_robust),
                "severity": min(abs(z_robust) / 5.0, 1.0)
            })
    
    mu_report = float(df_an["residuo"].median())
    sd_report = float(sigma_rob.median())

    return {"mu": mu_report, "sd": sd_report, "threshold": thresh, "anomalies": anomalies}


# --- GENERACIÓN DE GRÁFICOS ---
def generate_prediction_series(target_date: str | None = None):
    """Genera series de predicción para una fecha específica o datos recientes."""
    logger.info(f"🔄 Iniciando generate_prediction_series con target_date={target_date}")
    
    if 'LSTM' not in MODELS:
        error_msg = "Error: Modelo LSTM no cargado."
        logger.error(error_msg)
        return None, None, error_msg
    
    if X_SCALER is None or Y_SCALER is None:
        error_msg = "Error crítico: Scalers X/Y no cargados."
        logger.error(error_msg)
        return None, None, error_msg

    features = ['valor_real', 'valor_previsto', 'valor_programado']
    
    df_history = pd.DataFrame()
    df_future_covariates = pd.DataFrame()
    df_actuals_for_plot = pd.DataFrame()
    prediction_start_time = None
    freq_str = '5min'

    try:
        if target_date:
            # MODO 1: Predecir un día específico (pasado)
            logger.info(f"📅 Modo predicción para fecha específica: {target_date}")
            prediction_day_start = datetime.strptime(target_date, "%Y-%m-%d")
            prediction_start_time = prediction_day_start
            seven_days_ago = datetime.now() - timedelta(days=7)
            table_to_use = tabla_semanal if prediction_day_start.date() >= seven_days_ago.date() else tabla_historica
            
            if table_to_use is None:
                return None, None, "Error: Conexión a BD no disponible."

            end_history = prediction_day_start - timedelta(microseconds=1)
            start_history = end_history - timedelta(days=PREDICTION_DATA_WINDOW_DAYS)
            df_history = fetch_data_from_db(table_to_use, start_history, end_history)
            
            if df_history.empty:
                error_msg = f"No se encontraron datos históricos para el periodo {start_history} - {end_history}"
                logger.error(error_msg)
                return None, None, error_msg
            
            logger.info(f"✓ Datos históricos: {len(df_history)} registros")
            
            end_prediction_day = prediction_day_start + timedelta(days=1) - timedelta(microseconds=1)
            df_future_covariates = fetch_data_from_db(table_to_use, prediction_day_start, end_prediction_day)
            
            if len(df_future_covariates) < PREDICTION_STEPS:
                logger.warning(f"⚠ Faltan datos de 'previsto'/'programado' para el día {target_date}. Se obtuvieron {len(df_future_covariates)}.")
            
            df_actuals_for_plot = df_future_covariates.copy()
        
        else:
            # MODO 2: Predecir "a partir de ahora"
            logger.info("🕐 Modo predicción en tiempo real (últimos datos)")
            table_to_use = tabla_semanal
            
            if table_to_use is None:
                return None, None, "Error: Tabla semanal no disponible."
            
            end_dt = datetime.now()
            start_dt = datetime.now() - timedelta(days=PREDICTION_DATA_WINDOW_DAYS)
            
            logger.info(f"📊 Consultando datos desde {start_dt} hasta {end_dt}")
            df_history = fetch_data_from_db(table_to_use, start_dt, end_dt)
            
            if df_history.empty:
                error_msg = "No se encontraron datos históricos recientes en la tabla semanal."
                logger.error(error_msg)
                return None, None, error_msg
            
            logger.info(f"✓ Datos históricos obtenidos: {len(df_history)} registros")
            
            if len(df_history) < LOOK_BACK:
                error_msg = f"Datos históricos insuficientes ({len(df_history)}). Se necesitan al menos {LOOK_BACK}."
                logger.error(error_msg)
                return None, None, error_msg
            
            freq_obj = _infer_freq(df_history.index)
            freq_str = freq_obj.freqstr
            prediction_start_time = df_history.index[-1] + freq_obj
            df_actuals_for_plot = df_history.copy()
            
            # Buscar datos proxy para covariables futuras
            start_proxy = prediction_start_time - timedelta(days=365)
            end_proxy = start_proxy + timedelta(minutes=PREDICTION_STEPS * _step_minutes(df_history.index))
            
            logger.info(f"🔍 Buscando datos proxy desde {start_proxy} hasta {end_proxy}")
            df_proxy_futures = fetch_data_from_db(tabla_historica, start_proxy, end_proxy)
            
            if len(df_proxy_futures) < PREDICTION_STEPS:
                logger.warning(f"⚠ Datos proxy insuficientes: {len(df_proxy_futures)} de {PREDICTION_STEPS}")
                # Usar última observación como proxy
                proxy_data = pd.DataFrame({
                    'valor_previsto': [df_history['valor_previsto'].iloc[-1]] * PREDICTION_STEPS,
                    'valor_programado': [df_history['valor_programado'].iloc[-1]] * PREDICTION_STEPS
                })
                proxy_index = pd.date_range(start=prediction_start_time, periods=PREDICTION_STEPS, freq=freq_obj)
                df_proxy_futures = pd.DataFrame(proxy_data, index=proxy_index)
            else:
                proxy_index = pd.date_range(start=prediction_start_time, periods=len(df_proxy_futures), freq=freq_obj)
                df_proxy_futures.index = proxy_index
            
            df_future_covariates = df_proxy_futures.copy()

        if df_history.empty:
            return None, None, "df_history está vacío, no se puede predecir."
        
        # Validar que existan las columnas necesarias
        missing_cols = [col for col in features if col not in df_history.columns]
        if missing_cols:
            error_msg = f"Faltan columnas en df_history: {missing_cols}"
            logger.error(error_msg)
            return None, None, error_msg
        
        # Rellenar valores faltantes
        logger.info("🔧 Rellenando valores faltantes...")
        df_history[features] = df_history[features].ffill().fillna(0)
        df_future_covariates[features] = df_future_covariates[features].ffill().fillna(0)
        
        # Validar que no hay NaN después del relleno
        if df_history[features].isna().any().any():
            logger.warning("⚠ Aún hay valores NaN después de ffill, usando fillna(0)")
            df_history[features] = df_history[features].fillna(0)
        
        logger.info(f"🚀 Iniciando predicción multi-step ({PREDICTION_STEPS} pasos)...")
        predictions_series = forecast_multi_step(
            MODELS['LSTM'], X_SCALER, Y_SCALER,
            df_hist=df_history,
            look_back=LOOK_BACK,
            horizon_steps=PREDICTION_STEPS,
            future_cov=df_future_covariates,
            fallback="persist",
            feats=['valor_real', 'valor_previsto', 'valor_programado']
        )
        
        logger.info(f"✓ Predicción completada: {len(predictions_series)} puntos generados")
        return predictions_series, df_actuals_for_plot, None

    except Exception as e:
        error_msg = f"Error interno durante la predicción: {e}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return None, None, error_msg



def _get_anomaly_results(predictions_series, df_actuals, thresh=2.5):
    """Detecta anomalías usando lógica robusta."""
    if df_actuals is None or 'valor_real' not in df_actuals.columns:
        return {"anomalies": [], "error": "No hay datos reales para analizar."}

    common_index = predictions_series.index.intersection(df_actuals.index)
    if common_index.empty:
        return {"anomalies": [], "error": "No hay solapamiento entre predicción y reales."}

    y_pred = predictions_series[common_index]
    y_true = df_actuals.loc[common_index, 'valor_real']
    
    valid_idx = y_true.notna() & y_pred.notna()
    if not valid_idx.any():
        return {"anomalies": [], "error": "No hay datos válidos solapados."}

    y_true = y_true[valid_idx]
    y_pred = y_pred[valid_idx]
    
    df_an = pd.DataFrame({"real": y_true, "referencia": y_pred})
    df_an["residuo"] = df_an["real"] - df_an["referencia"]

    try:
        umbral, sigma_rob, step_min = _calculate_hybrid_thresholds(df_an, K=thresh)
    except Exception as e:
        print(f"✗ ERROR en _calculate_hybrid_thresholds: {e}")
        return {"anomalies": [], "error": f"Error al calcular umbrales: {e}"}

    stride = max(1, ANOM_CHECK_EVERY_MIN // step_min) if ANOM_CHECK_EVERY_MIN > 0 else 1
    check_idx = df_an.index[::stride]
    
    anomalies = []
    
    for ts in check_idx:
        if ts not in df_an.index:
            continue
            
        residuo = df_an.loc[ts, "residuo"]
        umbral_mw = umbral.loc[ts]
        
        if pd.isna(residuo) or pd.isna(umbral_mw):
            continue

        if abs(residuo) > umbral_mw:
            sigma = sigma_rob.loc[ts]
            z_robust = (residuo / sigma) if pd.notna(sigma) and sigma > 0.1 else 0.0
            
            anomalies.append({
                "ts": ts.isoformat(),
                "residual": float(residuo),
                "z": float(z_robust),
                "severity": min(abs(z_robust) / 5.0, 1.0)
            })
    
    mu_report = float(df_an["residuo"].median())
    sd_report = float(sigma_rob.median())

    return {"mu": mu_report, "sd": sd_report, "threshold": thresh, "anomalies": anomalies}


# --- WARM-UP DE CACHE ---
def warm_up_prediction_cache():
    """Pre-carga las predicciones en caché al iniciar."""
    global LATEST_PRED_CACHE
    logger.info("🔥 [WARM-UP] Iniciando predicción de 'últimos datos'...")

    try:
        predictions_series, df_actuals, err = generate_prediction_series(target_date=None)
        
        if err:
            logger.error(f"✗ [WARM-UP] Falló la predicción: {err}")
            LATEST_PRED_CACHE = {
                "data_json": None, 
                "img_base64": None, 
                "anomaly_results": None, 
                "error": err
            }
            return

        if predictions_series is None or predictions_series.empty:
            error_msg = "La serie de predicciones está vacía"
            logger.error(f"✗ [WARM-UP] {error_msg}")
            LATEST_PRED_CACHE = {
                "data_json": None,
                "img_base64": None,
                "anomaly_results": None,
                "error": error_msg
            }
            return

        # Crear JSON de datos
        try:
            data_json = {
                "model": "LSTM",
                "interval_min": _step_minutes(predictions_series.index),
                "start": predictions_series.index[0].isoformat(),
                "end": predictions_series.index[-1].isoformat(),
                "points": [{"ts": d.isoformat(), "y": float(v)} for d, v in predictions_series.items()],
            }
            logger.info(f"✓ JSON creado con {len(data_json['points'])} puntos")
        except Exception as e:
            error_msg = f"Error al crear JSON: {e}"
            logger.error(f"✗ [WARM-UP] {error_msg}")
            LATEST_PRED_CACHE["error"] = error_msg
            return

        # Generar imagen
        logger.info("📊 Generando imagen de predicción...")
        img_data_buffer, img_err = generate_prediction_plot_image(predictions_series, df_actuals, target_date=None)
        if img_err:
            logger.error(f"✗ [WARM-UP] Error en imagen: {img_err}")
            LATEST_PRED_CACHE["error"] = img_err
            return
        
        img_base64 = base64.b64encode(img_data_buffer.read()).decode('utf-8')
        logger.info(f"✓ Imagen generada ({len(img_base64)} bytes)")

        # Detectar anomalías
        logger.info("🔍 Detectando anomalías...")
        try:
            anomaly_results = _get_anomaly_results(predictions_series, df_actuals)
            logger.info(f"✓ Anomalías detectadas: {len(anomaly_results.get('anomalies', []))}")
        except Exception as e:
            logger.warning(f"⚠ Error detectando anomalías: {e}")
            anomaly_results = {"anomalies": [], "error": str(e)}

        # Actualizar caché
        LATEST_PRED_CACHE = {
            "data_json": data_json,
            "img_base64": img_base64,
            "anomaly_results": anomaly_results,
            "error": None
        }
        logger.info("✓ [WARM-UP] Predicción cacheada correctamente")
        
    except Exception as e:
        error_msg = f"Error inesperado en warm_up: {e}"
        logger.error(f"✗ [WARM-UP] {error_msg}")
        import traceback
        logger.error(traceback.format_exc())
        LATEST_PRED_CACHE = {
            "data_json": None,
            "img_base64": None,
            "anomaly_results": None,
            "error": error_msg
        }

# ==================== ENDPOINTS HTML ====================

@app.route("/")
def home():
    """Página principal."""
    return render_template("index.html", api_data_url=API_DATA_URL)


@app.route("/predict_latest/<model_name>", methods=["GET"])
def predict_latest(model_name):
    """Muestra la predicción más reciente."""
    cache = LATEST_PRED_CACHE
    
    # Si hay error en el cache, intentar regenerar
    if cache.get("error"):
        logger.warning(f"⚠ Cache con error: {cache['error']}")
        logger.info("🔄 Intentando regenerar predicción...")
        
        try:
            warm_up_prediction_cache()
            cache = LATEST_PRED_CACHE
            
            if cache.get("error"):
                logger.error("✗ No se pudo regenerar la predicción")
                return render_template("plot_view.html", 
                                     error=cache["error"], 
                                     model='LSTM')
        except Exception as e:
            logger.error(f"✗ Error al intentar regenerar: {e}")
            return render_template("plot_view.html", 
                                 error=f"Error al regenerar predicción: {e}", 
                                 model='LSTM')
    
    # Verificar que tenemos datos válidos
    if not cache.get("img_base64"):
        return render_template("plot_view.html", 
                             error="No hay datos de predicción disponibles", 
                             model='LSTM')
    
    return render_template("plot_view.html",
                         model='LSTM',
                         img_data=cache["img_base64"],
                         anomaly_results=cache.get("anomaly_results"),
                         date="Últimos datos disponibles")

@app.route("/predict_custom", methods=["POST"])
def predict_custom():
    """Genera predicción para una fecha personalizada."""
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
    """Página de dashboard."""
    return render_template("dashboard.html")


@app.route("/documentacion")
def documentation():
    """Página de documentación."""
    return render_template("documentacion.html")


@app.route("/chatbot")
def chatbot_page():
    """Página de chatbot."""
    return render_template("chatbot.html")


# ==================== ENDPOINTS API ====================

@app.route("/health")
def health_check():
    """Endpoint de salud del sistema."""
    return jsonify({
        "status": "ok",
        "database": "connected" if engine else "disconnected",
        "models_loaded": len(MODELS) > 0
    }), 200


@app.route("/api/series", methods=["GET"])
def api_series():
    """
    Endpoint para consultar series históricas de demanda.
    Query params:
      - metric: tipo de métrica (solo 'demanda' soportado)
      - start: fecha de inicio (YYYY-MM-DD o ISO)
      - end: fecha de fin (opcional)
      - agg: agregación temporal (5min, 15min, hour, day)
    """
    try:
        metric = request.args.get("metric")
        start_str = request.args.get("start")
        end_str = request.args.get("end")
        agg = request.args.get("agg", "5min")
        
        if not metric:
            return jsonify({"error": "Parámetro 'metric' es obligatorio"}), 400
        
        if metric != "demanda":
            return jsonify({"error": f"Métrica '{metric}' no soportada. Usa 'demanda'"}), 400
        
        if not start_str:
            return jsonify({"error": "Parámetro 'start' es obligatorio"}), 400
        
        try:
            start_dt = _parse_datetime(start_str)
            end_dt = _parse_datetime(end_str) if end_str else None
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        
        if end_dt and start_dt >= end_dt:
            return jsonify({"error": "La fecha 'start' debe ser anterior a 'end'"}), 400
        
        if end_dt is None:
            is_specific_time = 'T' in start_str
            if is_specific_time:
                end_dt = start_dt + timedelta(minutes=5)
            else:
                end_dt = start_dt + timedelta(days=1) - timedelta(microseconds=1)
        
        table_to_use = _select_table(start_dt)
        table_name = "demanda_peninsula_semana" if table_to_use == tabla_semanal else "demanda_peninsula"
        
        if table_to_use is None:
            return jsonify({"error": "Conexión a BD no disponible"}), 500
        
        query = select(table_to_use).where(
            and_(
                table_to_use.c.fecha >= start_dt,
                table_to_use.c.fecha <= end_dt
            )
        ).order_by(table_to_use.c.fecha)
        
        try:
            with engine.connect() as conn:
                df = pd.read_sql(query, conn)
        except Exception as e:
            print(f"✗ Error al consultar BD: {e}")
            return jsonify({"error": f"Error al consultar la base de datos: {str(e)}"}), 500
        
        if df.empty:
            return jsonify({
                "error": f"No se encontraron datos para el periodo {start_str} - {end_str or start_str}",
                "start": start_str,
                "end": end_str
            }), 404
        
        df['fecha'] = pd.to_datetime(df['fecha'])
        df = df.set_index('fecha').sort_index()
        
        if agg != "5min" and len(df) > 1:
            df = _apply_aggregation(df, agg)
        
        points = []
        for ts, row in df.iterrows():
            value = row.get('valor_real', None)
            if pd.notna(value):
                points.append({
                    "ts": ts.isoformat(),
                    "value": round(float(value), 2)
                })
        
        single_point = len(points) == 1
        
        response = {
            "metric": metric,
            "start": start_str,
            "end": end_str or start_str,
            "aggregation": agg,
            "points": points,
            "single_point": single_point,
            "metadata": {
                "total_points": len(points),
                "source_table": table_name
            }
        }
        
        if len(points) > 20:
            values = [p["value"] for p in points]
            response["summary"] = {
                "total_points": len(points),
                "periodo": f"{points[0]['ts']} a {points[-1]['ts']}",
                "min_mw": round(min(values), 2),
                "max_mw": round(max(values), 2),
                "promedio_mw": round(sum(values) / len(values), 2)
            }
            response["points"] = points[:3] + points[-3:]
        
        print(f"✓ Consulta exitosa: {len(points)} puntos de {table_name}")
        return jsonify(response), 200
        
    except Exception as e:
        print(f"✗ Error en /api/series: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Error interno: {str(e)}"}), 500


@app.get("/api/predictions_latest")
def api_predictions_latest():
    """Devuelve la predicción más reciente en formato JSON."""
    cache = LATEST_PRED_CACHE
    if cache["error"]:
        return jsonify({"error": cache["error"]}), 400
    return jsonify(cache["data_json"])


@app.get("/api/predictions")
def api_predictions_for_date():
    """
    Devuelve predicciones para una fecha específica.
    Query params:
      - date: fecha en formato YYYY-MM-DD
      - model: nombre del modelo (opcional, solo 'LSTM' soportado)
      - start: fecha de inicio alternativa (opcional)
      - end: fecha de fin (opcional)
      - limit: límite de puntos (opcional)
    """
    date = request.args.get("date")
    model = request.args.get("model", "LSTM")
    start_str = request.args.get("start")
    end_str = request.args.get("end")
    limit = int(request.args.get("limit", "200"))
    
    # Si no hay fecha pero hay start/end, generar datos mock para desarrollo
    if not date and (start_str or end_str):
        if engine:
            try:
                meta = MetaData()
                try:
                    table_preds = Table("predictions", meta, autoload_with=engine, schema="dbo")
                except Exception:
                    table_preds = None
                
                if table_preds is not None:
                    q = select(table_preds).limit(limit)
                    with engine.connect() as conn:
                        df = pd.read_sql(q, conn)
                    preds = []
                    for _, row in df.iterrows():
                        preds.append({
                            "ts": pd.to_datetime(row["ts"]).isoformat(),
                            "pred": float(row["pred"]),
                            "model": row.get("model", model)
                        })
                    return jsonify({"predictions": preds, "metadata": {"total": len(preds)}}), 200
            except Exception as e:
                print(f"✗ Error consultando tabla predictions: {e}")
        
        # Modo mock
        preds = []
        if start_str and end_str:
            try:
                s = _parse_datetime(start_str)
                e = _parse_datetime(end_str)
                cur = s
                count = 0
                delta = max(timedelta(hours=12), (e - s) / max(1, limit))
                while cur <= e and count < limit:
                    preds.append({
                        "ts": cur.isoformat(),
                        "pred": round(20000 + random.uniform(-2000, 2000), 2),
                        "model": model
                    })
                    cur = cur + delta
                    count += 1
            except Exception:
                preds = []
        else:
            now = datetime.now().replace(minute=0, second=0, microsecond=0)
            for i in range(min(limit, 24)):
                ts = (now - timedelta(hours=i)).isoformat()
                preds.append({
                    "ts": ts,
                    "pred": round(20000 + random.uniform(-2000, 2000), 2),
                    "model": model
                })
            preds = list(reversed(preds))
        
        if not preds:
            return jsonify({"error": f"No se encontraron predicciones para el periodo {start_str} - {end_str}"}), 404
        
        return jsonify({"predictions": preds, "metadata": {"total": len(preds)}}), 200
    
    # Predicción normal con fecha
    if not date:
        return jsonify({"error": "Falta parámetro: date=YYYY-MM-DD"}), 400
        
    predictions_series, df_actuals, err = generate_prediction_series(target_date=date)
    if err:
        return jsonify({"error": err}), 400

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
    """
    Calcula KPIs (MAPE, RMSE) para una fecha específica.
    Query params:
      - date: fecha en formato YYYY-MM-DD (opcional, usa datos recientes si no se especifica)
    """
    date = request.args.get("date")
    predictions_series, df_actuals, err = generate_prediction_series(target_date=date)
    
    if err:
        return jsonify({"error": err}), 400
    if df_actuals is None or 'valor_real' not in df_actuals.columns:
        return jsonify({
            "window_points": 0,
            "mape": None,
            "rmse": None,
            "error": "No hay datos reales para comparar."
        })

    common_index = predictions_series.index.intersection(df_actuals.index)
    if common_index.empty:
        return jsonify({
            "window_points": 0,
            "mape": None,
            "rmse": None,
            "error": "No hay solapamiento entre predicción y reales."
        })

    y_pred = predictions_series[common_index]
    y_true = df_actuals.loc[common_index, 'valor_real']
    
    valid_idx = y_true.notna() & y_pred.notna()
    y_true = y_true[valid_idx].values
    y_pred = y_pred[valid_idx].values

    if len(y_true) == 0:
        return jsonify({"window_points": 0, "mape": None, "rmse": None})

    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100.0)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    return jsonify({
        "window_points": len(y_true),
        "mape": mape,
        "rmse": rmse
    })


@app.get("/api/anomalies")
def api_anomalies_for_date():
    """
    Detecta anomalías para una fecha específica.
    Query params:
      - date: fecha en formato YYYY-MM-DD (opcional)
      - z_threshold: umbral de desviaciones estándar (default: 2.5)
    """
    date = request.args.get("date")
    thresh = float(request.args.get("z_threshold", 2.5))
    
    predictions_series, df_actuals, err = generate_prediction_series(target_date=date)
    if err:
        return jsonify({"error": err}), 400
    
    results = _get_anomaly_results(predictions_series, df_actuals, thresh)
    if "error" in results:
        return jsonify(results), 400
    return jsonify(results)


@app.route("/api/model/performance", methods=["GET"])
def api_model_performance():
    """
    Endpoint de desarrollo para métricas de rendimiento del modelo.
    Devuelve datos mock si no hay información real disponible.
    """
    # Intentar obtener KPIs reales
    try:
        predictions_series, df_actuals, err = generate_prediction_series(target_date=None)
        if not err and df_actuals is not None:
            common_index = predictions_series.index.intersection(df_actuals.index)
            if not common_index.empty:
                y_pred = predictions_series[common_index]
                y_true = df_actuals.loc[common_index, 'valor_real']
                valid_idx = y_true.notna() & y_pred.notna()
                
                if valid_idx.any():
                    y_true_arr = y_true[valid_idx].values
                    y_pred_arr = y_pred[valid_idx].values
                    
                    mape = float(np.mean(np.abs((y_true_arr - y_pred_arr) / np.clip(np.abs(y_true_arr), 1e-6, None))) * 100.0)
                    rmse = float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))
                    mae = float(np.mean(np.abs(y_true_arr - y_pred_arr)))
                    
                    return jsonify({
                        "model": "LSTM",
                        "metrics": {
                            "mape": round(mape, 2),
                            "rmse": round(rmse, 2),
                            "mae": round(mae, 2)
                        },
                        "samples": len(y_true_arr),
                        "timestamp": datetime.now().isoformat()
                    }), 200
    except Exception as e:
        print(f"✗ Error calculando métricas reales: {e}")
    
    # Fallback a datos mock
    return jsonify({
        "model": "LSTM",
        "metrics": {
            "mape": round(random.uniform(2.0, 5.0), 2),
            "rmse": round(random.uniform(300, 600), 2),
            "mae": round(random.uniform(200, 400), 2)
        },
        "samples": 288,
        "timestamp": datetime.now().isoformat(),
        "note": "Datos mock para desarrollo"
    }), 200


# ==================== INICIALIZACIÓN ====================
# ==================== INICIALIZACIÓN ====================

if __name__ == "__main__":
    logger.info("=" * 70)
    logger.info("🌱 GreenEnergy Insights - API Principal Unificada")
    logger.info("=" * 70)
    
    # Inicializar base de datos
    db_connected = init_database()
    
    if not db_connected:
        logger.error("✗ No se pudo conectar a la base de datos")
        logger.warning("⚠ Sistema en modo desarrollo (BD desconectada)")
    else:
        logger.info("✓ Base de datos conectada")
    
    # Cargar modelos ML (solo si BD está conectada)
    if db_connected:
        logger.info("📦 Cargando modelos ML...")
        load_models_at_startup()
        
        # Verificar que los modelos se cargaron
        if MODELS:
            logger.info(f"✓ Modelos cargados: {list(MODELS.keys())}")
            
            # Pre-calentar cache de predicciones
            logger.info("🔥 Precalentando cache de predicciones...")
            try:
                warm_up_prediction_cache()
                if LATEST_PRED_CACHE.get("error"):
                    logger.warning(f"⚠ Cache inicializado con error: {LATEST_PRED_CACHE['error']}")
                else:
                    logger.info("✓ Cache precalentado exitosamente")
            except Exception as e:
                logger.error(f"✗ Error al precalentar cache: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning("⚠ Modelos ML no cargados, funcionalidad de predicción limitada")
    
    logger.info("=" * 70)
    logger.info("✓ Sistema listo")
    logger.info("  - Endpoints HTML: /, /dashboard, /documentacion, /chatbot")
    logger.info("  - Endpoints API: /api/series, /api/predictions, /api/kpis, /api/anomalies")
    logger.info("  - Health check: /health")
    logger.info("=" * 70)
    
    app.run(debug=True, port=5001, host="0.0.0.0")
