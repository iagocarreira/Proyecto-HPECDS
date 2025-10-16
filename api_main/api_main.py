import os
import io
import pandas as pd
import numpy as np

# Configurar el backend de Matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from datetime import datetime, timedelta

from flask import Flask, request, render_template

# Importaciones de Modelado y BD
from sqlalchemy import create_engine, MetaData, Table, select, and_
from tensorflow.keras.models import load_model
import joblib

# --- CONFIGURACIÓN GLOBAL Y PARÁMETROS CRÍTICOS ---
app = Flask(__name__)
MODELS, SCALER = {}, None
LOOK_BACK, NUM_FEATURES, PREDICTION_STEPS = 24, 3, 288
PREDICTION_DATA_WINDOW_DAYS, API_DATA_URL = 3, "http://127.0.0.1:5002"

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


def load_models_at_startup():
    global MODELS, SCALER
    try:
        SCALER = joblib.load("scaler_lstm_multivariate.joblib")
        print("INFO: Scaler cargado.")
    except Exception as e:
        print(f"ERROR CRÍTICO al cargar el scaler: {e}")
    try:
        MODELS['LSTM'] = load_model("modelo_lstm_multivariate.keras")
        print("INFO: Modelo LSTM cargado.")
    except Exception as e:
        print(f"ERROR al cargar el modelo LSTM: {e}")

def fetch_data_from_db(table_to_query: Table, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    if not engine or table_to_query is None: return pd.DataFrame()
    query = select(table_to_query).where(and_(table_to_query.c.fecha >= start_date, table_to_query.c.fecha <= end_date)).order_by(table_to_query.c.fecha)
    try:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
            print(f"DEBUG: Se obtuvieron {len(df)} registros de '{table_to_query.name}'")
            return df
    except Exception as e:
        print(f"ERROR al consultar '{table_to_query.name}': {e}")
        return pd.DataFrame()

def generate_prediction_plot_image(model_name: str, target_date: str = None):
    # --- 1. Comprobaciones y selección de tabla ---
    if 'LSTM' not in MODELS: return None, "Error: Modelo LSTM no cargado."
    if SCALER is None: return None, "Error crítico: SCALER no cargado."

    # --- 2. Lógica de obtención de datos ---
    features = ['valor_real', 'valor_previsto', 'valor_programado']

    if target_date:
        # MODO 1: Predecir un día específico (requiere datos futuros)
        prediction_day = datetime.strptime(target_date, "%Y-%m-%d")
        seven_days_ago = datetime.now() - timedelta(days=7)
        table_to_use = tabla_semanal if prediction_day.date() >= seven_days_ago.date() else tabla_historica
        if table_to_use is None: return None, "Error: Conexión a BD no disponible."

        end_history = prediction_day - timedelta(microseconds=1)
        start_history = end_history - timedelta(days=PREDICTION_DATA_WINDOW_DAYS)
        df_history = fetch_data_from_db(table_to_use, start_history, end_history)
        
        end_prediction_day = prediction_day + timedelta(days=1) - timedelta(microseconds=1)
        df_future_features = fetch_data_from_db(table_to_use, prediction_day, end_prediction_day)
        if len(df_future_features) < PREDICTION_STEPS: return None, f"Faltan datos de 'previsto'/'programado' para el día {prediction_day.date()}. Se necesitan {PREDICTION_STEPS}."
        
        df_actuals_for_plot = df_future_features.copy()
        
    else:
        # MODO 2: Predecir "a partir de ahora" (usando proxy de datos de hace un año)
        table_to_use = tabla_semanal
        end_dt, start_dt = datetime.now(), datetime.now() - timedelta(days=PREDICTION_DATA_WINDOW_DAYS)
        df_history = fetch_data_from_db(table_to_use, start_dt, end_dt)
        if len(df_history) < LOOK_BACK: return None, f"Datos históricos insuficientes ({len(df_history)}). Se necesitan {LOOK_BACK}."
        
        df_actuals_for_plot = df_history.copy()
        prediction_day = df_history['fecha'].iloc[-1]
        
        # --- ¡NUEVA LÓGICA! ---
        # Buscamos los datos de 'previsto' y 'programado' de hace un año como sustitutos.
        print("DEBUG: Buscando datos proxy de hace un año para 'previsto' y 'programado'...")
        start_proxy = prediction_day - timedelta(days=365)
        end_proxy = start_proxy + timedelta(days=1)
        df_proxy_futures = fetch_data_from_db(tabla_historica, start_proxy, end_proxy)
        
        if len(df_proxy_futures) < PREDICTION_STEPS:
            return None, "No se encontraron datos de 'previsto'/'programado' para hoy, y tampoco hay datos de respaldo de hace un año."
        
        # Renombramos las columnas para que coincidan y las usamos como nuestros "datos futuros"
        df_future_features = df_proxy_futures.copy()
        # --- FIN NUEVA LÓGICA ---

    # --- 3. Preparar datos para el bucle de predicción ---
    df_history_processed = df_history[features].fillna(method='ffill').fillna(0)
    scaled_history = SCALER.transform(df_history_processed)
    
    future_features_processed = df_future_features[features].fillna(method='ffill').fillna(0)
    scaled_future_features = SCALER.transform(future_features_processed)

    current_batch = scaled_history[-LOOK_BACK:].reshape(1, LOOK_BACK, NUM_FEATURES)
    all_predictions_scaled = []

    # --- 4. Bucle de predicción autoregresiva ---
    model_lstm = MODELS['LSTM']
    for i in range(PREDICTION_STEPS):
        next_pred_scaled = model_lstm.predict(current_batch, verbose=0)
        all_predictions_scaled.append(next_pred_scaled[0, 0])
        
        new_row = np.array([
            next_pred_scaled[0, 0],
            scaled_future_features[i, 1],
            scaled_future_features[i, 2]
        ]).reshape(1, 1, NUM_FEATURES)
        current_batch = np.append(current_batch[:, 1:, :], new_row, axis=1)

    # --- 5. Invertir la escala y preparar la gráfica ---
    predictions_dummy = np.zeros((len(all_predictions_scaled), NUM_FEATURES)); predictions_dummy[:, 0] = all_predictions_scaled
    predictions_real_mw = SCALER.inverse_transform(predictions_dummy)[:, 0]

    if not df_actuals_for_plot.empty:
        real_demand, dates_real = df_actuals_for_plot['valor_real'], pd.to_datetime(df_actuals_for_plot['fecha'])
    else:
        real_demand, dates_real = pd.Series([]), pd.Series([])

    dates_pred = pd.to_datetime(pd.date_range(start=prediction_day, periods=PREDICTION_STEPS, freq='5min'))

    # --- 6. Generar la gráfica ---
    plt.figure(figsize=(14, 7)); plt.style.use('seaborn-v0_8-whitegrid')
    plt.plot(dates_real, real_demand, label='Demanda Real (MW)', color='steelblue', marker='.', markersize=4)
    plt.plot(dates_pred, predictions_real_mw, label='Predicción LSTM', color='darkorange', linestyle='--', linewidth=2)
    title_date = f"para el día {prediction_day.date()}" if target_date else f"a partir de {prediction_day.strftime('%Y-%m-%d %H:%M')}"
    plt.title(f'Predicción de Demanda con LSTM {title_date}', fontsize=16)
    plt.xlabel("Tiempo"); plt.ylabel("Potencia (MW)"); plt.legend()
    all_values = pd.concat([pd.Series(predictions_real_mw), real_demand])
    if not all_values.dropna().empty: plt.ylim(all_values.min()*0.9, all_values.max()*1.1)
    plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
    
    img_data = io.BytesIO(); plt.savefig(img_data, format='png'); plt.close()
    img_data.seek(0)
    return img_data, None

# --- INICIALIZACIÓN Y ENDPOINTS ---
load_models_at_startup()
@app.route("/")
def home(): return render_template("index.html", api_data_url=API_DATA_URL)

@app.route("/predict_latest/<model_name>", methods=["GET"])
def predict_latest(model_name):
    img_data_buffer, error = generate_prediction_plot_image('LSTM', target_date=None)
    if error: return render_template("plot_view.html", error=error, model='LSTM')
    img_base64 = base64.b64encode(img_data_buffer.read()).decode('utf-8')
    return render_template("plot_view.html", model='LSTM', img_data=img_base64)

@app.route("/predict_custom", methods=["POST"])
def predict_custom():
    # 1. Obtenemos la fecha del campo correcto del formulario ('desde_fecha')
    prediction_date = request.form.get("desde_fecha")
    
    # 2. El resto de la lógica es idéntica
    if not prediction_date:
        return render_template("plot_view.html", error="No se seleccionó ninguna fecha.", model='LSTM')
        
    img_data_buffer, error = generate_prediction_plot_image('LSTM', target_date=prediction_date)
    
    if error:
        return render_template("plot_view.html", error=error, model='LSTM')
        
    img_base64 = base64.b64encode(img_data_buffer.read()).decode('utf-8')
    
    # Pasamos la fecha a la plantilla para que se muestre en el título
    return render_template("plot_view.html", model='LSTM', img_data=img_base64, date=prediction_date)
if __name__ == "__main__":
    app.run(debug=True, port=5001)