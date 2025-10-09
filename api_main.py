# api_main.py
import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import requests 
from datetime import datetime, timedelta # Necesario para calcular las fechas de la API de Datos
from flask import Flask, jsonify, request, send_file, render_template

# Importaciones de Modelado
from tensorflow.keras.models import load_model 
from sklearn.preprocessing import MinMaxScaler 
import lightgbm as lgb 

# --- CONFIGURACIÓN GLOBAL Y ESTRUCTURA DEL MODELO ---
app = Flask(__name__)
MODELS = {}
SCALER = MinMaxScaler() 

# Constantes del modelo (Debe coincidir con tu entrenamiento LSTM/LGBM)
LOOK_BACK = 24 
NUM_FEATURES = 3 # (demanda_real, demanda_prevista, demanda_programada)

# CONFIGURACIÓN DEL MICROSERVICIO DE DATOS (Puerto 5002)
API_DATA_URL = "http://127.0.0.1:5002" 
DEFAULT_PREDICT_DAYS = 3 # Predice sobre los datos de los últimos 3 días
DEFAULT_VIEW_DAYS = 7    # Muestra datos de los últimos 7 días


def fetch_data_from_api(days: int):
    """
    Solicita datos históricos a la API de Datos (http://127.0.0.1:5002/records).
    Incluye todos los parámetros necesarios para satisfacer la validación de FastAPI.
    """
    
    # 1. Calcular el rango de fechas en formato ISO completo
    hoy = datetime.now()
    inicio_dt = hoy - timedelta(days=days)

    # Formato ISO 8601 explícito: Inicio del día y fin a la hora actual (sin segundos flotantes)
    inicio = inicio_dt.strftime("%Y-%m-%d") + "T00:00:00" 
    fin = hoy.strftime("%Y-%m-%d") + hoy.strftime("T%H:%M:%S") 

    # 2. Parámetros explícitos para satisfacer la validación de FastAPI
    limit = 10000
    offset = 0
    geo_name = "Península" # Valor por defecto extraído del formulario de tu compañera
    fields = "" # Cadena vacía para devolver todas las columnas por defecto

    # 3. Construir la URL con TODOS los parámetros necesarios
    url = (f"{API_DATA_URL}/records?limit={limit}&offset={offset}"
           f"&desde={inicio}&hasta={fin}&geo_name={geo_name}&fields={fields}")

    print(f"DEBUG: Llamando a API de Datos con URL: {url}")
    
    try:
        response = requests.get(url) 
        # Si la API de Datos responde con un error (4xx o 5xx), se lanza la excepción.
        response.raise_for_status() 
        
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Fallo al conectar con la API de Datos. URL: {url}. Detalles: {e}")
        return None
    
# --- FUNCIÓN DE UTILIDAD: CARGA DE OBJETOS ---
def load_models_at_startup():
    """Carga los modelos pesados y el objeto scaler una única vez al iniciar la API."""
    global MODELS
    
    # 1. Simulación o Carga del Scaler
    temp_data = np.zeros((10, NUM_FEATURES)) 
    SCALER.fit(temp_data)
    print("INFO: Scaler cargado/simulado.")
    
    # 2. Carga del Modelo LSTM (Keras)
    try:
        MODELS['LSTM'] = load_model("modelo_lstm_multivariate.keras") 
        print("INFO: Modelo LSTM cargado.")
    except Exception as e:
        print(f"ERROR: No se pudo cargar el modelo LSTM. {e}")
        
    # 3. Carga del Modelo LightGBM
    try:
        MODELS['LGBM'] = lgb.Booster(model_file="modelo_lgbm_multivariate.txt")
        print("INFO: Modelo LightGBM cargado.")
    except Exception as e:
        print(f"ERROR: No se pudo cargar el modelo LightGBM. {e}")


# --- FUNCIÓN DE PLOT Y PREDICCIÓN ---
def generate_prediction_plot_image(model_name: str):
    """Genera la predicción usando datos del microservicio y devuelve la gráfica."""
    if model_name not in MODELS:
        return None, f"Modelo '{model_name}' no cargado o no encontrado."

    # 1. FETCHING DE DATOS REALES (usamos los últimos 3 días por defecto)
    historical_data = fetch_data_from_api(days=DEFAULT_PREDICT_DAYS)
    if not historical_data:
        return None, "No se pudo obtener datos para la predicción. La API de Datos está inaccesible."
    
    df_data = pd.DataFrame(historical_data)
    # *******************************************************************
    # En la realidad, AQUÍ iría la lógica completa de:
    # 1. Preprocesar y re-secuenciar df_data.
    # 2. ESCALAR (con SCALER) y predecir con MODELS[model_name].
    # *******************************************************************

    # --- SIMULACIÓN DE RESULTADOS ---
    # Usamos los datos obtenidos para que el plot sea coherente.
    real_demand = df_data['valor_real'].iloc[:100]
    dates = pd.to_datetime(df_data['fecha']).iloc[:100]
    
    model_type = "LSTM" if model_name == 'LSTM' else "LGBM"
    color = 'forestgreen' if model_type == 'LSTM' else 'darkorange'
    
    # Simulación de la predicción, usando el valor real para el eje.
    prediction = real_demand + np.random.normal(0, 250, len(real_demand)) * (0.5 if model_type == 'LSTM' else 1.5) 

    # --- Generación de la Gráfica en Memoria ---
    img_data = io.BytesIO()
    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.plot(dates, real_demand, label='Demanda Real (MW)', linewidth=1.5, color='steelblue')
    plt.plot(dates, prediction, label=f'Predicción {model_name} ({DEFAULT_PREDICT_DAYS} Días)', linewidth=2.5, alpha=0.8, color=color)
    
    plt.title(f'Resultados de Predicción | Modelo: {model_name}', fontsize=16)
    plt.xlabel("Tiempo"); plt.ylabel("Potencia (MW)"); plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
    
    plt.savefig(img_data, format='png')
    plt.close()
    img_data.seek(0)
    return img_data, None


# --- ¡NUEVO! LLAMADA DE INICIALIZACIÓN ---
load_models_at_startup() 


# --- ENDPOINT PRINCIPAL (Página de Selección) ---
@app.route("/", methods=["GET"])
def home():
    """Sirve la página principal (Hero Section)."""
    return render_template("index.html")


# --- ENDPOINT DE VISUALIZACIÓN DE DATOS (/data_view) ---
@app.route("/data_view", methods=["GET"])
def data_view():
    """Obtiene datos de la API de Datos y los muestra en una tabla HTML."""
    records = fetch_data_from_api(days=DEFAULT_VIEW_DAYS)
    
    if records:
        # Asumiendo que la API de datos devuelve una lista de diccionarios
        return render_template("data_view.html", records=records)
    else:
        return render_template("data_view.html", error="No se pudieron cargar los datos. Verifique que la API de Datos (Puerto 5002) esté corriendo.")


# --- ENDPOINT DE PREDICCIÓN (Devuelve el HTML con la gráfica) ---
@app.route("/predict_and_plot/<model_name>", methods=["GET"])
def predict_plot(model_name):
    """
    Genera la gráfica de predicción, la codifica y la renderiza en el visor HTML.
    """
    model_name = model_name.upper()
    
    img_data_buffer, error = generate_prediction_plot_image(model_name)
    
    if error:
        # Usa la plantilla de visor para mostrar el error
        return render_template("plot_view.html", error=error) 
        
    # Codificar la imagen a Base64 para incrustarla en el HTML
    img_bytes = img_data_buffer.read()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    
    # Renderizar la plantilla que muestra la imagen
    return render_template(
        "plot_view.html",
        model=model_name,
        img_data=img_base64
    )


# --- EJECUCIÓN ---
if __name__ == "__main__":
    app.run(debug=True)