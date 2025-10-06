# api_main.py
import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
from flask import Flask, jsonify, request, send_file, render_template
import lightgbm as lgb

# Importaciones de Deep Learning (Asegúrate de que TensorFlow está instalado)
from tensorflow.keras.models import load_model 
from sklearn.preprocessing import MinMaxScaler 

# --- CONFIGURACIÓN GLOBAL Y ESTRUCTURA DEL MODELO ---
app = Flask(__name__)
MODELS = {}
SCALER = MinMaxScaler() 

# Constantes del modelo (Debe coincidir con tu entrenamiento LSTM/LGBM)
LOOK_BACK = 24 
NUM_FEATURES = 3 # (demanda_real, demanda_prevista, demanda_programada)

# --- FUNCIÓN DE UTILIDAD: CARGA DE OBJETOS ---
def load_models_at_startup():
    """Carga los modelos pesados y el objeto scaler una única vez al iniciar la API."""
    global MODELS
    
    # 1. Simulación o Carga del Scaler (CRUCIAL para LSTM)
    # NOTA: En producción, aquí cargarías el objeto SCALER guardado con joblib/pickle.
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
        MODELS['LGBM'] = lgb.Booster(model_file="modelo_demanda_lgbm.txt")
        print("INFO: Modelo LightGBM cargado.")
    except Exception as e:
        print(f"ERROR: No se pudo cargar el modelo LightGBM. {e}")


# --- FUNCIÓN DE PLOT Y PREDICCIÓN ---
def generate_prediction_plot_image(model_name: str):
    """Genera la predicción simulada, la gráfica y la devuelve como buffer de bytes."""
    if model_name not in MODELS:
        return None, f"Modelo '{model_name}' no cargado o no encontrado."

    # --- SIMULACIÓN DE RESULTADOS REALES Y PREDICCIÓN ---
    # *En la realidad, AQUÍ va la lógica de obtención de datos, escalado y predicción.*
    dates = pd.to_datetime(pd.date_range(start='2025-09-22', periods=100, freq='5T'))
    real_demand = 27000 + np.sin(np.arange(100) / 10) * 4000 + np.random.normal(0, 150, 100)
    
    model_type = "LSTM" if model_name == 'LSTM' else "LGBM"
    color = 'forestgreen' if model_type == 'LSTM' else 'darkorange'
    
    # Simulación de la predicción
    prediction = real_demand + np.random.normal(0, 250, 100) * (0.5 if model_type == 'LSTM' else 1.5) 

    # --- Generación de la Gráfica en Memoria ---
    img_data = io.BytesIO()
    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.plot(dates, real_demand, label='Demanda Real (MW)', linewidth=1.5, color='steelblue')
    plt.plot(dates, prediction, label=f'Predicción {model_name}', linewidth=2.5, alpha=0.8, color=color)
    
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


# --- ENDPOINT DE PREDICCIÓN (Devuelve el HTML con la gráfica) ---
@app.route("/predict_and_plot/<model_name>", methods=["GET"])
def predict_plot(model_name):
    """
    Genera la gráfica de predicción y la renderiza dentro de un HTML 
    con el botón de cierre.
    """
    model_name = model_name.upper()
    
    img_data_buffer, error = generate_prediction_plot_image(model_name)
    
    if error:
        return jsonify({"status": "error", "message": error}), 500
        
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