
import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request, send_file, render_template 
import lightgbm as lgb

# Importaciones de Deep Learning 
from tensorflow.keras.models import load_model 
from sklearn.preprocessing import MinMaxScaler 


# --- CONFIGURACIÓN GLOBAL Y ESTRUCTURA DEL MODELO ---
app = Flask(__name__)
MODELS = {}
SCALER = MinMaxScaler() 

# Constantes del modelo (de tu entrenamiento LSTM/LGBM)
LOOK_BACK = 24 
NUM_FEATURES = 3 


# --- LÓGICA DE PLOT Y PREDICCIÓN (Integrada) ---
def generate_prediction_plot_image(model_name: str):
    
    if model_name not in MODELS:
        return None, f"Modelo '{model_name}' no cargado o no encontrado."

    

    # --- SIMULACIÓN DE RESULTADOS REALES Y PREDICCIÓN ---
    dates = pd.to_datetime(pd.date_range(start='2025-09-22', periods=100, freq='5T'))
    real_demand = 27000 + np.sin(np.arange(100) / 10) * 4000 + np.random.normal(0, 150, 100)
    
    model_type = "LSTM" if model_name == 'LSTM' else "LGBM"
    color = 'forestgreen' if model_type == 'LSTM' else 'darkorange'
    
    
    prediction = real_demand + np.random.normal(0, 250, 100) * (0.5 if model_type == 'LSTM' else 1.5) 

    
    img_data = io.BytesIO()
    plt.figure(figsize=(12, 6))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    plt.plot(dates, real_demand, label='Demanda Real (MW)', linewidth=1.5, color='steelblue')
    plt.plot(dates, prediction, label=f'Predicción {model_name} (Carga Rápida)', linewidth=2.5, alpha=0.8, color=color)
    
    plt.title(f'Resultados de Predicción | Modelo: {model_name}', fontsize=16)
    plt.xlabel("Tiempo"); plt.ylabel("Potencia (MW)"); plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6); plt.tight_layout()
    
    
    plt.savefig(img_data, format='png')
    plt.close() 
    img_data.seek(0) 
    return img_data, None


# --- FUNCIÓN DE UTILIDAD: CARGA DE OBJETOS ---
def load_models_at_startup():
    
    global MODELS
    
    temp_data = np.zeros((10, NUM_FEATURES)) 
    SCALER.fit(temp_data)
    print("DEBUG: Scaler simulado cargado.")
    
    try:
        # 1. Carga del Modelo LSTM (Keras)
        MODELS['LSTM'] = load_model("modelo_lstm_multivariate.keras") 
        print("INFO: Modelo LSTM cargado.")
    except Exception as e:
        print(f"ERROR: No se pudo cargar el modelo LSTM. Asegúrese de que el archivo existe. {e}")
        
    try:
        # 2. Carga del Modelo LightGBM
        MODELS['LGBM'] = lgb.Booster(model_file="modelo_demanda_lgbm.txt")
        print("INFO: Modelo LightGBM cargado.")
    except Exception as e:
        print(f"ERROR: No se pudo cargar el modelo LightGBM. Asegúrese de que el archivo existe. {e}")



load_models_at_startup() 

# --- ENDPOINT PRINCIPAL (El que cargará la página) ---
@app.route("/", methods=["GET"])
def home():
    """
    Endpoint raíz que sirve la página de bienvenida con los enlaces de selección de modelo.
    """
    return render_template("index.html")


# --- ENDPOINT PRINCIPAL ---
@app.route("/predict_and_plot/<model_name>", methods=["GET"])
def predict_plot(model_name):
    """
    ENDPOINT para seleccionar un modelo y devolver su gráfica de predicción.
    """
    model_name = model_name.upper()
    
    img_data, error = generate_prediction_plot_image(model_name)
    
    if error:
        return jsonify({"status": "error", "message": error}), 500
        
   
    return send_file(img_data, mimetype='image/png')


# --- EJECUCIÓN ---
if __name__ == "__main__":
    
    app.run(debug=True)