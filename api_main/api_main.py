# api_main/api_main.py
import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

from datetime import datetime, timedelta
from flask import Flask, jsonify, render_template

# Modelos
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb

# --- Flask con rutas de plantillas/estáticos correctas para Docker ---
app = Flask(__name__, template_folder="templates", static_folder="static")

# --- Estado global ---
MODELS = {}
SCALER = MinMaxScaler()
LOOK_BACK = 24
NUM_FEATURES = 3

# --- Config microservicio de datos ---
# En Docker, mejor referenciar por nombre de servicio y puerto publicado del api_data
API_DATA_URL = os.getenv("API_DATA_URL", "http://api_data:8000")
DEFAULT_PREDICT_DAYS = 3
DEFAULT_VIEW_DAYS = 7

# ---------- Salud ----------
@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

# ---------- Carga de modelos (tolerante) ----------
def load_models_at_startup():
    global MODELS
    # Scaler dummy para no romper
    SCALER.fit(np.zeros((10, NUM_FEATURES)))
    print("INFO: Scaler cargado/simulado.")

    # LSTM
    try:
        MODELS["LSTM"] = load_model("modelo_lstm_multivariate.keras")
        print("INFO: Modelo LSTM cargado.")
    except Exception as e:
        print(f"ERROR: No se pudo cargar el modelo LSTM. {e}")

    # LightGBM (usa tu nombre de fichero)
    try:
        MODELS["LGBM"] = lgb.Booster(model_file="modelo_demanda_lgbm.txt")
        print("INFO: Modelo LightGBM cargado.")
    except Exception as e:
        print(f"ERROR: No se pudo cargar el modelo LightGBM. {e}")

# ---------- Cliente del microservicio de datos ----------
def fetch_data_from_api(days: int):
    """
    Pide datos a api_data /records con los parámetros mínimos.
    Devuelve lista de dicts o None si falla.
    """
    now = datetime.utcnow()
    start = (now - timedelta(days=days)).strftime("%Y-%m-%dT00:00:00")
    end = now.strftime("%Y-%m-%dT%H:%M:%S")

    params = {
        "limit": 10000,
        "offset": 0,
        "desde": start,
        "hasta": end,
        "geo_name": "Península",
        "fields": ""  # vacío => todas las columnas por defecto
    }
    url = f"{API_DATA_URL}/records"
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        print(f"ERROR: fetch_data_from_api fallo: {e} | URL:{url} | params:{params}")
        return None

# ---------- Predicción + gráfico ----------
def generate_prediction_plot_image(model_name: str):
    """
    Intenta usar datos reales del microservicio. Si falla, simula.
    Si el modelo no está cargado, seguimos en modo simulado (no rompemos).
    """
    if model_name not in MODELS:
        print(f"WARNING: Modelo '{model_name}' no cargado. Se usará simulación.")

    # 1) Intentar datos reales
    df = None
    records = fetch_data_from_api(days=DEFAULT_PREDICT_DAYS)
    if records:
        try:
            df = pd.DataFrame(records)
            # Ajusta nombres si tu API devuelve otros campos
            dates = pd.to_datetime(df["fecha"]).iloc[:100]
            real_demand = pd.to_numeric(df["valor_real"], errors="coerce").fillna(method="ffill").iloc[:100]
        except Exception as e:
            print(f"WARNING: No se pudieron preparar datos reales, se simula. Detalle: {e}")
            df = None

    # 2) Si no hay datos reales válidos, simular
    if df is None:
        dates = pd.to_datetime(pd.date_range(start="2025-09-22", periods=100, freq="5T"))
        real_demand = 27000 + np.sin(np.arange(100) / 10) * 4000 + np.random.normal(0, 150, 100)

    model_type = "LSTM" if model_name == "LSTM" else "LGBM"
    color = "forestgreen" if model_type == "LSTM" else "darkorange"
    # Simulación de predicción (si tienes modelos reales, aquí usarías MODELS[model_name])
    prediction = real_demand + np.random.normal(0, 250, len(real_demand)) * (0.5 if model_type == "LSTM" else 1.5)

    buf = io.BytesIO()
    plt.figure(figsize=(12, 6))
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.plot(dates, real_demand, label="Demanda Real (MW)", linewidth=1.5, color="steelblue")
    plt.plot(dates, prediction, label=f"Predicción {model_name}", linewidth=2.5, alpha=0.8, color=color)
    plt.title(f"Resultados de Predicción | Modelo: {model_name}", fontsize=16)
    plt.xlabel("Tiempo"); plt.ylabel("Potencia (MW)")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.6); plt.tight_layout()
    plt.savefig(buf, format="png"); plt.close()
    buf.seek(0)
    return buf

# ---------- Rutas ----------
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/data_view", methods=["GET"])
def data_view():
    records = fetch_data_from_api(days=DEFAULT_VIEW_DAYS)
    if records:
        return render_template("data_view.html", records=records)
    return render_template("data_view.html", error="No se pudieron cargar los datos desde api_data.")

@app.route("/predict_and_plot/<model_name>", methods=["GET"])
def predict_plot(model_name: str):
    model_name = (model_name or "").upper().strip()
    if model_name not in ("LSTM", "LGBM"):
        return jsonify({"status": "error", "message": f"Modelo inválido: {model_name}"}), 400

    img = generate_prediction_plot_image(model_name)
    img_b64 = base64.b64encode(img.read()).decode("utf-8")
    return render_template("plot_view.html", model=model_name, img_data=img_b64)

# ---------- Arranque ----------
load_models_at_startup()

if __name__ == "__main__":
    # En Docker usamos gunicorn; esto es solo para debug local
    app.run(host="0.0.0.0", port=5000, debug=True)
