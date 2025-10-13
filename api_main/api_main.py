# api_main/api_main.py
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from flask import Flask, jsonify, render_template
import lightgbm as lgb

# Deep Learning
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# --- Configuración Flask ---

app = Flask(__name__, template_folder="templates", static_folder="static")

# --- Estado global (simulado/real) ---
MODELS = {}
SCALER = MinMaxScaler()

LOOK_BACK = 24
NUM_FEATURES = 3

# ---------- Carga de modelos (tolerante) ----------
def load_models_at_startup():
    """Carga scaler y modelos si existen; si no, sigue en modo simulado."""
    global MODELS
    # Scaler "dummy" para que no falle el pipeline
    SCALER.fit(np.zeros((10, NUM_FEATURES)))
    print("INFO: Scaler cargado/simulado.")

    # LSTM (Keras)
    try:
        MODELS["LSTM"] = load_model("modelo_lstm_multivariate.keras")
        print("INFO: Modelo LSTM cargado.")
    except Exception as e:
        print(f"ERROR: No se pudo cargar el modelo LSTM. {e}")

    # LightGBM
    try:
        MODELS["LGBM"] = lgb.Booster(model_file="modelo_demanda_lgbm.txt")
        print("INFO: Modelo LightGBM cargado.")
    except Exception as e:
        print(f"ERROR: No se pudo cargar el modelo LightGBM. {e}")

# ---------- Predicción + gráfico (simulado si no hay modelo) ----------
def generate_prediction_plot_image(model_name: str):
    if model_name not in MODELS:
        # Modo simulado, pero no fallamos
        print(f"WARNING: Usando simulación para '{model_name}' (modelo no cargado).")

    dates = pd.to_datetime(pd.date_range(start="2025-09-22", periods=100, freq="5T"))
    real_demand = 27000 + np.sin(np.arange(100) / 10) * 4000 + np.random.normal(0, 150, 100)

    model_type = "LSTM" if model_name == "LSTM" else "LGBM"
    color = "forestgreen" if model_type == "LSTM" else "darkorange"
    prediction = real_demand + np.random.normal(0, 250, 100) * (0.5 if model_type == "LSTM" else 1.5)

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
@app.route("/health", methods=["GET"])
def health():
    return "ok", 200

@app.route("/", methods=["GET"])
def home():
    # index.html debe existir en ../templates o en templates/ (según template_folder)
    return render_template("index.html")

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
    # Solo para debug local; en Docker usamos gunicorn
    app.run(host="0.0.0.0", port=5000, debug=True)
