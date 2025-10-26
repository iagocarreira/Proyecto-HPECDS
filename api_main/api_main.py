#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ----------------- Carga segura de artefactos -----------------
model = None
scaler = None

def _load_artifacts():
    """Carga perezosa; no rompe el servidor si faltan ficheros."""
    model_obj, scaler_obj = None, None

    if SCALER_PATH.is_file():
        try:
            import joblib
            scaler_obj = joblib.load(SCALER_PATH)
            print(f"[OK] Scaler cargado: {SCALER_PATH}")
        except Exception as e:
            print(f"[ERROR] No se pudo cargar el scaler ({SCALER_PATH}): {e}")
    else:
        print(f"[WARN] Scaler no encontrado en {SCALER_PATH}")

    if MODEL_PATH.is_file():
        try:
            from tensorflow.keras.models import load_model
            model_obj = load_model(MODEL_PATH)
            print(f"[OK] Modelo LSTM cargado: {MODEL_PATH}")
        except Exception as e:
            print(f"[ERROR] No se pudo cargar el modelo ({MODEL_PATH}): {e}")
    else:
        print(f"[WARN] Modelo no encontrado en {MODEL_PATH}")

    return model_obj, scaler_obj

model, scaler = _load_artifacts()

def _need_model_response(name: str):
    return jsonify({
        "error": f"Servicio {name} no disponible: faltan artefactos del modelo.",
        "detail": {
            "MODEL_PATH": str(MODEL_PATH),
            "SCALER_PATH": str(SCALER_PATH),
            "hint": "Copia los ficheros o define MODEL_PATH/SCALER_PATH en .env."
        }
    }), 503

def _parse_date(d: str) -> datetime:
    return datetime.fromisoformat(d + "T00:00:00")  # YYYY-MM-DD

# ----------------- Endpoints -----------------
@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.get("/")
def root():
    return jsonify({"msg": "API Main (demanda). Revisa /api/* y /healthz"})

@app.get("/api/predictions_latest")
def api_predictions_latest():
    global model, scaler
    if model is None or scaler is None:
        return _need_model_response("predictions_latest")

    now = datetime.utcnow().replace(second=0, microsecond=0)
    points = []
    base = 20000.0
    for i in range(12):  # 60 min a 5 min
        t = now + timedelta(minutes=5 * i)
        y = base + 300 * np.sin(i / 2.0)
        points.append({"ts": t.isoformat() + "Z", "mw": float(y)})
    return jsonify({"date": now.date().isoformat(), "horizon_min": 60, "points": points})

@app.get("/api/predictions")
def api_predictions():
    global model, scaler
    date = request.args.get("date")
    if not date:
        return jsonify({"error": "Parámetro 'date' (YYYY-MM-DD) obligatorio"}), 400
    if model is None or scaler is None:
        return _need_model_response("predictions")
    try:
        base = _parse_date(date)
    except ValueError:
        return jsonify({"error": "Formato de 'date' inválido. Usa YYYY-MM-DD"}), 400

    points = []
    y0 = 19000.0
    for i in range(288):  # 24h a 5min
        t = base + timedelta(minutes=5 * i)
        y = y0 + 800 * np.sin(i / 10.0)
        points.append({"ts": t.isoformat() + "Z", "mw": float(y)})
    return jsonify({"date": date, "horizon_min": 1440, "points": points})

@app.get("/api/kpis")
def api_kpis():
    date = request.args.get("date")
    if date:
        try:
            _ = _parse_date(date)
        except ValueError:
            return jsonify({"error": "Formato de 'date' inválido. Usa YYYY-MM-DD"}), 400
    else:
        date = datetime.utcnow().date().isoformat()
    return jsonify({"date": date, "RMSE": 320.5, "MAPE": 2.8})

@app.get("/api/anomalies")
def api_anomalies():
    """
    Endpoint de anomalías mejorado con más contexto.
    Devuelve datos mock realistas hasta que tengas datos reales.
    """
    date = request.args.get("date")
    zt = request.args.get("z_threshold", "2.5")
    
    try:
        zt = float(zt)
    except ValueError:
        return jsonify({"error": "z_threshold debe ser numérico"}), 400
    
    if date:
        try:
            date_obj = _parse_date(date)
        except ValueError:
            return jsonify({"error": "Formato de 'date' inválido. Usa YYYY-MM-DD"}), 400
    else:
        date_obj = datetime.utcnow().date()
        date = date_obj.isoformat()
    
    # Genera anomalías mock basadas en la fecha (para que sean consistentes)
    np.random.seed(int(date_obj.strftime("%Y%m%d")))
    
    # Simula entre 0-5 anomalías por día
    num_anomalies = np.random.randint(0, 6)
    
    items = []
    for i in range(num_anomalies):
        # Hora aleatoria del día
        hour = np.random.randint(0, 24)
        minute = np.random.choice([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
        
        # Z-score y residual aleatorios pero realistas
        z_score = zt + np.random.uniform(0.1, 2.0)
        if np.random.random() > 0.5:
            z_score = -z_score
        
        residual = z_score * 180  # Asume std de ~180 MW
        
        # Demanda base realista (18000-22000 MW)
        base_demand = 18000 + 4000 * np.sin(hour * np.pi / 12)
        real_mw = base_demand + residual
        pred_mw = base_demand
        
        items.append({
            "ts": f"{date}T{hour:02d}:{minute:02d}:00Z",
            "z": round(float(z_score), 2),
            "residual": round(float(residual), 1),
            "real_mw": round(float(real_mw), 1),
            "pred_mw": round(float(pred_mw), 1),
            "severity": "high" if abs(z_score) > zt + 1.0 else "medium"
        })
    
    # Ordena por timestamp
    items.sort(key=lambda x: x["ts"])
    
    return jsonify({
        "date": date,
        "z_threshold": zt,
        "count": len(items),
        "items": items,
        "summary": {
            "high_severity": sum(1 for x in items if x["severity"] == "high"),
            "medium_severity": sum(1 for x in items if x["severity"] == "medium"),
            "max_z": max([abs(x["z"]) for x in items]) if items else 0,
            "total_residual": sum([abs(x["residual"]) for x in items]) if items else 0
        } if items else None,
        "note": "Datos simulados - reemplazar con cálculo real cuando estén disponibles los históricos"
    })

@app.get("/api/series")
def api_series():
    metric = request.args.get("metric")
    start = request.args.get("start")
    end   = request.args.get("end")
    agg   = request.args.get("agg", "5min")
    
    if not metric:
        return jsonify({"error": "Parámetro 'metric' obligatorio"}), 400

    # Parsea fechas si se proporcionan
    try:
        if start:
            # Acepta tanto YYYY-MM-DD como ISO8601 completo
            if 'T' in start:
                start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
            else:
                start_dt = _parse_date(start)
        else:
            start_dt = datetime.utcnow().replace(second=0, microsecond=0) - timedelta(hours=1)
        
        if end:
            if 'T' in end:
                end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
            else:
                end_dt = _parse_date(end) + timedelta(days=1)  # Incluye todo el día
        else:
            end_dt = start_dt + timedelta(hours=1)
    
    except ValueError as e:
        return jsonify({"error": f"Formato de fecha inválido: {e}"}), 400

    # Genera datos mock basados en la fecha (consistente con seed)
    seed_value = int(start_dt.strftime("%Y%m%d%H"))
    np.random.seed(seed_value)
    
    # Calcula número de puntos según agregación
    total_minutes = int((end_dt - start_dt).total_seconds() / 60)
    
    if agg == "5min":
        interval = 5
    elif agg == "15min":
        interval = 15
    elif agg == "hour":
        interval = 60
    elif agg == "day":
        interval = 1440
    else:
        interval = 5
    
    num_points = max(1, total_minutes // interval)
    
    points = []
    for i in range(min(num_points, 288)):  # Límite máximo de 288 puntos (1 día a 5min)
        t = start_dt + timedelta(minutes=interval * i)
        
        # Demanda realista basada en hora del día
        hour = t.hour
        base = 18000 + 4000 * np.sin((hour - 6) * np.pi / 12)  # Patrón diario
        noise = np.random.normal(0, 200)  # Ruido
        value = base + noise
        
        points.append({
            "ts": t.isoformat() + "Z",
            "value": round(float(value), 2)
        })
    
    # Si solo hay un punto (consulta de hora específica), devuelve info extra
    if len(points) == 1:
        return jsonify({
            "metric": metric,
            "agg": agg,
            "points": points,
            "single_point": True,
            "value": points[0]["value"]
        })
    
    return jsonify({
        "metric": metric,
        "agg": agg,
        "start": start_dt.isoformat() + "Z",
        "end": end_dt.isoformat() + "Z",
        "points": points,
        "count": len(points)
    })

# ----------------- Arranque -----------------
if __name__ == "__main__":
    # reintenta carga por si copiaste los ficheros después
    if model is None or scaler is None:
        print("[INFO] Reintentando carga de artefactos al inicio...")
        _m, _s = _load_artifacts()
        if _m is not None:
            model = _m
        if _s is not None:
            scaler = _s

    port = int(os.getenv("PORT", "5001"))
    app.run(host="0.0.0.0", port=port, debug=True)
