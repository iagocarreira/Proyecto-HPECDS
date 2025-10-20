import os, httpx
from typing import Optional

API_MAIN = os.getenv("API_MAIN_URL", "http://localhost:5001")
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "25"))

async def get_predictions(date: Optional[str] = None):
    url = f"{API_MAIN}/api/predictions" if date else f"{API_MAIN}/api/predictions_latest"
    params = {"date": date} if date else None
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.get(url, params=params); r.raise_for_status()
        return r.json()

async def get_kpis(date: Optional[str] = None):
    params = {"date": date} if date else None
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.get(f"{API_MAIN}/api/kpis", params=params); r.raise_for_status()
        return r.json()

async def get_anomalies(date: Optional[str] = None, z_threshold: float = 2.5):
    params = {"date": date, "z_threshold": z_threshold}
    async with httpx.AsyncClient(timeout=TIMEOUT) as c:
        r = await c.get(f"{API_MAIN}/api/anomalies", params=params); r.raise_for_status()
        return r.json()

OPENAI_TOOLS = [
  {"type":"function","function":{
    "name":"get_predictions",
    "description":"Predicción LSTM (5min). Usa 'date' (YYYY-MM-DD) o vacío para latest.",
    "parameters":{"type":"object","properties":{"date":{"type":"string"}}}
  }},
  {"type":"function","function":{
    "name":"get_kpis",
    "description":"KPIs (MAPE, RMSE) si hay solape real vs pred.",
    "parameters":{"type":"object","properties":{"date":{"type":"string"}}}
  }},
  {"type":"function","function":{
    "name":"get_anomalies",
    "description":"Anomalías por z-score del residuo.",
    "parameters":{"type":"object","properties":{
      "date":{"type":"string"},
      "z_threshold":{"type":"number","default":2.5}
    }}}
  },
]
