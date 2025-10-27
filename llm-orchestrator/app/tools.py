#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tools del chatbot (HPECDS) - Versión final con Groq
- Parser de fechas naturales completo
- Soporte para 2024 y 2025
- Manejo correcto de anomalías (campo "items")
"""

from __future__ import annotations
import os
import re
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import httpx

# --- Config ---
API_MAIN = os.getenv("API_MAIN_URL", "http://localhost:5001").rstrip("/")
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))

# --- Helper seguro ---
async def _safe_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """GET a api-main con manejo de errores."""
    url = f"{API_MAIN}{path}"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as c:
            r = await c.get(url, params=params)
            if r.status_code == 404:
                try:
                    return r.json()
                except:
                    return {"error": "No se encontraron datos"}
            r.raise_for_status()
            return r.json()
    except httpx.HTTPStatusError as e:
        try:
            return e.response.json()
        except:
            text = e.response.text[:300] if e.response.text else ""
            return {"error": f"Error {e.response.status_code}", "detail": text}
    except httpx.TimeoutException:
        return {"error": f"Timeout: la consulta tardó más de {TIMEOUT}s"}
    except Exception as e:
        return {"error": f"Error de red: {type(e).__name__}: {e}"}


def _normalize_date(date_str: Optional[str]) -> Optional[str]:
    """
    Normaliza fechas a YYYY-MM-DD o YYYY-MM-DDTHH:MM:SS.
    Soporta: hoy, ayer, mañana, "3 de diciembre de 2024", "10 y 10", "tres de la tarde"
    """
    if not date_str:
        return None
    
    date_lower = date_str.lower().strip()
    today = datetime.now()
    
    # Fechas simples
    if date_lower in ['hoy', 'today']:
        return today.strftime("%Y-%m-%d")
    elif date_lower in ['ayer', 'yesterday']:
        return (today - timedelta(days=1)).strftime("%Y-%m-%d")
    elif date_lower in ['mañana', 'tomorrow', 'manana']:
        return (today + timedelta(days=1)).strftime("%Y-%m-%d")
    
    # Diccionarios
    meses = {
        'enero': 1, 'febrero': 2, 'marzo': 3, 'abril': 4, 'mayo': 5, 'junio': 6,
        'julio': 7, 'agosto': 8, 'septiembre': 9, 'octubre': 10, 'noviembre': 11, 'diciembre': 12
    }
    
    horas_texto = {
        'cero': 0, 'una': 1, 'dos': 2, 'tres': 3, 'cuatro': 4, 'cinco': 5,
        'seis': 6, 'siete': 7, 'ocho': 8, 'nueve': 9, 'diez': 10,
        'once': 11, 'doce': 12, 'trece': 13, 'catorce': 14, 'quince': 15,
        'dieciséis': 16, 'dieciseis': 16, 'diecisiete': 17, 'dieciocho': 18,
        'diecinueve': 19, 'veinte': 20, 'veintiuna': 21, 'veintiuno': 21,
        'veintidós': 22, 'veintidos': 22, 'veintitrés': 23, 'veintitres': 23
    }
    
    # "3 de la tarde" = 15:00
    if 'de la tarde' in date_lower or 'de la mañana' in date_lower or 'de la noche' in date_lower:
        match = re.search(r'(\w+)\s+de\s+la\s+(tarde|mañana|noche)', date_lower)
        if match:
            hora_text = match.group(1)
            periodo = match.group(2)
            hora = horas_texto.get(hora_text)
            if hora is not None:
                if periodo == 'tarde' and hora < 12:
                    hora += 12
                elif periodo == 'noche' and hora < 12:
                    hora += 12
                return today.strftime(f"%Y-%m-%dT{hora:02d}:00:00")
    
    # Patrones con fecha completa + hora
    # "22 de octubre de 2024 a las 10:10" o "22 de octubre de 2024 a las 10 y 10"
    pattern1 = r'(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})\s+a\s+las\s+(\d{1,2})(?:\s*[y:]\s*(\d{2}))?'
    match1 = re.search(pattern1, date_lower)
    if match1:
        day, month_name, year = int(match1.group(1)), match1.group(2), int(match1.group(3))
        hour = int(match1.group(4))
        minute = int(match1.group(5)) if match1.group(5) else 0
        month = meses.get(month_name)
        if month:
            try:
                dt = datetime(year, month, day, hour, minute)
                return dt.strftime("%Y-%m-%dT%H:%M:%S")
            except ValueError:
                pass
    
    # "22 de octubre de 2024 a las diez y diez"
    pattern2 = r'(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})\s+a\s+las\s+(\w+)(?:\s+y\s+(\w+))?'
    match2 = re.search(pattern2, date_lower)
    if match2:
        day, month_name, year = int(match2.group(1)), match2.group(2), int(match2.group(3))
        hour_text, min_text = match2.group(4), match2.group(5) or 'cero'
        month = meses.get(month_name)
        hour = horas_texto.get(hour_text)
        minute = horas_texto.get(min_text, 0)
        if month and hour is not None:
            try:
                dt = datetime(year, month, day, hour, minute)
                return dt.strftime("%Y-%m-%dT%H:%M:%S")
            except ValueError:
                pass
    
    # "3 de diciembre de 2024"
    pattern3 = r'(\d{1,2})\s+de\s+(\w+)\s+de\s+(\d{4})'
    match3 = re.search(pattern3, date_lower)
    if match3:
        day, month_name, year = int(match3.group(1)), match3.group(2), int(match3.group(3))
        month = meses.get(month_name)
        if month:
            try:
                dt = datetime(year, month, day)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
    
    # "26 de octubre" (sin año - usa año actual o anterior si está en futuro)
    pattern4 = r'(\d{1,2})\s+de\s+(\w+)(?!\s+de\s+\d{4})'
    match4 = re.search(pattern4, date_lower)
    if match4:
        day, month_name = int(match4.group(1)), match4.group(2)
        month = meses.get(month_name)
        if month:
            try:
                dt = datetime(today.year, month, day)
                if dt > today:
                    dt = datetime(today.year - 1, month, day)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
    
    # Ya viene en formato válido
    return date_str


async def get_predictions(date: Optional[str] = None) -> Dict[str, Any]:
    """Predicción LSTM. Usa 'date' (YYYY-MM-DD) o vacío para latest."""
    date = _normalize_date(date)
    if date and "T" in date:
        date = date.split("T")[0]
    
    if date:
        result = await _safe_get("/api/predictions", params={"date": date})
    else:
        result = await _safe_get("/api/predictions_latest")
    
    if "error" in result:
        return result
    
    points = result.get("points", [])
    if len(points) > 20:
        values = [p.get("y", p.get("mw", 0)) for p in points]
        result["summary"] = {
            "total_points": len(points),
            "periodo": f"{points[0]['ts']} a {points[-1]['ts']}",
            "min_mw": round(min(values), 2),
            "max_mw": round(max(values), 2),
            "promedio_mw": round(sum(values) / len(values), 2)
        }
        result["points"] = points[:3] + points[-3:]
    
    return result


async def get_kpis(date: Optional[str] = None) -> Dict[str, Any]:
    """KPIs (MAPE, RMSE)."""
    date = _normalize_date(date)
    if date and "T" in date:
        date = date.split("T")[0]
    params = {"date": date} if date else None
    return await _safe_get("/api/kpis", params=params)


async def get_anomalies(date: Optional[str] = None, z_threshold: float = 2.5) -> Dict[str, Any]:
    """Anomalías por z-score. Campo de respuesta: 'items'."""
    date = _normalize_date(date)
    if date and "T" in date:
        date = date.split("T")[0]
    
    params: Dict[str, Any] = {"z_threshold": z_threshold}
    if date:
        params["date"] = date
    
    result = await _safe_get("/api/anomalies", params=params)
    
    if "error" in result:
        return result
    
    items = result.get("items", [])
    
    if not items:
        return {
            "total": 0,
            "items": [],
            "message": f"No se detectaron anomalías para {date or 'la fecha especificada'}"
        }
    
    # Resume si hay más de 5
    if len(items) > 5:
        z_scores = [abs(item["z"]) for item in items]
        sorted_items = sorted(items, key=lambda x: abs(x["z"]), reverse=True)
        
        result["stats"] = {
            "total": len(items),
            "max_z": round(max(z_scores), 2),
            "promedio_z": round(sum(z_scores) / len(z_scores), 2)
        }
        result["items"] = sorted_items[:3]
        result["message"] = f"Top 3 de {len(items)} anomalías"
    
    return result


async def query_series(metric: str, start: Optional[str] = None, end: Optional[str] = None, agg: Optional[str] = None) -> Dict[str, Any]:
    """Series históricas. metric='demanda', start obligatorio."""
    start = _normalize_date(start) if start else start
    end = _normalize_date(end) if end else end
    
    params: Dict[str, Any] = {"metric": metric}
    if start: params["start"] = start
    if end: params["end"] = end
    if agg: params["agg"] = agg
    
    result = await _safe_get("/api/series", params=params)
    
    if "error" in result:
        return result
    
    if result.get("single_point"):
        return result
    
    points = result.get("points", [])
    if len(points) > 20:
        values = [p["value"] for p in points if p["value"] is not None]
        if values:
            result["summary"] = {
                "total_points": len(points),
                "min_value": round(min(values), 2),
                "max_value": round(max(values), 2),
                "promedio": round(sum(values) / len(values), 2)
            }
        result["points"] = points[:3] + points[-3:]
    
    return result


# --- Declaración de tools ---
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_predictions",
            "description": "Obtiene predicciones LSTM de demanda. Si no hay fecha, devuelve últimas predicciones.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Fecha YYYY-MM-DD. Acepta: 'hoy', 'mañana', '3 de diciembre de 2024'"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_kpis",
            "description": "Métricas del modelo: MAPE y RMSE.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Fecha opcional YYYY-MM-DD"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_anomalies",
            "description": "Detecta anomalías comparando real vs predicho. Devuelve campo 'items' con: ts, real_mw, pred_mw, residual, z.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Fecha YYYY-MM-DD. Acepta: 'hoy', '18 de octubre de 2024'"},
                    "z_threshold": {"type": "number", "default": 2.5, "description": "Umbral z-score (2.5=moderadas, 3.0=graves)"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_series",
            "description": "Consulta datos HISTÓRICOS reales de demanda (2024/2025). Para fechas pasadas.",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {"type": "string", "enum": ["demanda"], "description": "Usa 'demanda'"},
                    "start": {"type": "string", "description": "Fecha inicio OBLIGATORIA. Formatos: 'YYYY-MM-DD', 'YYYY-MM-DDTHH:MM:SS', 'hoy', '3 de diciembre de 2024'"},
                    "end": {"type": "string", "description": "Fecha fin opcional"},
                    "agg": {"type": "string", "enum": ["5min", "15min", "hour", "day"], "description": "Agregación (default: 5min)"}
                },
                "required": ["metric", "start"]
            }
        }
    }
]

__all__ = ["get_predictions", "get_kpis", "get_anomalies", "query_series", "OPENAI_TOOLS"]
