#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tools del chatbot (HPECDS) - Versión extendida
- query_series (existente)
- model_performance (nuevo): devuelve métricas de rendimiento del modelo
- query_predictions (nuevo): devuelve predicciones guardadas en la base de datos

Requiere que el servicio "api-main" exponga:
- GET /api/model/performance[?model=name]
- GET /api/predictions?model=...&start=...&end=...&limit=...
- GET /api/series?metric=...&start=...&end=...&agg=...
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import httpx

# --- CONFIG: usar puerto 5001 por defecto (api_main.py arranca ahí) ---
API_MAIN = os.getenv("API_MAIN_URL", "http://localhost:5001").rstrip("/")
TIMEOUT = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))

# --- Helper de requests ---
async def _safe_get(path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """GET a api-main con manejo robusto de errores y logging para debug."""
    url = f"{API_MAIN}{path}"
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(url, params=params)
            # Log de debug: URL, params, status
            try:
                print(f"[tools._safe_get] GET {response.url} status={response.status_code}")
                print(f"[tools._safe_get] response.text: {response.text[:2000]}")
            except Exception:
                pass
            response.raise_for_status()
            return response.json()
    except httpx.HTTPStatusError as e:
        try:
            error_data = e.response.json()
            print(f"[tools._safe_get] HTTPStatusError JSON: {error_data}")
            return {"error": error_data.get("error", f"HTTP {e.response.status_code}")}
        except Exception:
            print(f"[tools._safe_get] HTTPStatusError text: {e.response.text}")
            return {"error": f"Error HTTP {e.response.status_code}"}
    except httpx.TimeoutException:
        print(f"[tools._safe_get] Timeout contacting {url}")
        return {"error": f"Timeout: la consulta tardó más de {TIMEOUT}s"}
    except Exception as e:
        print(f"[tools._safe_get] Error de conexión: {type(e).__name__} - {e}")
        return {"error": f"Error de conexión: {type(e).__name__}"}


# ============================================================================
# 1) query_series: consulta series temporales históricas (demanda)
# ============================================================================
async def query_series(
    metric: str = "demanda",
    start: Optional[str] = None,
    end: Optional[str] = None,
    agg: Optional[str] = None
) -> Dict[str, Any]:
    """
    Consulta datos de demanda eléctrica histórica.
    
    Args:
        metric: Métrica a consultar (por defecto 'demanda')
        start: Fecha de inicio en formato YYYY-MM-DD o ISO (OBLIGATORIO)
        end: Fecha de fin en formato YYYY-MM-DD o ISO (opcional)
        agg: Agregación temporal (e.g., 'hourly', 'daily')
        
    Returns:
        Dict con 'data' (lista de registros) o 'error'
    """
    if not start:
        return {"error": "El parámetro 'start' es obligatorio"}
    
    params = {"metric": metric}
    params["start"] = start
    
    if end:
        params["end"] = end
    if agg:
        params["agg"] = agg

    result = await _safe_get("/api/series", params=params)
    return result


# ============================================================================
# 2) model_performance: métricas de rendimiento del modelo
# ============================================================================
async def model_performance(
    model: Optional[str] = None
) -> Dict[str, Any]:
    """
    Obtiene métricas de rendimiento del modelo (MAE, RMSE, etc.).
    
    Args:
        model: Nombre del modelo (opcional). Si no se especifica,
               retorna métricas de todos los modelos disponibles.
                   
    Returns:
        Dict con métricas del modelo o 'error'
    """
    params = {}
    if model:
        params["model"] = model
    
    result = await _safe_get("/api/model/performance", params=params)
    return result


# ============================================================================
# 3) query_predictions: predicciones guardadas en la base de datos
# ============================================================================
async def query_predictions(
    model: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Consulta predicciones del modelo guardadas en la base de datos.
    
    Args:
        model: Nombre del modelo (e.g., 'modelo-A', 'modelo-B')
        start: Fecha de inicio en formato YYYY-MM-DD
        end: Fecha de fin en formato YYYY-MM-DD
        limit: Número máximo de predicciones a retornar
        
    Returns:
        Dict con 'predictions' (lista de predicciones) o 'error'
    """
    params = {}
    if model:
        params["model"] = model
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    if limit:
        params["limit"] = limit

    result = await _safe_get("/api/predictions", params=params)
    return result


# ============================================================================
# OPENAI_TOOLS: definición de las herramientas para OpenAI
# ============================================================================
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_series",
            "description": "Consulta datos históricos de demanda eléctrica en España (años 2024-2025)",
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "description": "Métrica a consultar. SIEMPRE debe ser 'demanda'",
                        "default": "demanda"
                    },
                    "start": {
                        "type": "string",
                        "description": "Fecha/hora de inicio en formato ISO (YYYY-MM-DD o YYYY-MM-DDTHH:MM:SS). OBLIGATORIO"
                    },
                    "end": {
                        "type": "string",
                        "description": "Fecha/hora de fin en formato ISO (YYYY-MM-DD o YYYY-MM-DDTHH:MM:SS). Opcional"
                    },
                    "agg": {
                        "type": "string",
                        "description": "Agregación temporal: 'hourly', 'daily', etc. Opcional"
                    }
                },
                "required": ["start"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "model_performance",
            "description": "Obtiene métricas de rendimiento del modelo de predicción (MAE, RMSE, R², etc.)",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Nombre del modelo (e.g., 'modelo-A', 'modelo-B'). Si no se especifica, retorna todas las métricas disponibles"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_predictions",
            "description": "Consulta predicciones del modelo guardadas en la base de datos",
            "parameters": {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "description": "Nombre del modelo de predicción (e.g., 'modelo-A', 'modelo-B')"
                    },
                    "start": {
                        "type": "string",
                        "description": "Fecha de inicio en formato YYYY-MM-DD"
                    },
                    "end": {
                        "type": "string",
                        "description": "Fecha de fin en formato YYYY-MM-DD"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Número máximo de predicciones a retornar",
                        "default": 100
                    }
                },
                "required": []
            }
        }
    }
]
