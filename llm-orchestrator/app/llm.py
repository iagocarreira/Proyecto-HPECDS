#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, asyncio
from typing import Dict, Any, List
from datetime import datetime
from openai import OpenAI
from .tools import OPENAI_TOOLS, get_predictions, get_anomalies, get_kpis, query_series

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
)
MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")  # Mejor para function calling

WELCOME_MESSAGE = """¡Hola! Soy el asistente de GreenEnergy Insights, el portal inteligente de análisis de demanda eléctrica del proyecto HPECDS.

¿Qué puedo hacer por ti?
- Consultar demanda eléctrica histórica (ej: "¿Cuál fue la demanda el 3 de diciembre de 2024?")
- Obtener predicciones futuras del modelo LSTM
- Detectar anomalías en la demanda
- Mostrarte métricas de calidad del modelo (MAPE, RMSE)

Ejemplos de consultas:
- "¿Cuál fue la demanda hoy a las 3 de la tarde?"
- "Dame predicciones para mañana"
- "¿Hubo anomalías el 26 de octubre?"
- "¿Qué tal funciona el modelo?"

Pregúntame lo que necesites."""

def get_system_prompt() -> str:
    """Genera el system prompt con fecha actual dinámica."""
    today = datetime.now()
    meses_es = ['enero','febrero','marzo','abril','mayo','junio','julio','agosto','septiembre','octubre','noviembre','diciembre']
    mes_es = meses_es[today.month - 1]
    
    return f"""Eres el asistente del proyecto HPECDS (GreenEnergy Insights) especializado en demanda eléctrica.

FECHA ACTUAL: Hoy es {today.strftime('%Y-%m-%d')} ({today.day} de {mes_es} de {today.year})

HERRAMIENTAS DISPONIBLES:
1. query_series: Consulta datos históricos REALES (2024/2025)
   - OBLIGATORIO: metric="demanda", start="YYYY-MM-DD"
   - Ejemplo: {{"metric":"demanda","start":"2024-12-03"}}
   
2. get_predictions: Predicciones futuras del modelo LSTM
   - Parámetro opcional: date="YYYY-MM-DD"
   
3. get_anomalies: Detecta anomalías (campo "items")
   - Cada item tiene: ts, real_mw, pred_mw, residual, z
   
4. get_kpis: Métricas MAPE y RMSE

REGLAS CRÍTICAS:
- Demanda PASADA → query_series
- Demanda FUTURA → get_predictions
- Años disponibles: 2024 y 2025
- SIEMPRE usar formato JSON limpio en los argumentos
- Al reportar anomalías: menciona hora, real_mw, pred_mw, residual
- Respuestas en español, concisas, con unidades (MW)

EJEMPLOS:
Usuario: "demanda del 3 de diciembre de 2024"
→ query_series({{"metric":"demanda","start":"2024-12-03"}})

Usuario: "demanda de hoy"
→ query_series({{"metric":"demanda","start":"{today.strftime('%Y-%m-%d')}"}})

Usuario: "anomalías del 18 de octubre"
→ get_anomalies({{"date":"2024-10-18"}}) [si contexto indica 2024]
"""

async def _call_tool(name: str, args: Dict[str, Any]):
    tools_map = {
        "get_predictions": get_predictions,
        "get_anomalies": get_anomalies,
        "get_kpis": get_kpis,
        "query_series": query_series
    }
    return await tools_map[name](**args)

async def chat_with_tools(user_msg: str) -> Dict[str, Any]:
    # Detectar saludos
    if user_msg.lower().strip() in ['hola', 'hello', 'hi', 'buenos días', 'buenas tardes', 'buenas noches', 'hey']:
        return {"answer": WELCOME_MESSAGE, "used_tools": []}
    
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": user_msg}
    ]
    
    # Primera llamada con manejo de errores robusto
    try:
        first = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=2000
        )
    except Exception as e:
        error_str = str(e)
        
        # Si hay error de function calling, reintentar sin herramientas
        if any(x in error_str for x in ["tool_use_failed", "Failed to call a function", "failed_generation"]):
            print(f"[WARN] Error en function calling: {error_str[:200]}")
            print(f"[INFO] Reintentando sin herramientas...")
            
            try:
                fallback = client.chat.completions.create(
                    model=MODEL,
                    messages=messages + [{
                        "role": "system",
                        "content": "IMPORTANTE: No uses herramientas. Responde directamente basándote en tu conocimiento. Si necesitas datos específicos, indica que no tienes acceso a ellos en este momento."
                    }],
                    temperature=0.2
                )
                answer = fallback.choices[0].message.content or "No pude procesar la consulta."
                return {
                    "answer": f"{answer}\n\n⚠️ Nota: No pude acceder a las herramientas de consulta. Considera cambiar el modelo a 'llama-3.3-70b-versatile' en tu .env",
                    "used_tools": []
                }
            except Exception as e2:
                return {
                    "answer": f"Error al procesar la consulta. Intenta cambiar LLM_MODEL a 'llama-3.3-70b-versatile' en tu .env\n\nError: {e2}",
                    "used_tools": []
                }
        
        return {"answer": f"Error: {e}", "used_tools": []}
    
    msg = first.choices[0].message
    used: List[str] = []

    if not msg.tool_calls:
        return {"answer": msg.content or "(sin contenido)", "used_tools": used}

    # Ejecutar herramientas
    tasks, tool_msgs = [], []
    for tc in msg.tool_calls:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError as e:
            print(f"[ERROR] JSON inválido en {tc.function.name}: {tc.function.arguments}")
            args = {}
        
        used.append(tc.function.name)
        tasks.append(_call_tool(tc.function.name, args))
        tool_msgs.append({
            "id": tc.id,
            "name": tc.function.name,
            "args": tc.function.arguments
        })
    
    results = await asyncio.gather(*tasks, return_exceptions=True)

    messages.append({
        "role": "assistant",
        "tool_calls": [
            {
                "id": m["id"],
                "type": "function",
                "function": {
                    "name": m["name"],
                    "arguments": m["args"]
                }
            }
            for m in tool_msgs
        ]
    })
    
    for tc, res in zip(msg.tool_calls, results):
        if isinstance(res, Exception):
            content = json.dumps({"error": f"{type(res).__name__}: {str(res)}"})
        else:
            content = json.dumps(res, ensure_ascii=False)
        
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "name": tc.function.name,
            "content": content
        })

    # Segunda llamada
    try:
        final = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2
        )
        return {
            "answer": final.choices[0].message.content or "No pude generar una respuesta.",
            "used_tools": used
        }
    except Exception as e:
        return {
            "answer": f"Error al procesar la respuesta: {e}",
            "used_tools": used
        }
