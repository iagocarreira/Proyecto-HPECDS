#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo LLM - Orquestador del chatbot con OpenAI
Versión mejorada: inicialización perezosa del cliente OpenAI para evitar fallos
en importación cuando no se ha configurado OPENAI_API_KEY.
Soporta consultas de demanda histórica, métricas de modelos y predicciones.
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime

# Importar cliente OpenAI de forma normal; la instanciación será perezosa.
from openai import OpenAI

# Importar herramientas
from .tools import OPENAI_TOOLS, query_series, model_performance, query_predictions

# --- Configuración ---
# No instanciar el cliente en el momento del import para evitar excepciones
# que rompan el arranque del servidor si falta la variable de entorno.
_client: Optional[OpenAI] = None


def _init_client() -> Optional[OpenAI]:
    """Crea e inicializa el cliente OpenAI si hay API key disponible.
    Devuelve None si no está configurado (se maneja en tiempo de ejecución)."""
    global _client
    if _client is not None:
        return _client

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")
    if not api_key:
        print("⚠️ OPENAI_API_KEY no está configurada. El servicio LLM no podrá usarse.")
        _client = None
        return None

    try:
        _client = OpenAI(api_key=api_key, base_url=base_url)
        print("✓ Cliente OpenAI inicializado correctamente")
        return _client
    except Exception as e:
        print(f"✗ Error inicializando cliente OpenAI: {e}")
        _client = None
        return None


MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

# --- Mensaje de bienvenida ---
WELCOME_MESSAGE = """¡Hola! 👋 Soy el asistente de **GreenEnergy Insights**.

Puedo ayudarte con:

📊 **Consultas de demanda histórica** (años 2024-2025)
- "¿Cuál fue la demanda hoy a las 15:00?"
- "Muéstrame la demanda del 3 de diciembre de 2024"
- "Dame la demanda entre el 20 y 25 de octubre"

📈 **Rendimiento de modelos**
- "¿Cuál es el rendimiento del modelo-A?"
- "Muéstrame las métricas de todos los modelos"

🔮 **Predicciones almacenadas**
- "Dame las predicciones del modelo-A para hoy"
- "Muestra predicciones entre el 20 y 25 de octubre"

¿Qué te gustaría consultar?"""


# --- System Prompt ---
def get_system_prompt() -> str:
    """Genera el system prompt con fecha actual."""
    today = datetime.now()

    meses_es = [
        'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
        'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre'
    ]
    mes_es = meses_es[today.month - 1]

    return f"""Eres el asistente de **GreenEnergy Insights**, un portal de análisis de demanda eléctrica.

**FECHA ACTUAL:** {today.strftime('%Y-%m-%d')} ({today.day} de {mes_es} de {today.year})

**TU TAREA:**
Ayudar a consultar datos históricos REALES de demanda eléctrica en España (años 2024 y 2025).

**HERRAMIENTAS DISPONIBLES:**
- `query_series`: Consulta demanda histórica
  - Parámetros: metric="demanda", start (obligatorio), end (opcional), agg
  - El parámetro `start` es OBLIGATORIO
  - Acepta fechas en formato ISO o texto natural
- `model_performance`: Métricas de rendimiento de modelos
  - Parámetros: model (opcional)
- `query_predictions`: Predicciones almacenadas
  - Parámetros: model (opcional), start, end, limit

**REGLAS CRÍTICAS:**
1. SIEMPRE usar `query_series` para consultas de demanda histórica
2. El parámetro `metric` SIEMPRE debe ser "demanda" para `query_series`
3. El parámetro `start` es OBLIGATORIO para `query_series`
4. Si el usuario no especifica año, asumir 2024
5. Presentar resultados con unidades (MW)
6. Si hay muchos puntos, mencionar el resumen (min, max, promedio)
7. Ser conciso pero informativo

**EJEMPLOS:**

Usuario: "demanda del 3 de diciembre de 2024"
→ query_series({{"metric": "demanda", "start": "2024-12-03"}})

Usuario: "demanda de hoy"
→ query_series({{"metric": "demanda", "start": "{today.strftime('%Y-%m-%d')}"}})

Usuario: "demanda hoy a las 15:00"
→ query_series({{"metric": "demanda", "start": "{today.strftime('%Y-%m-%d')}T15:00:00"}})

Usuario: "demanda entre el 20 y 25 de octubre"
→ query_series({{"metric": "demanda", "start": "2024-10-20", "end": "2024-10-25"}})

Usuario: "¿Cuál es el rendimiento del modelo-A?"
→ model_performance({{"model": "modelo-A"}})

Usuario: "Dame las predicciones del modelo-A para hoy"
→ query_predictions({{"model": "modelo-A", "start": "{today.strftime('%Y-%m-%d')}"}})

**FORMATO DE RESPUESTA:**
- Para un punto: "La demanda el [fecha] a las [hora] fue de [valor] MW"
- Para múltiples puntos: Mencionar periodo, y dar estadísticas (mín, máx, promedio)
- Para rendimiento de modelo: "Modelo X — MAE: Y, RMSE: Z, R2: W (evaluado en YYYY-MM-DD)"
- Para predicciones: resumir (count, min, max, mean) y mostrar ejemplos si procede
- Si hay error: Explicar claramente qué salió mal

Responde siempre en español, de forma natural y profesional."""


# --- Ejecución de herramientas ---
async def _call_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Ejecuta una herramienta según su nombre."""
    tools_map = {
        "query_series": query_series,
        "model_performance": model_performance,
        "query_predictions": query_predictions
    }

    if name not in tools_map:
        return {"error": f"Herramienta '{name}' no reconocida"}

    try:
        return await tools_map[name](**args)
    except Exception as e:
        return {"error": f"Error al ejecutar {name}: {str(e)}"}


# --- Función principal del chatbot ---
async def chat_with_tools(user_msg: str) -> Dict[str, Any]:
    """
    Procesa un mensaje del usuario y devuelve una respuesta.

    Args:
        user_msg: Mensaje del usuario

    Returns:
        Dict con:
        - answer: Respuesta del asistente
        - used_tools: Lista de herramientas usadas
    """
    # Detectar saludos
    saludos = ['hola', 'hello', 'hi', 'buenos días', 'buenas tardes', 'buenas noches', 'hey']
    if user_msg.lower().strip() in saludos:
        return {"answer": WELCOME_MESSAGE, "used_tools": []}

    # Intentar inicializar el cliente OpenAI (si no está configurado devuelve None)
    client = _init_client()
    if client is None:
        # No interrumpir el servidor; devolver mensaje informativo para el usuario.
        return {
            "answer": (
                "⚠️ El servicio de LLM no está configurado en este entorno. "
                "Falta la variable de entorno OPENAI_API_KEY o hubo un error al inicializar el cliente. "
                "Por favor configura OPENAI_API_KEY para habilitar respuestas generadas por el modelo."
            ),
            "used_tools": []
        }

    # Construir mensajes
    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": user_msg}
    ]

    # Primera llamada al LLM
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=OPENAI_TOOLS,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=2000
        )
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Error en primera llamada al LLM: {error_msg}")

        # Reintentar sin herramientas si hay error de function calling
        if "tool_use_failed" in error_msg or "Failed to call" in error_msg:
            try:
                fallback = client.chat.completions.create(
                    model=MODEL,
                    messages=messages + [{
                        "role": "system",
                        "content": "No uses herramientas. Responde directamente."
                    }],
                    temperature=0.2
                )
                answer = fallback.choices[0].message.content or "No pude procesar la consulta."
                return {
                    "answer": f"{answer}\n\n⚠️ Nota: Hubo un problema al acceder a los datos.",
                    "used_tools": []
                }
            except Exception as e2:
                return {
                    "answer": f"Error: {str(e2)}",
                    "used_tools": []
                }

        return {
            "answer": f"Error al comunicarse con el modelo: {error_msg}",
            "used_tools": []
        }

    # Obtener mensaje del asistente
    assistant_msg = response.choices[0].message
    used_tools: List[str] = []

    # Si no hay llamadas a herramientas, devolver respuesta directa
    if not getattr(assistant_msg, "tool_calls", None):
        answer = assistant_msg.content or "No pude generar una respuesta."
        return {"answer": answer, "used_tools": used_tools}

    # Ejecutar herramientas
    print(f"✓ Ejecutando {len(assistant_msg.tool_calls)} herramienta(s)...")

    tasks = []
    tool_metadata = []

    for tool_call in assistant_msg.tool_calls:
        tool_name = tool_call.function.name
        used_tools.append(tool_name)

        # Parsear argumentos
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            print(f"✗ JSON inválido en {tool_name}: {tool_call.function.arguments}")
            args = {}

        print(f"  → {tool_name}({args})")

        # Crear tarea
        tasks.append(_call_tool(tool_name, args))
        tool_metadata.append({
            "id": tool_call.id,
            "name": tool_name,
            "args": tool_call.function.arguments
        })

    # Ejecutar todas las herramientas en paralelo
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Añadir mensaje del asistente con las llamadas a herramientas
    messages.append({
        "role": "assistant",
        "tool_calls": [
            {
                "id": meta["id"],
                "type": "function",
                "function": {
                    "name": meta["name"],
                    "arguments": meta["args"]
                }
            }
            for meta in tool_metadata
        ]
    })

    # Añadir resultados de herramientas
    for tool_call, result in zip(assistant_msg.tool_calls, results):
        if isinstance(result, Exception):
            content = json.dumps({
                "error": f"{type(result).__name__}: {str(result)}"
            })
            print(f"  ✗ {tool_call.function.name} falló: {result}")
        else:
            content = json.dumps(result, ensure_ascii=False)
            print(f"  ✓ {tool_call.function.name} completado")

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "content": content
        })

    # Segunda llamada al LLM para generar respuesta final
    try:
        final_response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=1500
        )

        final_answer = final_response.choices[0].message.content or "No pude generar una respuesta."

        return {
            "answer": final_answer,
            "used_tools": used_tools
        }

    except Exception as e:
        print(f"✗ Error en segunda llamada al LLM: {e}")
        return {
            "answer": f"Error al procesar la respuesta: {str(e)}",
            "used_tools": used_tools
        }


# --- Exports ---
__all__ = ["chat_with_tools", "WELCOME_MESSAGE"]
