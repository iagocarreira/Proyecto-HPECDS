import os, json, asyncio
from typing import Dict, Any, List
from openai import OpenAI
from .tools import OPENAI_TOOLS, get_predictions, get_anomalies, get_kpis, query_series

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL","https://api.openai.com/v1"))
MODEL = os.getenv("LLM_MODEL","gpt-4o-mini")

SYSTEM = ("Eres un asistente para operaciones energéticas. Responde en español, claro y conciso. "
          "Cuando pidan predicciones/anomalías/KPIs/series, usa herramientas. Menciona siempre fechas ISO.")

async def _call_tool(name:str, args:Dict[str,Any]):
    return await {"get_predictions":get_predictions, "get_anomalies":get_anomalies,
                  "get_kpis":get_kpis, "query_series":query_series}[name](**args)

async def chat_with_tools(user_msg:str) -> Dict[str,Any]:
    messages = [{"role":"system","content":SYSTEM},{"role":"user","content":user_msg}]
    first = client.chat.completions.create(model=MODEL, messages=messages, tools=OPENAI_TOOLS,
                                           tool_choice="auto", temperature=0.2)
    msg = first.choices[0].message
    used: List[str] = []

    if msg.tool_calls:
        tasks, tool_msgs = [], []
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            used.append(tc.function.name)
            tasks.append(_call_tool(tc.function.name, args))
            tool_msgs.append({"id": tc.id, "name": tc.function.name, "args": tc.function.arguments})
        results = await asyncio.gather(*tasks)

        messages.append({"role":"assistant","tool_calls":[{"id":m["id"],"type":"function",
                       "function":{"name":m["name"],"arguments":m["args"]}} for m in tool_msgs]})
        for tc, res in zip(msg.tool_calls, results):
            messages.append({"role":"tool","tool_call_id":tc.id,"name":tc.function.name,"content":json.dumps(res)})

        final = client.chat.completions.create(model=MODEL, messages=messages, temperature=0.2)
        return {"answer": final.choices[0].message.content, "used_tools": used}

    return {"answer": msg.content or "(sin contenido)", "used_tools": used}
