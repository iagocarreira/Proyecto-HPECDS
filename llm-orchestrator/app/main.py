#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from app.llm import chat_with_tools

# Cargar variables de entorno (p. ej., OPENAI_API_KEY)
load_dotenv()


# --- Configuración de FastAPI ---
app = FastAPI(title="Chatbot LLM Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Modelos Pydantic ---
class ChatIn(BaseModel):
    message: str

class ChatOut(BaseModel):
    answer: str
    used_tools: list | None = None

# --- Rutas de la API ---

@app.get("/", response_class=HTMLResponse)
async def home():
    """
    Sirve el archivo HTML estático para la interfaz del chatbot.
    Busca index.html relativo a este archivo (no al current working dir).
    """
    base_dir = Path(__file__).resolve().parent
    html_file = base_dir / "index.html"
    print(f"Sirviendo interfaz de chatbot desde: {html_file}")
    if not html_file.exists():
        # Devuelve detalle para facilitar debugging en logs (no el HTML en producción)
        raise HTTPException(status_code=500, detail="index.html no encontrado en el mismo directorio que main.py")
    return FileResponse(html_file, media_type="text/html")

@app.get("/favicon.ico")
def favicon():
    return PlainTextResponse("", status_code=204)

# Health endpoints (añadimos /health para compatibilidad con scripts de tests)
@app.get("/health")
def health():
    # Rellena "database" con lo que corresponda en tu entorno; aquí un valor por defecto
    return {"status": "ok", "database": os.getenv("DB_NAME", "unknown")}

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatOut)
async def chat(body: ChatIn):
    """
    Endpoint principal del chatbot. Procesa el mensaje del usuario y 
    llama al orquestador LLM.
    """
    t0 = time.time()
    try:
        # Llama a la lógica del chatbot
        result = await chat_with_tools(body.message)
        answer = result["answer"]
        used_tools = result["used_tools"]
        
        # Log del terminal
        print(f"[CHAT] q={body.message!r} tools={used_tools} dt={time.time()-t0:.2f}s")
        
        return ChatOut(answer=answer, used_tools=used_tools)
        
    except Exception as e:
        import traceback
        print("ERROR EN CHATBOT:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
