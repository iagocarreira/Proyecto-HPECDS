#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from .llm import chat_with_tools

app = FastAPI(title="Chatbot LLM Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatIn(BaseModel):
    message: str

class ChatOut(BaseModel):
    answer: str
    used_tools: list | None = None

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!doctype html><meta charset="utf-8">
    <title>GreenEnergy Insights - Chatbot</title>
    <style>
      body{font-family:system-ui;margin:2rem;max-width:720px;background:#f5f5f5}
      h1{color:#2c3e50;text-align:center}
      .msg{padding:.6rem .8rem;border-radius:.6rem;margin:.4rem 0;max-width:80%}
      .you{background:#007bff;color:white;margin-left:auto;text-align:right}
      .bot{background:white;border:1px solid #ddd}
      .tools{background:#fff3cd;padding:.4rem;margin:.2rem 0;font-size:.85em;border-radius:.3rem}
      #log{border:1px solid #ccc;padding:1rem;height:420px;overflow-y:auto;background:white;border-radius:.5rem}
      #input-area{margin-top:1rem;display:flex;gap:.5rem}
      input{flex:1;padding:.7rem;border:1px solid #ccc;border-radius:.3rem}
      button{padding:.7rem 1.5rem;background:#007bff;color:white;border:none;border-radius:.3rem;cursor:pointer}
      button:hover{background:#0056b3}
    </style>
    <h1>🌱 GreenEnergy Insights</h1>
    <p style="text-align:center;color:#666">Chatbot de Análisis de Demanda Eléctrica</p>
    <div id="log"></div>
    <div id="input-area">
      <input id="t" placeholder="Escribe tu consulta..." autofocus />
      <button onclick="send()">Enviar</button>
    </div>
    <script>
      const API="/chat", log=document.getElementById('log'), t=document.getElementById('t');
      
      // Mensaje de bienvenida inicial
      window.onload = () => {
        fetch(API, {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body:JSON.stringify({message:'hola'})
        }).then(r=>r.json()).then(j=>{
          add('bot', j.answer || 'Error');
        });
      };
      
      function add(w,txt){
        const d=document.createElement('div');
        d.className='msg '+w; d.textContent=txt;
        log.appendChild(d); log.scrollTop=log.scrollHeight;
      }
      
      async function send(){
        const msg=t.value.trim(); if(!msg) return;
        add('you', msg); t.value='';
        try{
          const r=await fetch(API,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg})});
          const j=await r.json();
          add('bot', j.answer || (j.detail ? ('Error: '+j.detail) : 'Error'));
          if (j.used_tools && j.used_tools.length){ 
            const toolDiv=document.createElement('div');
            toolDiv.className='tools';
            toolDiv.textContent='🔧 Herramientas: ' + j.used_tools.join(', ');
            log.appendChild(toolDiv);
            log.scrollTop=log.scrollHeight;
          }
        }catch(e){ add('bot','Error al conectar con el servidor'); }
      }
      
      t.addEventListener('keydown', e=>{ if(e.key==='Enter') send(); });
    </script>
    """

@app.get("/favicon.ico")
def favicon():
    return PlainTextResponse("", status_code=204)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatOut)
async def chat(body: ChatIn):
    t0 = time.time()
    try:
        result = await chat_with_tools(body.message)
        answer = result["answer"]
        used_tools = result["used_tools"]
        print(f"[CHAT] q={body.message!r} tools={used_tools} dt={time.time()-t0:.2f}s")
        return ChatOut(answer=answer, used_tools=used_tools)
    except Exception as e:
        import traceback
        print("ERROR EN CHATBOT:\n", traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
