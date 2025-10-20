import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .llm import chat_with_tools

class ChatReq(BaseModel):
    message: str
    session_id: str | None = None

app = FastAPI(title="LLM Orchestrator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

@app.get("/healthz")
async def healthz(): return {"status":"ok"}

@app.post("/chat")
async def chat(req: ChatReq):
    return await chat_with_tools(req.message)
