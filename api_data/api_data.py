import os
from datetime import datetime, time
from typing import Optional, List
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, MetaData, Table, select, and_, or_, func
from sqlalchemy.engine import Engine

# ---------- Conexión a Azure SQL (pyodbc) ----------
SERVER   = os.getenv("SQL_SERVER",   "udcserver2025.database.windows.net")
DATABASE = os.getenv("SQL_DB",       "grupo_1")
USER     = os.getenv("SQL_USER",     "ugrupo1")
PASSWORD = os.getenv("SQL_PASSWORD", "HK9WXIJaBp2Q97haePdY")

ENGINE_URL = (
    f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER}:1433/{DATABASE}"
    f"?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no"
)
engine: Engine = create_engine(ENGINE_URL, pool_pre_ping=True, future=True)

# ---------- Config ----------
TABLE_NAME = os.getenv("TABLE_NAME", "demanda_peninsula")
PAGE_MAX = 1000

meta = MetaData()
tabla = Table(TABLE_NAME, meta, autoload_with=engine, schema="dbo")

# ---------- Modelo solo para /records/{ts} ----------
class Registro(BaseModel):
    fecha: datetime
    fecha_utc: datetime
    tz_time: datetime
    valor_real: Optional[int] = None
    geo_id_real: Optional[int] = None
    geo_name_real: Optional[str] = None
    valor_previsto: Optional[int] = None
    geo_id_previsto: Optional[int] = None
    geo_name_previsto: Optional[str] = None
    valor_programado: Optional[int] = None
    geo_id_programado: Optional[int] = None
    geo_name_programado: Optional[str] = None
    class Config:
        from_attributes = False

app = FastAPI(
    title="API Read-Only de Demanda Eléctrica",
    version="1.0.0",
    description="API de consulta de demanda eléctrica"
)

# ---------- UI estática ----------
STATIC_DIR = Path(__file__).parent / "web"
app.mount("/ui", StaticFiles(directory=str(STATIC_DIR), html=True), name="web")

@app.get("/")
def root():
    return RedirectResponse(url="/ui/")

# ---------- Utils ----------
def _parse_dt(s: Optional[str], is_hasta: bool = False) -> Optional[datetime]:
    if not s:
        return None
    s = s.strip()
    # Solo fecha YYYY-MM-DD
    if len(s) == 10 and s[4] == "-" and s[7] == "-":
        d = datetime.strptime(s, "%Y-%m-%d").date()
        return datetime.combine(d, time(23, 59, 59)) if is_hasta else datetime.combine(d, time(0, 0, 0))
    # YYYY-MM-DDTHH:MM (añadimos :00)
    if len(s) == 16 and "T" in s:
        s = s + ":00"
    # YYYY-MM-DD HH:MM -> normalizamos a T
    s = s.replace(" ", "T")
    try:
        return datetime.fromisoformat(s)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Fecha inválida: {s}")

# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"ok": True}

# /records con filtros por fecha, geo_name y selección de columnas
@app.get("/records")
def list_records(
    desde: Optional[str] = Query(None, description="ISO o YYYY-MM-DD"),
    hasta: Optional[str] = Query(None, description="ISO o YYYY-MM-DD"),
    geo_name: Optional[str] = Query(None, description="match parcial en geo_name_* (sin may/min)"),
    fields: Optional[str] = Query(None, description="columnas CSV a devolver"),
    limit: int = Query(100, ge=1, le=PAGE_MAX),
    offset: int = Query(0, ge=0),
    order: str = Query("fecha", description="fecha|fecha_utc|tz_time")
):
    desde_dt = _parse_dt(desde, is_hasta=False)
    hasta_dt = _parse_dt(hasta, is_hasta=True)

    conds = []
    if "fecha" in tabla.c and desde_dt:
        conds.append(tabla.c.fecha >= desde_dt)
    if "fecha" in tabla.c and hasta_dt:
        conds.append(tabla.c.fecha <= hasta_dt)

    if geo_name:
        term = f"%{geo_name.strip().lower()}%"
        name_cols = [c for n, c in tabla.c.items()
                     if n in ("geo_name_real", "geo_name_previsto", "geo_name_programado")]
        if name_cols:
            conds.append(or_(*[func.lower(c).like(term) for c in name_cols]))

    # Selección de columnas
    selected_cols = list(tabla.c)
    if fields:
        req = [f.strip() for f in fields.split(",") if f.strip()]
        valid = set(tabla.c.keys())
        unknown = [f for f in req if f not in valid]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Columnas desconocidas: {', '.join(unknown)}")
        selected_cols = [tabla.c[f] for f in req]

    order_col = tabla.c.get(order, tabla.c.fecha)

    with engine.connect() as conn:
        base = select(*selected_cols).select_from(tabla)
        if conds:
            base = base.where(and_(*conds))
        rows = conn.execute(
            base.order_by(order_col).limit(limit).offset(offset)
        ).mappings().all()

    return [dict(r) for r in rows]

# Detalle por timestamp exacto (sigue usando el modelo)
@app.get("/records/{ts}", response_model=List[Registro] if False else Registro)
def get_by_fecha(ts: datetime):
    with engine.connect() as conn:
        row = conn.execute(
            select(tabla).where(tabla.c.fecha == ts).limit(1)
        ).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="No encontrado")
    return Registro(**row)
