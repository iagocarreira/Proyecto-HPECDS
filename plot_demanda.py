# plot_desde_bd.py
# Uso:
#   python plot_desde_bd.py            # últimos 7 días
#   python plot_desde_bd.py 3          # últimos 3 días

import sys
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine, text



SERVER   = "udcserver2025.database.windows.net"
DATABASE = "grupo_1"
USER     = "ugrupo1"
PASSWORD = "HK9WXIJaBp2Q97haePdY"


ENGINE_URL = (
    f"mssql+pyodbc://{USER}:{PASSWORD}@{SERVER}:1433/{DATABASE}"
    "?driver=ODBC+Driver+18+for+SQL+Server&Encrypt=yes&TrustServerCertificate=no"
)
engine = create_engine(ENGINE_URL, pool_pre_ping=True)


dias = int(sys.argv[1]) if len(sys.argv) == 2 else 7
tabla = "dbo.Demanda"


with engine.connect() as con:
    cols = pd.read_sql(
        text("""
        SELECT COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_NAME = :tname AND TABLE_SCHEMA = :tschema
        ORDER BY ORDINAL_POSITION
        """),
        con,
        params={"tname": tabla.split(".")[1], "tschema": tabla.split(".")[0]},
    )


nombre_dt = None
nombre_val = None


candidatos_dt = ["datetime", "fecha_hora", "dt", "fecha", "timestamp"]
candidatos_val = ["value", "valor", "mw", "demanda"]

for c in candidatos_dt:
    if (cols["COLUMN_NAME"].str.lower() == c).any():
        nombre_dt = cols.loc[cols["COLUMN_NAME"].str.lower() == c, "COLUMN_NAME"].iloc[0]
        break

for c in candidatos_val:
    if (cols["COLUMN_NAME"].str.lower() == c).any():
        nombre_val = cols.loc[cols["COLUMN_NAME"].str.lower() == c, "COLUMN_NAME"].iloc[0]
        break


if nombre_dt is None:
    tipots = cols[cols["DATA_TYPE"].str.contains("date", case=False, na=False)]
    if not tipots.empty:
        nombre_dt = tipots["COLUMN_NAME"].iloc[0]

if nombre_val is None:
    tipoval = cols[cols["DATA_TYPE"].str.contains("float|real|decimal|numeric|int", case=False, na=False)]
    if not tipoval.empty:
        nombre_val = tipoval["COLUMN_NAME"].iloc[0]

if not nombre_dt or not nombre_val:
    raise SystemExit(f"No se han podido identificar las columnas de fecha/valor en {tabla}.\n"
                     f"Columnas disponibles:\n{cols}")


sql = text(f"""
SELECT [{nombre_dt}] AS dt, [{nombre_val}] AS mw
FROM {tabla}
WHERE [{nombre_dt}] >= DATEADD(day, -:dias, SYSDATETIMEOFFSET())
ORDER BY [{nombre_dt}];
""")

with engine.connect() as con:
    df = pd.read_sql(sql, con, params={"dias": dias})

if df.empty:
    raise SystemExit(f"Sin datos en {tabla} para los últimos {dias} días.")

# -------- Preprocesado: asegurar frecuencia 5 min --------
df["dt"] = pd.to_datetime(df["dt"])           # convierte a datetime (mantiene offset si es DATETIMEOFFSET)
df = df.sort_values("dt").set_index("dt")

# Reindexar estrictamente a 5 minutos y rellenar huecos suavemente
df = df.resample("5T").mean()
df["mw"] = pd.to_numeric(df["mw"], errors="coerce").interpolate(limit_direction="both")

# -------- Plot --------
plt.figure(figsize=(14,6))
plt.plot(df.index, df["mw"], linewidth=0.9, label="Demanda (MW)")
plt.title(f"Demanda eléctrica — últimos {dias} días (resolución 5 min)")
plt.xlabel("Tiempo")
plt.ylabel("MW")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
