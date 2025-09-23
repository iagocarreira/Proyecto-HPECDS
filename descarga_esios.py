import os, requests, pandas as pd
from datetime import datetime, timedelta
from dateutil import tz

ESIOS_TOKEN = os.environ.get("ESIOS_TOKEN") or "ddcbd54fa41b494243f3a6094062af3f41a4675956a8f50a2b92b80bd0fbc71a"

HEADERS = {
    "Accept": "application/json; application/vnd.esios-api-v1+json",
    "Content-Type": "application/json",
    "x-api-key": ESIOS_TOKEN   # 👈 aquí el cambio
}

INDICATOR_ID = 1293  # Demanda real

def fetch_demanda(start_dt_iso: str, end_dt_iso: str):
    url = f"https://api.esios.ree.es/indicators/{INDICATOR_ID}"
    params = {"start_date": start_dt_iso, "end_date": end_dt_iso}

    r = requests.get(url, headers=HEADERS, params=params, timeout=30)
    print("URL:", r.url, " ->", r.status_code)
    r.raise_for_status()
    values = r.json().get("indicator", {}).get("values", [])
    return pd.DataFrame(values)

if __name__ == "__main__":
    tz_madrid = tz.gettz("Europe/Madrid")
    hoy = datetime.now(tz_madrid).date()
    inicio = (hoy - timedelta(days=3)).isoformat()
    fin = hoy.isoformat()

    df = fetch_demanda(inicio + "T00:00:00+02:00", fin + "T00:00:00+02:00")
    print(df.head())
    df.to_csv(f"demanda_real_{inicio}_a_{fin}.csv", index=False)
