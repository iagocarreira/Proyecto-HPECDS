import logging, requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import pandas as pd

class ESIOSClient:
    """Cliente HTTP con reintentos para la API ESIOS."""
    def __init__(self, token: str):
        self.base_url = "https://api.esios.ree.es/indicators"
        self.headers = {
            "Accept": "application/json; application/vnd.esios-api-v1+json",
            "Content-Type": "application/json",
            "x-api-key": token
        }
        self.session = self._create_session()

    def _create_session(self):
        retries = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        session = requests.Session()
        session.mount("https://", HTTPAdapter(max_retries=retries))
        return session

    def fetch(self, indicator_id: int, start_iso: str, end_iso: str, col_prefix: str):
        url = f"{self.base_url}/{indicator_id}"
        try:
            r = self.session.get(url, headers=self.headers,
                                params={"start_date": start_iso, "end_date": end_iso},
                                timeout=60)
            r.raise_for_status()
            data = r.json().get("indicator", {}).get("values", [])

            # 🔧 Transformamos aquí los nombres de las columnas
            if data:
                df = pd.DataFrame(data)
                if 'value' in df.columns:
                    df.rename(columns={'value': f"valor_{col_prefix}"}, inplace=True)
                return df.to_dict(orient="records")
            else:
                logging.warning(f" No se recibieron datos para {col_prefix}")
                return []
        except Exception as e:
            logging.error(f"Error extrayendo {col_prefix}: {e}")
            return []

