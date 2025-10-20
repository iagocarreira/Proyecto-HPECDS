from datetime import datetime
from dateutil import tz

def now_madrid():
    """Devuelve la hora actual en Madrid."""
    tz_madrid = tz.gettz("Europe/Madrid")
    return datetime.now(tz_madrid)
