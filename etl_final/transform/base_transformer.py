import pandas as pd
from abc import ABC, abstractmethod

class BaseTransformer(ABC):
    """Clase base para pipelines de transformación."""
    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
