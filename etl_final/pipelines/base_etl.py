from abc import ABC, abstractmethod

class BaseETL(ABC):
    """Clase base para ETLs."""
    @abstractmethod
    def run(self):
        pass
