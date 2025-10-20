from config.logger import setup_logging
from pipelines.etl_weekly import ETLWeekly
from pipelines.etl_historic import ETLHistoric

if __name__ == "__main__":
    setup_logging()

    # Cambiar lo que se quiera ejecutar:
    etl = ETLWeekly()       # ETL semanal
    #etl = ETLHistoric()   # ETL histórica

    etl.run()
