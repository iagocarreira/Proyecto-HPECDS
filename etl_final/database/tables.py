import logging

class TableManager:
    """Crea y mantiene las tablas SQL necesarias."""
    def __init__(self, conn_manager):
        self.conn_manager = conn_manager

    def ensure_tables_exist(self):
        """Crea las tablas histórica y semanal si no existen."""
        conn = self.conn_manager.connect()
        cursor = conn.cursor()
        try:
            cursor.execute("""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'demanda_peninsula')
            BEGIN
                CREATE TABLE dbo.demanda_peninsula (
                    fecha DATETIME NOT NULL PRIMARY KEY,
                    fecha_utc DATETIME NOT NULL,
                    valor_real FLOAT NULL,
                    valor_previsto FLOAT NULL,
                    valor_programado FLOAT NULL,
                    hora INT NULL,
                    dia_semana NVARCHAR(20) NULL,
                    es_fin_semana BIT NULL,
                    error_absoluto FLOAT NULL,
                    error_relativo_pct FLOAT NULL
                );
            END
            """)
            cursor.execute("""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'demanda_peninsula_semana')
            BEGIN
                CREATE TABLE dbo.demanda_peninsula_semana (
                    fecha DATETIME NOT NULL PRIMARY KEY,
                    fecha_utc DATETIME NOT NULL,
                    valor_real FLOAT NULL,
                    valor_previsto FLOAT NULL,
                    valor_programado FLOAT NULL,
                    hora INT NULL,
                    dia_semana NVARCHAR(20) NULL,
                    es_fin_semana BIT NULL,
                    error_absoluto FLOAT NULL,
                    error_relativo_pct FLOAT NULL
                );
            END
            """)
            cursor.execute("""
            IF NOT EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME = 'etl_log')
            BEGIN
                CREATE TABLE dbo.etl_log (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    dag_name NVARCHAR(255),
                    start_time DATETIMEOFFSET,
                    end_time DATETIMEOFFSET,
                    duration_seconds FLOAT,
                    status NVARCHAR(50)
                );
            END
            """)
            conn.commit()
            logging.info("Tablas principales verificadas o creadas.")
        finally:
            cursor.close()
            conn.close()
