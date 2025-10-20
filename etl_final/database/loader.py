import logging
import pandas as pd
import datetime

class SQLLoader:
    """Carga datos en SQL Server usando MERGE para evitar duplicados y registra logs de ejecución."""
    
    def __init__(self, conn_manager):
        self.conn_manager = conn_manager  # nombre coherente con el resto del sistema

    # ------------------------------------------------------------------------
    # MÉTODO PRINCIPAL DE INSERCIÓN
    # ------------------------------------------------------------------------
    def merge_insert(self, df: pd.DataFrame, table_name: str):
        """Inserta o actualiza datos en SQL Server usando MERGE."""
        conn = self.conn_manager.connect()
        cursor = conn.cursor()
        cursor.fast_executemany = True

        try:
            rows = [
                (
                    row["fecha"], row["datetime_utc"], row.get("valor_real"),
                    row.get("valor_previsto"), row.get("valor_programado"),
                    row.get("hora"), row.get("dia_semana"),
                    int(row.get("es_fin_semana")) if pd.notna(row.get("es_fin_semana")) else None,
                    row.get("error_absoluto"), row.get("error_relativo_pct")
                )
                for _, row in df.iterrows() if pd.notna(row.get("fecha"))
            ]

            if not rows:
                logging.warning("No hay filas válidas para insertar.")
                return

            sql = f"""
            MERGE INTO dbo.{table_name} AS target
            USING (VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)) 
            AS source (fecha, fecha_utc, valor_real, valor_previsto, valor_programado,
                       hora, dia_semana, es_fin_semana, error_absoluto, error_relativo_pct)
            ON target.fecha = source.fecha
            WHEN NOT MATCHED THEN
                INSERT (fecha, fecha_utc, valor_real, valor_previsto, valor_programado,
                        hora, dia_semana, es_fin_semana, error_absoluto, error_relativo_pct)
                VALUES (source.fecha, source.fecha_utc, source.valor_real, source.valor_previsto,
                        source.valor_programado, source.hora, source.dia_semana,
                        source.es_fin_semana, source.error_absoluto, source.error_relativo_pct);
            """
            cursor.executemany(sql, rows)
            conn.commit()
            logging.info(f"{len(rows)} filas procesadas en {table_name}.")
        except Exception as e:
            logging.error(f"Error al insertar en {table_name}: {e}")
        finally:
            cursor.close()
            conn.close()

    # ------------------------------------------------------------------------
    # MÉTODO PARA REGISTRAR LA EJECUCIÓN DE UNA ETL
    # ------------------------------------------------------------------------
    def log_etl_run(self, dag_name: str, start_time: datetime.datetime, status: str):
        """Registra la ejecución de una ETL en la tabla etl_log."""
        try:
            # Aseguramos compatibilidad entre datetimes con/ sin zona horaria
            end_time = datetime.datetime.now(tz=start_time.tzinfo) if start_time.tzinfo else datetime.datetime.now()
            duration = (end_time - start_time).total_seconds()

            query = """
            INSERT INTO etl_log (dag_name, start_time, end_time, duration_seconds, status)
            VALUES (?, ?, ?, ?, ?)
            """
            values = (dag_name, start_time, end_time, duration, status)

            conn = self.conn_manager.connect()
            cursor = conn.cursor()
            cursor.execute(query, values)
            conn.commit()
            cursor.close()
            conn.close()

            logging.info(f"Registro de ETL '{dag_name}' insertado en etl_log con estado {status}.")
        except Exception as e:
            logging.error(f"Error registrando ejecución de '{dag_name}' en etl_log: {e}")
    
    def roll_week_window(self,
                         week_table: str = "demanda_peninsula_semana",
                         hist_table: str = "demanda_peninsula",
                         max_days: int = 8):
        """
        Garantiza que la tabla semanal tenga 'max_days' días exactos.
        Si hay más, mueve el/los día(s) más antiguo(s) a histórico y los borra de la semanal.
        - Idempotente: no duplica histórico (usa NOT EXISTS).
        """
        conn = self.conn_manager.connect()
        cur = conn.cursor()
        try:
            # 1) Lista de días distintos en la semanal (ordenados)
            cur.execute(f"""
                SELECT CAST(fecha AS date) AS d
                FROM dbo.{week_table}
                GROUP BY CAST(fecha AS date)
                ORDER BY d ASC;
            """)
            days = [row[0] for row in cur.fetchall()]

            if not days:
                logging.info(f"[rota] {week_table} está vacía; nada que rotar.")
                return

            # 2) Mientras haya más de max_days días, mueve el más antiguo a histórico
            moved = 0
            while len(days) > max_days:
                oldest = days[0]

                # 2.1) Insertar en histórico evitando duplicados exactos por 'fecha'
                cur.execute(f"""
                    INSERT INTO dbo.{hist_table}
                        (fecha, fecha_utc, valor_real, valor_previsto, valor_programado,
                         hora, dia_semana, es_fin_semana, error_absoluto, error_relativo_pct)
                    SELECT s.fecha, s.fecha_utc, s.valor_real, s.valor_previsto, s.valor_programado,
                           s.hora, s.dia_semana, s.es_fin_semana, s.error_absoluto, s.error_relativo_pct
                    FROM dbo.{week_table} AS s
                    WHERE CAST(s.fecha AS date) = ?
                      AND NOT EXISTS (
                          SELECT 1
                          FROM dbo.{hist_table} AS h
                          WHERE h.fecha = s.fecha
                      );
                """, (oldest,))

                # 2.2) Borrar ese día de la semanal
                cur.execute(f"""
                    DELETE FROM dbo.{week_table}
                    WHERE CAST(fecha AS date) = ?;
                """, (oldest,))

                conn.commit()
                moved += 1
                days.pop(0)  # ya no está en la semanal

            if moved > 0:
                logging.info(f"[rota] Movidos {moved} día(s) antiguo(s) de {week_table} → {hist_table}.")
            else:
                logging.info(f"[rota] {week_table} ya tenía ≤ {max_days} días. Nada que mover.")
        except Exception as e:
            conn.rollback()
            logging.error(f"[rota] Error al rotar ventana semanal: {e}")
            raise
        finally:
            cur.close()
            conn.close()
