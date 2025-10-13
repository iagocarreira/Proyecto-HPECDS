# Usa una imagen base oficial de Airflow
FROM apache/airflow:2.5.0

# Instalar dependencias de Python
RUN pip install --upgrade pip
COPY requirements.txt /opt/airflow/requirements.txt
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

# Copiar el archivo DAG de Airflow
COPY dags /opt/airflow/dags/

# Copiar el script de ETL
COPY etl_peninsula /opt/airflow/scripts/

# Establecer el directorio de trabajo de Airflow
ENV AIRFLOW_HOME=/opt/airflow

# Inicializar la base de datos de Airflow (ahora usando SQL Server)
RUN airflow db init

# Crear un usuario admin con las credenciales que deseas
RUN airflow users create \
    --username user1_app \
    --firstname "user1" \
    --lastname "app" \
    --role Admin \
    --email "ugrupo1@example.com" \
    --password TuPasswordSegura123!

# Comando para ejecutar Airflow (scheduler y webserver)
CMD ["bash", "-c", "airflow scheduler & airflow webserver"]

