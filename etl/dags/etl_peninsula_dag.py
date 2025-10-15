from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pendulum import timezone
import sys
import os

# Añadir la ruta de tus scripts
sys.path.insert(0, os.path.abspath('/opt/airflow/scripts'))
import etl  # etl.py debe exponer una función run_etl()

tz = timezone("Europe/Madrid")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Configurar el start_date para que esté en una fecha futura
# La hora debería ser 9:00 AM en Madrid, y no la hora exacta en que se ejecuta el código

with DAG(
    dag_id='etl_peninsula_pipeline',
    default_args=default_args,
    description='ETL Peninsula pipeline',
    schedule_interval='0 9 * * *',  # Ejecutar todos los días a las 9 AM (hora local de Madrid)
    start_date=datetime(2025, 10, 6, 9, 0, 0, tzinfo=tz),  # Asegúrate de que start_date sea una fecha futura
    catchup=False,  # No ejecutar tareas pasadas
    tags=['etl', 'pipeline'],
    is_paused_upon_creation=False  
) as dag:

    etl_task = PythonOperator(
        task_id='run_etl',
        python_callable=etl.run_etl,
    )
