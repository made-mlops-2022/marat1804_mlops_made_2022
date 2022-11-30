from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from docker.types import Mount
from common import LOCAL_DIR, default_args, check_file


AIRFLOW_RAW_DATA_PATH = "/opt/airflow/data/raw/{{ ds }}"
HOST_RAW_DATA_PATH = "/data/raw/{{ ds }}"
HOST_PROCESSED_DATA_PATH = "/data/processed/{{ ds }}"
HOST_PREDICTIONS_PATH = "/data/predictions/{{ ds }}"


with DAG(
    'predict_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=datetime(2022, 11, 5)
) as dag:
    wait_for_data = PythonSensor(
        task_id='wait-for-predict-data',
        python_callable=check_file,
        op_args=[f'{AIRFLOW_RAW_DATA_PATH}/data.csv'],
        timeout=6000,
        poke_interval=60,
        retries=20,
        mode='poke'
    )

    preprocess_data = DockerOperator(
        image='airflow-data-preprocess',
        command=f'--input-dir {HOST_RAW_DATA_PATH} --output-dir {HOST_PROCESSED_DATA_PATH}',
        network_mode="bridge",
        task_id="docker-operator-predict-data-preprocess",
        do_xcom_push=False,
        auto_remove='true',
        mounts=[Mount(source=LOCAL_DIR, target='/data', type='bind')]
    )

    predict = DockerOperator(
        image='airflow-predict',
        command=f'--input-dir {HOST_PROCESSED_DATA_PATH} --output-dir {HOST_PREDICTIONS_PATH} '
                '--model-dir /data/models/{{ var.value.model_dir }}',
        network_mode="bridge",
        task_id="docker-operator-predict-data",
        do_xcom_push=False,
        auto_remove='true',
        mounts=[Mount(source=LOCAL_DIR, target='/data', type='bind')]
    )

    wait_for_data >> preprocess_data >> predict
