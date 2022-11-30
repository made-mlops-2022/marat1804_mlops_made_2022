from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from docker.types import Mount
from common import LOCAL_DIR, default_args, check_file


AIRFLOW_RAW_DATA_PATH = "/opt/airflow/data/raw/{{ ds }}"
HOST_RAW_DATA_PATH = "/data/raw/{{ ds }}"
HOST_PROCESSED_DATA_PATH = "/data/processed/{{ ds }}"
HOST_SPLITTED_DATA_PATH = "/data/splitted/{{ ds }}"
HOST_PREDICTIONS_PATH = "/data/predictions/{{ ds }}"
HOST_MODELS_PATH = "/data/models/{{ ds }}"
HOST_METRICS_PATH = "/data/metrics/{{ ds }}"


with DAG(
    'train_pipeline',
    default_args=default_args,
    schedule_interval='@weekly',
    start_date=datetime(2022, 11, 5)
) as dag:
    wait_for_data = PythonSensor(
        task_id='wait-for-data',
        python_callable=check_file,
        op_args=[f'{AIRFLOW_RAW_DATA_PATH}/data.csv'],
        timeout=6000,
        poke_interval=60,
        retries=20,
        mode='poke'
    )

    wait_for_target = PythonSensor(
        task_id='wait-for-target',
        python_callable=check_file,
        op_args=[f'{AIRFLOW_RAW_DATA_PATH}/target.csv'],
        timeout=600,
        poke_interval=60,
        retries=20,
        mode='poke'
    )

    preprocess_data = DockerOperator(
        image='airflow-data-preprocess',
        command=f'--input-dir {HOST_RAW_DATA_PATH} --output-dir {HOST_PROCESSED_DATA_PATH}',
        network_mode="bridge",
        task_id="docker-operator-data-preprocess",
        do_xcom_push=False,
        auto_remove='true',
        mounts=[Mount(source=LOCAL_DIR, target='/data', type='bind')]
    )

    train_test_split = DockerOperator(
        image='airflow-train-test-split',
        command=f'--input-dir {HOST_PROCESSED_DATA_PATH} --output-dir {HOST_SPLITTED_DATA_PATH}',
        network_mode="bridge",
        task_id="docker-operator-train-test-split",
        do_xcom_push=False,
        auto_remove='true',
        mounts=[Mount(source=LOCAL_DIR, target='/data', type='bind')]
    )

    train_model = DockerOperator(
        image='airflow-train-model',
        command=f'--input-dir {HOST_SPLITTED_DATA_PATH} --output-dir {HOST_MODELS_PATH}',
        network_mode="bridge",
        task_id="docker-operator-train-model",
        do_xcom_push=False,
        auto_remove='true',
        mounts=[Mount(source=LOCAL_DIR, target='/data', type='bind')]
    )

    validate_model = DockerOperator(
        image='airflow-validate-model',
        command=f'--input-dir {HOST_SPLITTED_DATA_PATH} --model-dir {HOST_MODELS_PATH} '
                f'--output-dir {HOST_METRICS_PATH}',
        network_mode="bridge",
        task_id="docker-operator-validate-model",
        do_xcom_push=False,
        auto_remove='true',
        mounts=[Mount(source=LOCAL_DIR, target='/data', type='bind')]
    )

    [wait_for_target, wait_for_data] >> preprocess_data >> train_test_split >> train_model >> validate_model
