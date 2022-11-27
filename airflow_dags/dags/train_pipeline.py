from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.python import PythonSensor
from docker.types import Mount
from common import LOCAL_DIR, default_args, check_file


with DAG(
    'train_pipeline',
    default_args=default_args,
    schedule_interval='@weekly',
    start_date=datetime(2022, 11, 22)
) as dag:
    wait_for_data = PythonSensor(
        task_id='wait-for-data',
        python_callable=check_file,
        op_args=['/opt/airflow/data/raw/{{ ds }}/data.csv'],
        timeout=60,
        poke_interval=10,
        retries=20,
        mode='poke'
    )

    wait_for_target = PythonSensor(
        task_id='wait-for-target',
        python_callable=check_file,
        op_args=['/opt/airflow/data/raw/{{ ds }}/target.csv'],
        timeout=60,
        poke_interval=10,
        retries=20,
        mode='poke'
    )

    preprocess_data = DockerOperator(
        image='airflow-data-preprocess',
        command='--input-dir /data/raw/{{ ds }} --output-dir /data/preprocessed/{{ ds }}',
        network_mode="bridge",
        task_id="docker-operator-data-preprocess",
        do_xcom_push=False,
        auto_remove='true',
        mounts=[Mount(source=LOCAL_DIR, target='/data', type='bind')]
    )

    train_test_split = DockerOperator(
        image='airflow-train-test-split',
        command='--input-dir /data/preprocessed/{{ ds }} --output-dir /data/split/{{ ds }}',
        network_mode="bridge",
        task_id="docker-operator-train-test-split",
        do_xcom_push=False,
        auto_remove='true',
        mounts=[Mount(source=LOCAL_DIR, target='/data', type='bind')]
    )

    train_model = DockerOperator(
        image='airflow-train-model',
        command='--input-dir /data/split/{{ ds }} --output-dir /data/models/{{ ds }}',
        network_mode="bridge",
        task_id="docker-operator-train-model",
        do_xcom_push=False,
        auto_remove='true',
        mounts=[Mount(source=LOCAL_DIR, target='/data', type='bind')]
    )

    validate_model = DockerOperator(
        image='airflow-validate-model',
        command='--input-dir /data/split/{{ ds }} --model-dir /data/models/{{ ds }} '
                '--output-dir /data/metrics/{{ ds }}',
        network_mode="bridge",
        task_id="docker-operator-validate-model",
        do_xcom_push=False,
        auto_remove='true',
        mounts=[Mount(source=LOCAL_DIR, target='/data', type='bind')]
    )

    [wait_for_target, wait_for_data] >> preprocess_data >> train_test_split >> train_model >> validate_model
