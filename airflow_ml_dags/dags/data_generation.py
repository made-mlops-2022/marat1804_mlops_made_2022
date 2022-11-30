from datetime import datetime

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from docker.types import Mount
from common import default_args, LOCAL_DIR


with DAG(
    'data_generation',
    default_args=default_args,
    schedule_interval='@daily',
    start_date=datetime(2022, 11, 5)
) as dag:
    t1 = BashOperator(task_id="print_date", bash_command="date")
    t2 = DockerOperator(
        image='airflow-data-generation',
        command='--output-dir /data/raw/{{ ds }}',
        network_mode="bridge",
        task_id="docker-operator-data-generation",
        do_xcom_push=False,
        auto_remove='true',
        mounts=[Mount(source=LOCAL_DIR, target='/data', type='bind')]
    )

    t1 >> t2
