from datetime import timedelta
import os

from airflow.utils.email import send_email
from airflow.models import Variable


LOCAL_DIR = Variable.get('local_dir')
LOCAL_MODEL_DIR = Variable.get('local_model_dir')


def check_file(filename):
    return os.path.exists(filename)


def email_failure_notification(context):
    dag_run = context.get('dag_run')
    msg = "DAG failed"
    subject = f"DAG {dag_run} has failed"
    send_email(to=default_args['email'], subject=subject, html_content=msg)


default_args = {
    "owner": "marat_1804",
    "email": ["maratcoop11@gmail.com"],
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    'on_failure_callback': email_failure_notification
}
