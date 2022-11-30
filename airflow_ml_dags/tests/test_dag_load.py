import pytest

from airflow.models import DagBag


@pytest.fixture
def dag_bag():
    return DagBag()


def test_generate_dag_loaded(dag_bag):
    dag = dag_bag.get_dag(dag_id='data_generation')
    assert dag_bag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 2


def test_train_dag_loaded(dag_bag):
    dag = dag_bag.get_dag(dag_id='train_pipeline')
    assert dag_bag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 6


def test_predict_dag_loaded(dag_bag):
    dag = dag_bag.get_dag(dag_id='predict_pipeline')
    assert dag_bag.import_errors == {}
    assert dag is not None
    assert len(dag.tasks) == 3
