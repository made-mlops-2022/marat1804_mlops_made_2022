import pytest

from airflow.models import DagBag


@pytest.fixture
def dag_bag():
    return DagBag()


def assert_dag_dict_equal(source, dag):
    assert dag.task_dict.keys() == source.keys()
    for task_id, downstream_list in source.items():
        assert dag.has_task(task_id)
        task = dag.get_task(task_id)
        assert task.downstream_task_ids == set(downstream_list)


def test_generate_dag_structure(dag_bag):
    dag = dag_bag.get_dag(dag_id='data_generation')
    assert_dag_dict_equal(
        {
            'print_date': ['docker-operator-data-generation'],
            'docker-operator-data-generation': []
        },
        dag,
    )


def test_train_dag_structure(dag_bag):
    dag = dag_bag.get_dag(dag_id='train_pipeline')
    assert_dag_dict_equal(
        {
            'wait-for-data': ['docker-operator-data-preprocess'],
            'wait-for-target': ['docker-operator-data-preprocess'],
            'docker-operator-data-preprocess': ['docker-operator-train-test-split'],
            'docker-operator-train-test-split': ['docker-operator-train-model'],
            'docker-operator-train-model': ['docker-operator-validate-model'],
            'docker-operator-validate-model': []
        },
        dag,
    )


def test_predict_dag_structure(dag_bag):
    dag = dag_bag.get_dag(dag_id='predict_pipeline')
    assert_dag_dict_equal(
        {
            'wait-for-predict-data': ['docker-operator-predict-data-preprocess'],
            'docker-operator-predict-data-preprocess': ['docker-operator-predict-data'],
            'docker-operator-predict-data': [],
        },
        dag,
    )
