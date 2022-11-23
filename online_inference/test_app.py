import json

from fastapi.testclient import TestClient
import pytest

from main import load_model_and_transformer, app

client = TestClient(app)


@pytest.fixture
def init_model_and_transformer():
    load_model_and_transformer()


def test_predict_sick(init_model_and_transformer):
    request = {
        'age': 18,
        'sex': 0,
        'cp': 3,
        'trestbps': 105,
        'chol': 120,
        'fbs': 1,
        'restecg': 0,
        'thalach': 91,
        'exang': 0,
        'oldpeak': 1.7,
        'slope': 0,
        'ca': 0,
        'thal': 2
    }
    response = client.post(
        '/predict',
        data=json.dumps(request)
    )
    assert response.status_code == 200
    assert response.json() == {'result': '[0] - is for healthy'}


def test_predict_healthy(init_model_and_transformer):
    request = {
        'age': 65,
        'sex': 1,
        'cp': 3,
        'trestbps': 165,
        'chol': 120,
        'fbs': 1,
        'restecg': 0,
        'thalach': 91,
        'exang': 0,
        'oldpeak': 1.7,
        'slope': 0,
        'ca': 3,
        'thal': 2
    }
    response = client.post(
        '/predict',
        data=json.dumps(request)
    )
    assert response.status_code == 200
    assert response.json() == {'result': '[1] - is for sick'}


def test_health(init_model_and_transformer):
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {"status": "model and transformer are ready"}


def test_invalid_cat_arg(init_model_and_transformer):
    request = {
        'age': 65,
        'sex': 3,
        'cp': 3,
        'trestbps': 165,
        'chol': 120,
        'fbs': 1,
        'restecg': 0,
        'thalach': 91,
        'exang': 0,
        'oldpeak': 1.7,
        'slope': 0,
        'ca': 3,
        'thal': 2
    }
    response = client.post(
        '/predict',
        data=json.dumps(request)
    )
    assert response.status_code == 422
    assert 'sex' in response.json()['detail'][0]['loc']
    assert response.json()['detail'][0]['msg'] == 'unexpected value; permitted: 0, 1'


def test_invalid_cont_arg(init_model_and_transformer):
    request = {
        'age': 65,
        'sex': 1,
        'cp': 3,
        'trestbps': 1001,
        'chol': 120,
        'fbs': 1,
        'restecg': 0,
        'thalach': 91,
        'exang': 0,
        'oldpeak': 1.7,
        'slope': 0,
        'ca': 3,
        'thal': 2
    }
    response = client.post(
        '/predict',
        data=json.dumps(request)
    )
    assert response.status_code == 422
    assert 'trestbps' in response.json()['detail'][0]['loc']
    assert response.json()['detail'][0]['msg'] == 'Trestbps must be between 1 and 300'
