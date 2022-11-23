import os
import pickle

import pandas as pd
from fastapi import FastAPI
from schemas import InputData
from fastapi_health import health


app = FastAPI()
model = None
transformer = None


@app.on_event('startup')
def load_model_and_transformer():
    model_path = os.getenv('PATH_TO_MODEL')
    transformer_path = os.getenv('PATH_TO_TRANSFORMER')

    with open(transformer_path, 'rb') as f:
        global transformer
        transformer = pickle.load(f)

    with open(model_path, 'rb') as f:
        global model
        model = pickle.load(f)


@app.post('/predict')
def get_prediction(input_data: InputData):
    data = pd.DataFrame([input_data.dict()])
    X = transformer.transform(data)
    y = model.predict(X)
    text = 'healthy' if not y[0] else 'sick'
    return {'result': f'{y} - is for {text}'}


def check_model_is_ready():
    return model is not None


def check_transformer_is_ready():
    return transformer is not None


async def success_handler(**kwargs):
    return {"status": "model and transformer are ready"}


async def failure_handler(**kwargs):
    return {"status": "model or transformer are not ready"}


app.add_api_route("/health",
                  health([check_model_is_ready, check_transformer_is_ready],
                         success_handler=success_handler,
                         failure_handler=failure_handler,
                         success_status=200,
                         failure_status=503
                         )
                  )
