import json
import logging

import pandas as pd

import requests

logger = logging.getLogger('requests')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info("Reading test data")
df = pd.read_csv('synthetic_data.csv').drop(columns=['condition'])
data = df.to_dict('records')
logger.info("Data successfully read")

for request in data:
    logger.info('Sending request...')
    response = requests.post(
        'http://127.0.0.1:8000/predict',
        json.dumps(request)
    )
    logger.info(f'Status Code: {response.status_code}')
    logger.info(f'Message: {response.json()}')
