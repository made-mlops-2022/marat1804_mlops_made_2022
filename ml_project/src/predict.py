import logging

import click
import pandas as pd

from data import read_csv_data
from features import CustomTransformer
from model import Classifier
from utils import load_pickle_file

logger = logging.getLogger('predict')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False


@click.command()
@click.option('--model_path', type=click.Path(exists=True),
              default='models/model_logistic_regression.pkl',
              help='Path to model')
@click.option('--transformer_path', type=click.Path(exists=True),
              default='models/transformers/transformer_logistic_regression.pkl',
              help='Path to transformer')
@click.option('--data_path', type=click.Path(exists=True),
              default='data/test/heart_cleveland_upload_test_unlabeled.csv',
              help='Path to data')
@click.option('--output_path', type=click.Path(exists=False),
              default='models/predictions/pred_logistic_regression.csv',
              help='Path to predictions')
def run_predict_pipeline(model_path: str, transformer_path: str,
                         data_path: str, output_path: str):
    logger.info('Prediction started')
    logger.info(f'Reading data from {data_path}')
    df = read_csv_data(data_path)
    logger.info(f'Read data, shape {df.shape}')

    logger.info(f'Loading transformer from {transformer_path}')
    transformer: CustomTransformer = load_pickle_file(transformer_path)
    logger.info('Transformer loaded')

    logger.info('Preprocessing features')
    X = transformer.transform(df)
    logger.info('Preprocessing ended')

    logger.info(f'Loading model from {model_path}')
    model: Classifier = load_pickle_file(model_path)
    logger.info(f'{model.train_params.model_type} loaded')

    logger.info('Making prediction')
    y = model.predict(X)
    df_y = pd.DataFrame(y)
    df_y.to_csv(output_path, index=False, sep=',', encoding='utf-8')
    logger.info(f'Saved predictions to {output_path}')
    logger.info('Prediction ended')


if __name__ == '__main__':
    run_predict_pipeline()
