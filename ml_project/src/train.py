import json
import logging

import hydra
import numpy as np
import pandas as pd
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

from data import split_data, read_csv_data, get_x_and_y
from config_classes import TrainConfig
from features import CustomTransformer
from model import Classifier

logger = logging.getLogger()
logger.setLevel(logging.INFO)

cs = ConfigStore.instance()
cs.store(name='train', node=TrainConfig)


@hydra.main(version_base=None, config_path='../configs', config_name='train_config')
def run_pipeline(params: TrainConfig):
    params: TrainConfig = instantiate(params, _convert_='partial')
    logger.info(f'Train started')

    data = read_csv_data(params.input_data_path)
    logger.info(f'Read data, shape = {data.shape}')

    X, y = get_x_and_y(data, params.feature_details.target_column)

    X_train, X_test, y_train, y_test = split_data(X, y, params.train_test_split_params)
    logger.info(f'Train samples: {X_train.shape}')
    logger.info(f'Test samples: {X_test.shape}')
    X_test.to_csv(params.input_test_data_path, index=False, sep=',', encoding='utf-8')
    logger.info(f'Saved unlabeled data to {params.input_test_data_path}')

    logger.info('Preprocessing features')
    transformer = CustomTransformer(params.model.preprocessing_params, params.feature_details)
    transformer.fit(X_train)
    X_train = transformer.transform(X_train)
    df_train_processed = np.concatenate([X_train, np.array(y_train)[:, None]], axis=-1)
    df_processed = pd.DataFrame(df_train_processed)
    df_processed.to_csv(params.model.path_to_processed_data, index=False, sep=',', encoding='utf-8')
    logger.info(f'Saved preprocessed data to {params.model.path_to_processed_data}')
    logger.info(f'Saving transformer to {params.model.path_to_transformer}')
    transformer.save_transformer(params.model.path_to_transformer)
    logger.info(f'Transformer saved')

    logger.info(f'Start training {params.model.train_params.model_type}')

    model = Classifier(params.model.train_params)
    model.fit(X_train, y_train)
    if params.model.train_params.grid_search:
        logger.info(f'GridSearch was used')
        logger.info(f'Best hyper parameters for {params.model.train_params.model_type} are {model.model_best_params}')
        logger.info(f'Best score for cv {model.model_best_score_}')
    logger.info(f'{params.model.train_params.model_type} was trained')

    logger.info(f'Preprocessing test data')
    X_test = transformer.transform(X_test)
    logger.info(f'Predicting score for test data')
    y_pred = model.predict(X_test)
    logger.info(f'Evaluating model')
    metrics = model.evaluate_model(y_pred, y_test)

    for k, v in metrics.items():
        logger.info(f'{k} = {round(v, 5)}')

    logger.info(f'Saving model metrics...')
    with open(params.model.path_to_model_metric, 'w') as file:
        if params.model.train_params.grid_search:
            json.dump({**model.model_best_score_, **metrics}, file, indent=4)
        else:
            json.dump(metrics, file, indent=4)
    logger.info(f'Model metrics saved to {params.model.path_to_model_metric}')

    logger.info(f'Saving model to {params.model.path_to_output_model}')
    model.save_model(params.model.path_to_output_model)
    logger.info(f'Model saved')
    logger.info(f'Train ended')


if __name__ == '__main__':
    run_pipeline()
