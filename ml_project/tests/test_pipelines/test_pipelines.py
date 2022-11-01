import os
import sys
import unittest
import logging
import json
import shutil

from hydra import compose, initialize
from hydra.utils import instantiate
import pandas as pd

PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH, 'src'
)
sys.path.append(SOURCE_PATH)

from src import run_predict_pipeline, run_train_pipeline
from model import Classifier
from features import CustomTransformer
from utils import load_pickle_file, save_pickle_file


logging.disable(logging.CRITICAL)


class TestTrainAndPredictPipeline(unittest.TestCase):
    def test_a_run_train_pipeline(self):
        os.makedirs('tests/models', exist_ok=True)

        with initialize(version_base=None, config_path='../test_configs'):
            params = compose(config_name="test_config")
        run_train_pipeline(params)

        self.assertTrue(os.path.exists(params.input_test_data_path))
        self.assertTrue(os.path.exists(params.model.path_to_output_model))
        self.assertTrue(os.path.exists(params.model.path_to_model_metric))
        self.assertTrue(os.path.exists(params.model.path_to_processed_data))
        self.assertTrue(os.path.exists(params.model.path_to_transformer))

        model = load_pickle_file(params.model.path_to_output_model)
        self.assertIsInstance(model, Classifier)
        transformer = load_pickle_file(params.model.path_to_transformer)
        self.assertIsInstance(transformer, CustomTransformer)

        df_train = pd.read_csv(params.model.path_to_processed_data)
        self.assertEqual(df_train.shape, (112, 29))

        df_test = pd.read_csv(params.input_test_data_path)
        self.assertEqual(df_test.shape, (38, 13))

        with open(params.model.path_to_model_metric, 'r') as file:
            metrics = json.load(file)
        self.assertIn('recall_val', metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('roc_auc', metrics)

    def test_b_run_predict_pipeline(self):
        with initialize(version_base=None, config_path='../test_configs'):
            params = compose(config_name="test_config")
        params = instantiate(params, _convert_='partial')
        run_predict_pipeline.callback(model_path=params.model.path_to_output_model,
                                      transformer_path=params.model.path_to_transformer,
                                      data_path=params.input_test_data_path,
                                      output_path='tests/models/prediction.csv')
        self.assertTrue(os.path.exists('tests/models/prediction.csv'))
        y_pred = pd.read_csv('tests/models/prediction.csv')
        self.assertEqual(y_pred.shape, (38, 1))
        shutil.rmtree('tests/models')


if __name__ == '__main__':
    unittest.main()
