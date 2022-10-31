import os
import sys
import unittest

PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH, 'src'
)
sys.path.append(SOURCE_PATH)

from features import CustomTransformer
from config_classes import PreprocessingConfig, FeatureConfig, TrainTestSplitConfig, TrainConfig, TrainingParams
from model import Classifier
from utils import load_pickle_file, save_pickle_file

unittest.TestLoader.sortTestMethodsUsing = None


class TestUtilsForModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_column = 'condition'
        self.categorical = ['sex', 'cp', 'fbs', 'restecg', 'slope', 'exang', 'ca', 'thal']
        self.continual = ['age', 'trestbps', 'chol', 'thalach', 'slope']

        self.feature_conf = FeatureConfig(categorical_features=self.categorical,
                                          continual_features=self.continual,
                                          target_column=self.target_column)
        self.processing_conf = PreprocessingConfig()
        self.transformer = CustomTransformer(self.processing_conf, self.feature_conf)
        self.model = Classifier(TrainingParams())

    def test_save_model(self):
        my_model_path = 'model.pkl'
        save_pickle_file(my_model_path, self.model)
        self.assertTrue(os.path.exists(my_model_path))
        new_model = load_pickle_file(my_model_path)
        self.assertIsInstance(new_model, Classifier)
        os.remove(my_model_path)

    def test_save_transformer(self):
        my_transformer_path = 'transformer.pkl'
        save_pickle_file(my_transformer_path, self.transformer)
        self.assertTrue(os.path.exists(my_transformer_path))
        new_model = load_pickle_file(my_transformer_path)
        self.assertIsInstance(new_model, CustomTransformer)
        os.remove(my_transformer_path)


if __name__ == '__main__':
    unittest.main()
