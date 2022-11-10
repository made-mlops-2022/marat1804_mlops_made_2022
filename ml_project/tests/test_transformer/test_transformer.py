import os
import sys
import unittest


PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH, 'src'
)
sys.path.append(SOURCE_PATH)

from data import read_csv_data, split_data, get_x_and_y
from features import CustomTransformer
from config_classes import PreprocessingConfig, FeatureConfig, TrainTestSplitConfig

unittest.TestLoader.sortTestMethodsUsing = None


class TestCustomTransformer(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_file = 'tests/data/synthetic_data.csv'
        self.target_column = 'condition'
        self.data = read_csv_data(self.input_file)
        X, y = get_x_and_y(self.data, self.target_column)
        self.X_train, self.X_test, self.y_train, self.y_test = split_data(X, y, TrainTestSplitConfig())

        self.categorical = ['sex', 'cp', 'fbs', 'restecg', 'slope', 'exang', 'ca', 'thal']
        self.continual = ['age', 'trestbps', 'chol', 'thalach', 'slope']

        self.feature_conf = FeatureConfig(categorical_features=self.categorical,
                                          continual_features=self.continual,
                                          target_column=self.target_column)

    def test_default_transformer(self):
        processing_conf = PreprocessingConfig()
        transformer = CustomTransformer(processing_conf, self.feature_conf)
        self.assertEqual(len(transformer.transformer.transformers), 2)
        transformer.fit(self.X_train)
        new_X = transformer.transform(self.X_train)
        self.assertEqual(new_X.shape, (112, 28))
        new_X_test = transformer.transform(self.X_test)
        self.assertEqual(new_X_test.shape, (38, 28))

    def test_without_cat_feats(self):
        processing_conf = PreprocessingConfig(
            process_categorical=False,
            process_continual=True
        )
        transformer = CustomTransformer(processing_conf, self.feature_conf)
        self.assertEqual(len(transformer.transformer.transformers), 2)
        self.assertEqual(max(self.X_train.age), 75)
        self.assertEqual(max(self.X_train.chol), 408)
        transformer.fit(self.X_train)
        new_X = transformer.transform(self.X_train)
        self.assertNotEqual(max(new_X[:, len(self.categorical) + self.continual.index('age')]), 75)
        self.assertNotEqual(max(new_X[:, len(self.categorical) + self.continual.index('chol')]), 408)
        self.assertEqual(new_X.shape, (112, 13))
        new_X_test = transformer.transform(self.X_test)
        self.assertEqual(new_X_test.shape, (38, 13))

    def test_without_cont_feats(self):
        processing_conf = PreprocessingConfig(
            process_categorical=True,
            process_continual=False
        )
        transformer = CustomTransformer(processing_conf, self.feature_conf)
        self.assertEqual(len(transformer.transformer.transformers), 2)
        self.assertEqual(max(self.X_train.age), 75)
        self.assertEqual(max(self.X_train.chol), 408)
        transformer.fit(self.X_train)
        new_X = transformer.transform(self.X_train)
        self.assertEqual(max(new_X[:, 23]), 75)
        self.assertEqual(max(new_X[:, 25]), 408)
        self.assertEqual(new_X.shape, (112, 28))
        new_X_test = transformer.transform(self.X_test)
        self.assertEqual(new_X_test.shape, (38, 28))

    def test_without_all(self):
        processing_conf = PreprocessingConfig(
            process_categorical=False,
            process_continual=False
        )
        transformer = CustomTransformer(processing_conf, self.feature_conf)
        self.assertEqual(len(transformer.transformer.transformers), 2)
        self.assertEqual(max(self.X_train.trestbps), 181)
        self.assertEqual(max(self.X_train.slope), 2)
        transformer.fit(self.X_train)
        new_X = transformer.transform(self.X_train)
        self.assertEqual(max(new_X[:, len(self.categorical) + self.continual.index('trestbps')]), 181.)
        self.assertEqual(max(new_X[:, len(self.categorical) + self.continual.index('slope')]), 2)
        self.assertEqual(new_X.shape, (112, 13))
        new_X_test = transformer.transform(self.X_test)
        self.assertEqual(new_X_test.shape, (38, 13))


if __name__ == '__main__':
    unittest.main()
