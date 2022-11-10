import os
import sys
import unittest

import pandas as pd

PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH, 'src'
)
sys.path.append(SOURCE_PATH)

from data import read_csv_data, split_data, get_x_and_y
from config_classes import TrainTestSplitConfig

unittest.TestLoader.sortTestMethodsUsing = None


class TestDataManipulationModule(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_file = 'tests/data/synthetic_data.csv'
        self.target_column = 'condition'
        self.input_size = (150, 14)

        self.data = read_csv_data(self.input_file)

    def test_read_data(self):
        self.assertEqual(self.data.shape, (150, 14))
        self.assertIn(self.target_column, self.data.columns)
        for col in ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']:
            self.assertIn(col, self.data.columns)

    def test_get_x_and_y(self):
        X, y = get_x_and_y(self.data, self.target_column)
        self.assertEqual(X.shape, (150, 13))
        self.assertEqual(y.shape, (150, ))
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

    def test_split_data(self):
        X, y = get_x_and_y(self.data, self.target_column)
        X_train, X_test, y_train, y_test = split_data(X, y, TrainTestSplitConfig())

        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertIsInstance(y_test, pd.Series)

        self.assertEqual(X_train.shape, (112, 13))
        self.assertEqual(y_train.shape, (112,))
        self.assertEqual(X_test.shape, (38, 13))
        self.assertEqual(y_test.shape, (38,))

    def test_another_split_data(self):
        X, y = get_x_and_y(self.data, self.target_column)
        X_train, X_test, y_train, y_test = split_data(X, y, TrainTestSplitConfig(test_size=0.2))

        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(X_test, pd.DataFrame)
        self.assertIsInstance(y_test, pd.Series)

        self.assertEqual(X_train.shape, (120, 13))
        self.assertEqual(y_train.shape, (120,))
        self.assertEqual(X_test.shape, (30, 13))
        self.assertEqual(y_test.shape, (30,))


if __name__ == '__main__':
    unittest.main()
