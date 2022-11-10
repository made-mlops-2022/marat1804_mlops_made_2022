import os
import sys
import unittest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

PROJECT_PATH = os.getcwd()
SOURCE_PATH = os.path.join(
    PROJECT_PATH, 'src'
)
sys.path.append(SOURCE_PATH)

from data import read_csv_data, split_data, get_x_and_y
from features import CustomTransformer
from config_classes import PreprocessingConfig, FeatureConfig, TrainTestSplitConfig, TrainConfig, TrainingParams
from model import Classifier

unittest.TestLoader.sortTestMethodsUsing = None


class TestClassifierModel(unittest.TestCase):
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
        self.processing_conf = PreprocessingConfig()
        self.transformer = CustomTransformer(self.processing_conf, self.feature_conf)
        self.transformer.fit(self.X_train)

    def test_log_reg_with_grid_search(self):
        config = TrainingParams(
            model_type='LogisticRegression',
            grid_search=True
        )
        model = Classifier(config)
        self.assertIsInstance(model.model, LogisticRegression)
        X_train = self.transformer.transform(self.X_train)
        X_test = self.transformer.transform(self.X_test)
        model.fit(X_train, self.y_train)
        self.assertIn('recall_val', model.model_best_score_)
        y_pred = model.predict(X_test)
        self.assertEqual(y_pred.shape, (38,))
        metrics = model.evaluate_model(self.y_test, y_pred)
        self.assertIn('accuracy', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('roc_auc', metrics)

    def test_knn_without_grid_search(self):
        config = TrainingParams(
            model_type='KNeighborsClassifier',
            grid_search=False
        )
        model = Classifier(config)
        self.assertIsInstance(model.model, KNeighborsClassifier)
        X_train = self.transformer.transform(self.X_train)
        X_test = self.transformer.transform(self.X_test)
        model.fit(X_train, self.y_train)
        self.assertIsNone(model.model_best_score_)
        self.assertIsNone(model.model_best_params)
        y_pred = model.predict(X_test)
        self.assertEqual(y_pred.shape, (38,))
        metrics = model.evaluate_model(self.y_test, y_pred)
        self.assertIn('accuracy', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('f1_score', metrics)
        self.assertIn('roc_auc', metrics)

    def test_not_implemented(self):
        config = TrainingParams(
            model_type='SGDClassifier',
            grid_search=False
        )
        with self.assertRaises(NotImplementedError) as context:
            model = Classifier(config)


if __name__ == '__main__':
    unittest.main()
