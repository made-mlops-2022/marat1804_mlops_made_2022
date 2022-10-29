import pickle

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class CustomTransformer:
    def __init__(self, process_params, feature_params):
        self.params = process_params
        self.feature_params = feature_params

        if self.params.process_continual:
            continual = Pipeline([('impute', SimpleImputer(strategy='mean')),
                                  ('scaler', StandardScaler())])
        else:
            continual = Pipeline([('impute', SimpleImputer(strategy='mean'))])

        if self.params.process_categorical:
            categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                                    ('encoder', OneHotEncoder())])
        else:
            categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent'))])

        self.transformer = ColumnTransformer([
            ('categorical', categorical, self.feature_params.categorical_features),
            ('continual', continual, self.feature_params.continual_features)
        ])

    def transform(self, data):
        return self.transformer.transform(data)

    def fit(self, data):
        self.transformer.fit(data)

    def save_transformer(self, path_to_model):
        with open(path_to_model, 'wb') as f:
            pickle.dump(self.transformer, f)
