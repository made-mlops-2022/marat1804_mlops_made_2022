import click
import shutil
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@click.command('preprocess')
@click.option('--input-dir', type=click.Path(),
              help='Path for train data dir')
@click.option('--output-dir', type=click.Path(),
              help='Path for preprocessed data')
def data_preprocess(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(os.path.join(input_dir, 'data.csv'))

    continual = Pipeline([('impute', SimpleImputer(strategy='mean')),
                          ('scaler', StandardScaler())])

    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                            ('encoder', OneHotEncoder())])
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    continual_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    transformer = ColumnTransformer([
        ('categorical', categorical, categorical_features),
        ('continual', continual, continual_features)
    ])
    transformer.fit(df)
    X = transformer.transform(df)
    processed_data = pd.DataFrame(X)
    processed_data.to_csv(os.path.join(output_dir, 'train_data.csv'), index=False)

    shutil.copyfile(os.path.join(input_dir, 'target.csv'),
                    os.path.join(output_dir, 'target.csv'))


if __name__ == '__main__':
    data_preprocess()
