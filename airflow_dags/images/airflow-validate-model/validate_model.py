import json
import os
import pickle

import click
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


@click.command('validate')
@click.option('--input-dir', type=click.Path(),
              help='Path to split data')
@click.option('--model-dir', type=click.Path(),
              help='Path to model')
@click.option('--output-dir', type=click.Path(),
              help='Path for metrics')
def validate_model(input_dir, model_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    X = pd.read_csv(os.path.join(input_dir, 'X_test.csv'))
    y = pd.read_csv(os.path.join(input_dir, 'y_test.csv'))

    with open(os.path.join(model_dir, 'logreg_model.pkl'), 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(X)

    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'f1-score': f1_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'precision': precision_score(y, y_pred),
    }

    with open(os.path.join(output_dir, 'metric.json'), 'w') as file:
        json.dump(metrics, file)


if __name__ == '__main__':
    validate_model()
