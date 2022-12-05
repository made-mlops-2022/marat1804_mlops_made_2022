import os
import pickle

import click
import pandas as pd
from sklearn.linear_model import LogisticRegression


@click.command('train')
@click.option('--input-dir', type=click.Path(),
              help='Path for train data')
@click.option('--output-dir', type=click.Path(),
              help='Path for learned model')
def train_model(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X = pd.read_csv(os.path.join(input_dir, 'X_train.csv'))
    y = pd.read_csv(os.path.join(input_dir, 'y_train.csv'))

    model = LogisticRegression(random_state=42, solver='liblinear',
                               C=0.72, penalty='l2')
    model.fit(X, y)
    with open(os.path.join(output_dir, f'logreg_model.pkl'), 'wb') as file:
        pickle.dump(model, file)


if __name__ == '__main__':
    train_model()
