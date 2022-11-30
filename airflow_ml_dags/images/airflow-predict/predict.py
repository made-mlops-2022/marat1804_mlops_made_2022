import os
import pickle

import click
import pandas as pd


@click.command('predict')
@click.option('--input-dir', type=click.Path(),
              help='Path to preprocessed data')
@click.option('--model-dir', type=click.Path(),
              help='Path for model')
@click.option('--output-dir', type=click.Path(),
              help='Path for predictions')
def predict(input_dir, model_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    X = pd.read_csv(os.path.join(input_dir, 'train_data.csv'))

    with open(os.path.join(model_dir, 'logreg_model.pkl'), 'rb') as file:
        model = pickle.load(file)

    y_pred = model.predict(X)
    pd.DataFrame(y_pred).to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)


if __name__ == '__main__':
    predict()
