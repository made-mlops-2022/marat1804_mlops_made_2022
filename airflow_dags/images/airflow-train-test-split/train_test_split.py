import os
import click
import pandas as pd
from sklearn.model_selection import train_test_split


@click.command('split')
@click.option('--input-dir', type=click.Path(),
              help='Path to preprocessed data')
@click.option('--output-dir', type=click.Path(),
              help='Path for train and test data')
def train_test_split_data(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    X = pd.read_csv(os.path.join(input_dir, 'train_data.csv'))
    y = pd.read_csv(os.path.join(input_dir, 'target.csv'))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'),
                   index=False, sep=',', encoding='utf-8')
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'),
                  index=False, sep=',', encoding='utf-8')
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'),
                   index=False, sep=',', encoding='utf-8')
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'),
                  index=False, sep=',', encoding='utf-8')


if __name__ == '__main__':
    train_test_split_data()
