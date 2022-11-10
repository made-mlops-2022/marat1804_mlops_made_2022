import pandas as pd
from sklearn.model_selection import train_test_split


def read_csv_data(path, sep=','):
    df = pd.read_csv(path, sep=sep)
    return df


def get_x_and_y(df, target_col):
    X = df.drop(columns=[target_col])
    y = df.loc[:, target_col]
    return X, y


def split_data(X, y, params):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params.test_size,
        random_state=params.random_state
    )
    return X_train, X_test, y_train, y_test
