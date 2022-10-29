import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier


class Classifier:
    def __init__(self, train_params):
        if train_params.model_type == 'KNeighborsClassifier':
            self.model = KNeighborsClassifier()
            self.param_grid = {'n_neighbors': np.arange(1, 15),
                               'weights': ['uniform', 'distance'],
                               'metric': ['minkowski', 'manhattan']}
        elif train_params.model_type == 'LogisticRegression':
            self.model = LogisticRegression(random_state=train_params.random_state, solver='liblinear')
            self.param_grid = {'C': np.logspace(-3, 3, 10),
                               'penalty': ['l1', 'l2']}
        else:
            raise NotImplementedError()
        self.train_params = train_params
        self.model_best_params = None
        self.model_best_score_ = None

    def fit(self, X_train, y_train):
        if self.train_params.grid_search:
            model_grid_search = GridSearchCV(self.model, self.param_grid, scoring='f1', cv=5)
            model_grid_search.fit(X_train, y_train)

            self.model = model_grid_search.best_estimator_
            self.model_best_params = model_grid_search.best_params_
            self.model_best_score_ = {'f1_val': model_grid_search.best_score_}
        else:
            self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def save_model(self, path_to_model):
        with open(path_to_model, 'wb') as f:
            pickle.dump(self.model, f)

    def evaluate_model(self, y_pred, y):
        return {
            'accuracy': accuracy_score(y_pred, y),
            'recall': recall_score(y_pred, y),
            'precision': precision_score(y_pred, y),
            'f1_score': f1_score(y_pred, y),
            'roc_auc': roc_auc_score(y_pred, y)
        }
