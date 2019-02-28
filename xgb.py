from sklearn import svm
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import warnings

from preprocessing import get_modeling_data
from submission import create_submission

class XGB:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        # TODO how to add a transformer to a pipe

    def run_naive(self):
        pipe = Pipeline([
            ('clf', XGBClassifier())
        ])
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        print(acc)
        return pipe

    def run_standard_scaler(self):
        pipe = Pipeline([
            ('scl', StandardScaler()),
            ('clf', XGBClassifier())
        ])
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        print(acc)
        return

    def run_norm(self):
        pipe = Pipeline([
            ('scl', Normalizer()),
            ('clf', XGBClassifier())
        ])
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        print(acc)
        return pipe

    def run_minmax(self):
        pipe = Pipeline([
            ('scl', MinMaxScaler()),
            ('clf', XGBClassifier())
        ])
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        print(acc)
        return pipe

    def run_cv(self):
        pipe = Pipeline([
            ('clf', XGBClassifier())
        ])
        param_grid = {
            'clf__n_estimators': [10, 50, 100, 500],
            'clf__learning_rate': [0.1, 0.5, 1],
            'clf__max_depth': [3, 6, 10, 15]
        }

        fit_params = {
            'clf__eval_set': [(X_test, y_test)],
            'clf__early_stopping_rounds': 5,
            'clf__verbose': False
        }
        pipe.fit(X_train, y_train)
        search_cv = GridSearchCV(pipe, cv=5, param_grid=param_grid, fit_params=fit_params)
        search_cv.fit(X_train, y_train)

        print(search_cv.best_params_)
        print(search_cv.cv_results_['mean_train_score'])
        print(search_cv.cv_results_['mean_test_score'])

        print(search_cv.cv_results_['mean_train_score'].mean())
        print(search_cv.cv_results_['mean_test_score'].mean())

    def run_best(self):
        pipe = Pipeline([
            ('clf', XGBClassifier(n_estimators=10, learning_rate=0.5, max_depth=3))
        ])
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        print(acc)
        return pipe


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    X_train, X_test, y_train, y_test = get_modeling_data()
    model = XGB(X_train, X_test, y_train, y_test)
    model.run_naive()  # acc 0.791
    # acc does not increase by applying scaler, maybe because svm has instrinsic scaler
    model.run_standard_scaler()  # acc 0.791
    model.run_norm()  # acc 0.671
    # it makes no sense to normalize over all features as they have difference magnitude range
    model.run_minmax()  # acc 0.791
    model.run_cv()
    print("best")
    pipe = model.run_best()
    # create_submission(pipe, "xgb")

