from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import warnings

from preprocessing import get_modeling_data
from submission import create_submission


class LogReg:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def run_scaler(self):
        pipe = Pipeline([
            ('scl', StandardScaler()),
            ('clf', LogisticRegression(random_state=42))
        ])
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        print(acc)
        return pipe

    def run_naive(self):
        pipe = Pipeline([
            ('clf', LogisticRegression(random_state=42))
        ])
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        print(acc)
        return pipe

    def run_special_1(self):
        """
        logistic regression with params tuning
        should be able to achieve > 80% accuracy
        :return:
        """
        pipe = Pipeline([
            ('scl', StandardScaler()),
            ('clf', LogisticRegression(
                random_state=42,
            ))
        ])
        param_grid = {
            'clf__class_weight': ['balanced', None],
            'clf__penalty': ['l2'],
            'clf__C': [0.1, 0.5, 1.0, 1.5, 2.0],
            # C: (float) inverse regularization strength, default 1.0
            'clf__solver': ['liblinear', 'sag', 'newton-cg', 'saga']
        }
        search_cv = GridSearchCV(pipe, cv=4, param_grid=param_grid)
        search_cv.fit(X_train, y_train)

        print(search_cv.best_params_)
        print(search_cv.best_estimator_)
        print("best cv score:", search_cv.best_score_)
        pipe = search_cv.best_estimator_
        acc = pipe.score(X_test, y_test)
        print("test acc:", acc)
        create_submission(pipe, "logreg-best-cv")
        return pipe


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    X_train, X_test, y_train, y_test = get_modeling_data()
    model = LogReg(X_train, X_test, y_train, y_test)
    model.run_naive()
    model.run_scaler()
    model.run_special_1()
