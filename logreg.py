from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from preprocessing import get_modeling_data


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


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_modeling_data()
    logreg = LogReg(X_train, X_test, y_train, y_test)
    logreg.run_naive()
    logreg.run_scaler()