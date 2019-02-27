from sklearn import svm
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler
from sklearn.pipeline import Pipeline

from preprocessing import get_modeling_data


class SVM:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def run_naive(self):
        pipe = Pipeline([
            ('clf', svm.SVC(kernel='linear'))
        ])
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        print(acc)
        return pipe

    def run_standard_scaler(self):
        pipe = Pipeline([
            ('scl', StandardScaler()),
            ('clf', svm.SVC(kernel='linear'))
        ])
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        print(acc)
        return

    def run_norm(self):
        pipe = Pipeline([
            ('scl', Normalizer()),
            ('clf', svm.SVC(kernel='linear'))
        ])
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        print(acc)
        return pipe

    def run_minmax(self):
        pipe = Pipeline([
            ('scl', MinMaxScaler()),
            ('clf', svm.SVC(kernel='linear'))
        ])
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        print(acc)
        return pipe


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_modeling_data()
    svm_model = SVM(X_train, X_test, y_train, y_test)
    svm_model.run_naive()  # acc 0.791
    # acc does not increase by applying scaler, maybe because svm has instrinsic scaler
    svm_model.run_standard_scaler()  # acc 0.791
    svm_model.run_norm()  # acc 0.671
    # it makes no sense to normalize over all features as they have difference magnitude range
    svm_model.run_minmax()  # acc 0.791

