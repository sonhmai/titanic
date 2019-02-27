from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from submission import create_submission
from preprocessing import get_modeling_data


def run_pipe_1():
    X_train, X_test, y_train, y_test = get_modeling_data()
    pipe = Pipeline([
        ('scl', StandardScaler()),
        ('clf', LogisticRegression(random_state=42))
    ])
    pipe.fit(X_train, y_train)
    acc = pipe.score(X_test, y_test)
    print(acc)
    return pipe


def run_pipe_2():
    X_train, X_test, y_train, y_test = get_modeling_data()
    pipe = Pipeline([
        ('clf', LogisticRegression(random_state=42))
    ])
    pipe.fit(X_train, y_train)
    acc = pipe.score(X_test, y_test)
    print(acc)
    return pipe





if __name__ == '__main__':
    logreg_with_scaler = run_pipe_1()
    logreg = run_pipe_2()
    create_submission(logreg_with_scaler, "logreg_with_scaler")





