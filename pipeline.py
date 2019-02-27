from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from submission import create_submission


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


def get_modeling_data():
    train = pd.read_csv("data/train_processed_2.csv")

    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_cat', 'Embarked_cat']
    label = ["Survived"]
    X = train[features]
    y = train[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    logreg_with_scaler = run_pipe_1()
    logreg = run_pipe_2()
    create_submission(logreg_with_scaler, "logreg_with_scaler")





