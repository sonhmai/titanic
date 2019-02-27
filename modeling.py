import pandas as pd
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import logging


def run_xgb(X_train, X_test, y_train, y_test):
    """
    use xgboost to train and predict
    :return:
    """

    # train
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)  # y contains only 0,1
    acc = accuracy_score(y_test, y_pred)

    # accuracy for simple xgb model is 82%
    print(acc)
    return model


def run_logreg(X_train, X_test, y_train, y_test):
    """
    use logistic regression without feature scaling

    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    pred = logreg.predict(X_test)
    score = logreg.score(X_test, y_test)
    print(score)


def pipeline_processed_1():
    train = pd.read_csv("data/train_processed.csv")
    test = pd.read_csv("data/test_processed.csv")

    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_cat', 'Embarked_cat']
    label = ["Survived"]
    X = train[features]
    y = train[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    xgb_model = run_xgb(X_train, X_test, y_train, y_test)
    run_logreg(X_train, X_test, y_train, y_test)

    # predict test set
    pred = xgb_model.predict(test[features])
    test['Survived'] = pred
    sub = test[["PassengerId", "Survived"]]
    print(len(sub))
    sub.to_csv("data/submission.csv", index=False)

    sample = pd.read_csv("data/gender_submission.csv")
    print(len(sample))

    test_original = pd.read_csv("data/test.csv")
    print(len(test_original))


def pipeline_processed_2():
    """
    using data of data_process_2
    :return:
    """
    train = pd.read_csv("data/train_processed_2.csv")
    test = pd.read_csv("data/test_processed_2.csv")

    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_cat', 'Embarked_cat']
    label = ["Survived"]
    X = train[features]
    y = train[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    xgb_model = run_xgb(X_train, X_test, y_train, y_test)
    run_logreg(X_train, X_test, y_train, y_test)

    # predict test set
    pred = xgb_model.predict(test[features])
    test['Survived'] = pred
    sub = test[["PassengerId", "Survived"]]
    print(len(sub))
    sub.to_csv("data/submission.csv", index=False)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)

    pipeline_processed_2()


