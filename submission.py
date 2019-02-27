import pandas as pd


def create_submission(model, name):
    test = pd.read_csv("data/test_processed_2.csv")
    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_cat', 'Embarked_cat']
    pred = model.predict(test[features])
    test['Survived'] = pred
    sub = test[["PassengerId", "Survived"]]
    print(len(sub))
    path = "data/submission_" + name + ".csv"
    sub.to_csv(path, index=False)

