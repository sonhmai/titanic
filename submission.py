import pandas as pd
import subprocess
import pprint as pp


def create_submission(model, name):
    test = pd.read_csv("data/test_processed_2.csv")
    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_cat', 'Embarked_cat']
    pred = model.predict(test[features])
    test['Survived'] = pred
    submitted_df = test[["PassengerId", "Survived"]]
    path = "data/submission_" + name + ".csv"
    submitted_df.to_csv(path, index=False)
    post_kaggle(path, name)


def post_kaggle(path, message):
    command = f'kaggle competitions submit -c titanic -f {path} -m "{message}"'
    # command = "dir"
    bytes_output = subprocess.check_output(command, shell=True)
    pp.pprint(str(bytes_output, encoding='UTF-8'))


if __name__ == '__main__':
    path = "data/submission_xgb.csv"
    message = "xgb-submitted-by-python"
    post_kaggle(path, message)

