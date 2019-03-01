import xgboost as xgb
import pandas as pd
import pickle
import json


pipe = pickle.load(open('data/xgb_bestcv.pkl', 'rb'))


def api_predict(data):
    """
    this function takes a person data in json format and predicts
    whether that person survived in Titanic incidence.

    :param data: JSON data structure containing person info
    :return:
    """
    df = pd.DataFrame.from_dict(data, orient='columns')

    # re-order columns to match order in train set
    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_cat', 'Embarked_cat']

    # predict
    res = pipe.predict(df[features])
    df['Survived'] = res

    # return result
    res_cols = ['PassengerId', 'Survived']
    res = df[res_cols].to_dict(orient='list')
    print(res)
    return json.dumps(res)
