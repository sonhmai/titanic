from xgb_api import api_predict


def test_convert_json_to_df():
    data = {
        'PassengerId': [1],
        'Pclass': [3],
        'Age': [35],
        'Sex': ['male'],
        'SibSp': [1],
        'Parch': [0],
        'Fare': [12.75],
        'Sex_cat': [1],
        'Embarked_cat': [2],
    }
    res = api_predict(data)
    assert res == '{"PassengerId": [1], "Survived": [0]}'


if __name__ == '__main__':
    test_convert_json_to_df()


