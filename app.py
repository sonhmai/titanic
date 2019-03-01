import os
from flask import Flask
from xgb_api import api_predict


app = Flask(__name__)


@app.route('/')
def hello():
    return 'hello, world!'


@app.route('/predict')
def predict():
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
    return res
