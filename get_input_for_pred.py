from preprocessing import get_modeling_data
import warnings
import json


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_modeling_data()
    sample = X_test.iloc[5:6].values.tolist()
    print(json.dumps(sample))
    with open('pred-input/input.json', 'w') as outfile:
        json.dump(sample, outfile)
