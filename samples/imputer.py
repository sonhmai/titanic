from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.impute import SimpleImputer


# assume  train_X, val_X, train_y, and val_y are available
train_X = 0
train_y = 0
val_X = 0
val_y = 0

# have a name for every step in pipeline so we can call a step and set params
pipe = Pipeline([
    ('inputer', Imputer()),
    ('xgbrg', XGBRegressor())
])

('impute', Imputer(strategy="median"))

# missing_values : number, string, np.nan (default) or None
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = SimpleImputer(missing_values=np.nan, strategy='median')
imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imp = SimpleImputer(missing_values=np.nan, strategy='constant')


def imputer_test(input_path):
    df = pd.read_csv(input_path)

    # impute age
    imp = SimpleImputer(strategy='median')
    df["Age"] = imp.fit_transform(df[["Age"]]).ravel()
    print(df.info())


if __name__ == '__main__':
    train_path = "data/train.csv"
    test_path = "data/test.csv"

    imputer_test(train_path)
    imputer_test(test_path)