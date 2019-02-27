from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV


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


