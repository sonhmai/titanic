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

# gridsearchCV for tuning model
param_grid = {
    'xgb__n_estimators': [10, 50, 100, 500],
    'xgbrg__learning_rate': [0.1, 0.5, 1],
}

fit_params = {
    'xgbrg__eval_set': [(val_X, val_y)],
    'xgbrg_early_stopping_rounds': 10,
    'xgbrg__verbose': False
}

searchCV = GridSearchCV(pipe, cv=5, param_grid=param_grid, fit_params=fit_params)
searchCV.fit(train_X, train_y)

print(searchCV.best_params_)
print(searchCV.cv_results_['mean_train_score'])
print(searchCV.cv_results_['mean_test_score'])

print(searchCV.cv_results_['mean_train_score'].mean())
print(searchCV.cv_results_['mean_test_score'].mean())


