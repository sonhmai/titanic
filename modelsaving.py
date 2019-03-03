from sklearn.externals import joblib


def save_model(model, name):
    path = 'models/' + name + '.joblib'
    joblib.dump(model, path)

