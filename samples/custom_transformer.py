from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline


# custom transformer from a function
def all_but_first_column(X):
    return X[:, 1:]


pipe = Pipeline([
    FunctionTransformer(all_but_first_column),
])

# ----------------
preprocessing = Pipeline([
    ('make a copy', Replicator()),
    ('extract title', TitleExtractor()),
])

