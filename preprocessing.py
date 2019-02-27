import pandas as pd
from sklearn.model_selection import train_test_split


def prepare_data_dropna(input_path):
    """
    cannot drop na as test data will be missing records, submission requires full test data
    must impute the missing data

    :param input_path:
    :return:
    """
    df = pd.read_csv(input_path)

    # drop Name as it is not relevant for prediction
    df.drop("Name", axis=1, inplace=True)
    # drop Cabin as it has 687/891 missing values
    df.drop("Cabin", axis=1, inplace=True)
    # drop rows with null Age (177/891) and null Embarked (2)
    df.dropna(inplace=True)
    # after dropping, train should have 891-177-2(null Embarked) rows
    # ticket: removed, contains 681 distinct values
    df.drop("Ticket", axis=1, inplace=True)

    # encode sex
    df["Sex_cat"] = df["Sex"].astype("category")
    df["Sex_cat"] = df["Sex_cat"].cat.codes
    # encode embarked, 3 distinct values
    df["Embarked_cat"] = df["Embarked"].astype("category")
    df["Embarked_cat"] = df["Embarked_cat"].cat.codes

    print("length of dataframe before saving:", len(df))
    output_path = input_path.split('.')[0] + "_processed." + input_path.split('.')[1]
    df.to_csv(output_path)


def prepare_data_2(input_path):
    """
    impute age in test as this is an important feature
    impute fare as well as fare might be related to room location on the ship
    - train count 891, age 714 non-null, cabin 204, embarked 889
    - test count 418, age 332 non-null, cabin 91, fare 417

    :return:
    """
    df = pd.read_csv(input_path)
    # drop Name as it is not relevant for prediction
    df.drop("Name", axis=1, inplace=True)
    # drop Cabin as it has 687/891 missing values
    df.drop("Cabin", axis=1, inplace=True)
    # AGE - impute missing age with median
    age_median = df["Age"].median(skipna=True)
    df["Age"].fillna(age_median, inplace=True)
    # FARE - impute missing fare with median
    fare_median = df["Fare"].median(skipna=True)
    df["Fare"].fillna(fare_median, inplace=True)
    # EMBARKED - impute with most common values, only missing in train
    most_frequent_emb = df["Embarked"].value_counts().idxmax()
    df["Embarked"].fillna(most_frequent_emb, inplace=True)

    # encode sex
    df["Sex_cat"] = df["Sex"].astype("category")
    df["Sex_cat"] = df["Sex_cat"].cat.codes
    # encode embarked, 3 distinct values
    df["Embarked_cat"] = df["Embarked"].astype("category")
    df["Embarked_cat"] = df["Embarked_cat"].cat.codes

    print(df.info())
    output_path = input_path.split('.')[0] + "_processed_2." + input_path.split('.')[1]
    df.to_csv(output_path)


def get_modeling_data():
    train = pd.read_csv("data/train_processed_2.csv")

    features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_cat', 'Embarked_cat']
    label = ["Survived"]
    X = train[features]
    y = train[label]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    train_path = "data/train.csv"
    test_path = "data/test.csv"

    prepare_data_2(train_path)
    prepare_data_2(test_path)