import pandas as pd


pd.set_option('display.max_columns', None)
test = pd.read_csv('../data/test_processed_2.csv')
print(test.head())

