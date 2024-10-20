import pandas as pd
from collections import Counter
import joblib

# Read the Heart Disease Training Data from output_data.csv file in data folder
# If output_data.csv file in data folder doesn't exist, then first run the DB_to_CSV.py script
df = pd.read_csv('data/output_data.csv')

from DataCleaning import data_cleaning
from ModelBuilding import(
    train_test_split_and_features,
    fit_and_evaluate_model
)

df = data_cleaning(df)

train_df = df.iloc[:4000000]
test_df = df.iloc[4000000:5000000]
live_df = df.iloc[5000000:]

x_train, x_test, y_train, y_test,features=train_test_split_and_features(train_df, test_df)

print(df.head())
print(df.isnull().sum())
print(train_df.dtypes)

model = fit_and_evaluate_model(x_train, x_test, y_train, y_test)

print(Counter(y_train))

joblib.dump(model,'models\Model_Classifier_Fraud.pkl')
joblib.dump(features,'models\Features_Columns.pkl')