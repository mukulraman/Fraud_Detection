import pandas as pd

def data_cleaning(df):
    df.drop(columns=['nameOrig', 'nameDest'], inplace=True,axis=1)
    df.replace([''], pd.NA, inplace=True)
    df['newbalanceOrig'] = pd.to_numeric(df['newbalanceOrig'], errors='coerce')
    df['oldbalanceDest'] = pd.to_numeric(df['oldbalanceDest'], errors='coerce')
    df['step'] = pd.to_numeric(df['step'], errors='coerce')
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['oldbalanceOrg'] = pd.to_numeric(df['oldbalanceOrg'], errors='coerce')
    df['newbalanceDest'] = pd.to_numeric(df['newbalanceDest'], errors='coerce')
    df['isFlaggedFraud'] = pd.to_numeric(df['isFlaggedFraud'], errors='coerce')
    df['isFraud'] = pd.to_numeric(df['isFraud'], errors='coerce')

    Not_Nan = ['newbalanceOrig', 'oldbalanceDest']

    for mean_column in Not_Nan:
        mean_value = df[mean_column].mean()
        df[mean_column].fillna(mean_value, inplace=True)

    Not_Null = ['type']

    for column in Not_Null:
        mode_value = df[column].mode()[0]
        df[column].fillna(mode_value, inplace=True)

    df = pd.get_dummies(df, columns=['type'], dtype=int)
    return df