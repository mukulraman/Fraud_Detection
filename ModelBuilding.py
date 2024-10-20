from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
from  xgboost import XGBClassifier
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

def train_test_split_and_features(train_df, test_df ):
    y_train = train_df['isFraud']
    x_train = train_df.drop(['isFraud'], axis=1)
    y_test = test_df['isFraud']
    x_test = test_df.drop(['isFraud'], axis=1)
    feature_columns = x_train.columns

    scaler=MinMaxScaler()
    x_train= scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_train = pd.DataFrame(x_train, columns=feature_columns)
    x_test = pd.DataFrame(x_test, columns=feature_columns)

    features=list(x_train.columns)
    return x_train, x_test, y_train, y_test,features

def fit_and_evaluate_model(x_train, x_test, y_train, y_test):
    print("Original class distribution:", Counter(y_train))
    smote = SMOTE(sampling_strategy={0:3996619,1:1000000})  # Control the number of samples

    X_resampled, y_resampled = smote.fit_resample(x_train, y_train)
    print("Resampled class distribution:", Counter(y_resampled))
    xgb = XGBClassifier(
        scale_pos_weight=len(y_train) / (2 * (y_train.sum())))  # Dynamic weighting based on class distribution
    xgb.fit(X_resampled, y_resampled)
    xgb_predict = xgb.predict(x_test)
    xgb_conf_matrix = confusion_matrix(y_test, xgb_predict)
    xgb_acc_score = accuracy_score(y_test, xgb_predict)
    print("confussion matrix")
    print(xgb_conf_matrix)
    print("\n")
    print("Accuracy of XGBoost:",xgb_acc_score*100,'\n')
    print(classification_report(y_test,xgb_predict))
    return xgb