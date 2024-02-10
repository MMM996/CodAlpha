# This is the code for Task 4 Cod Alpha Internship
# Task 4: Disease Prediction from Medical Data
# The Code is written by Muhammad Mudassir Majeed
# The date is Jan-24.
# Dataset: https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning
#-----------------------------------------------------------------------------#

import pandas as pd
import seaborn as sns
import numpy as np

# Part 1: Load Dataset
data_train = pd.read_csv('Task 4\data 42 diseases\\training.csv')
data_test = pd.read_csv('Task 4\data 42 diseases\\testing.csv')

# Part 2: Preleminary EDA
data_train.head(20)
data_train.info()
data_train = data_train.drop(['Unnamed: 133'], axis = 1)

data_test.head(10)
data_test.info()

# Check Null Values
data_train.isnull().sum()
data_test.isnull().sum()

# Check Duplicates
data_train.duplicated().sum()
data_test.duplicated().sum()

# Part 3: Pre-processing

# Split into X and Y
X_train = data_train.drop(['prognosis'], axis =1)
Y_train = data_train['prognosis']
X_test = data_test.drop(['prognosis'], axis =1)
Y_test = data_test['prognosis']

# Encoding
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

Y_train = label.fit_transform(Y_train)
Y_test = label.fit_transform(Y_test) 

# Part 4: Model Training

# Logistic Regression
from sklearn.linear_model import LogisticRegression
log = LogisticRegression()
log.fit(X_train, Y_train)

log_predict = log.predict(X_test)

# Decision Trees
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
DT.fit(X_train,Y_train)

DT_predict = DT.predict(X_test)

# KNN
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(X_train, Y_train)

KNN_predict = KNN.predict(X_test)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50)
rf.fit(X_train, Y_train)

rf_predict = rf.predict(X_test)

# XGBoost
import xgboost as xgb
xg = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss")
xg.fit(X_train,Y_train)

xg_predict = xg.predict(X_test)


# Part 5: Model Evaluation
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# Accuracy Score
accuracy_logistic = accuracy_score(Y_test, log_predict)
accuracy_rf = accuracy_score(Y_test, rf_predict)
accuracy_xg = accuracy_score(Y_test, xg_predict)
accuracy_DT = accuracy_score(Y_test, DT_predict)
accuracy_KNN = accuracy_score(Y_test, KNN_predict)


# Precision Score
precision_logistic = precision_score(Y_test, log_predict, average = 'weighted')
precision_rf = precision_score(Y_test, rf_predict, average = 'weighted')
precision_xg = precision_score(Y_test, xg_predict, average = 'weighted')
precision_DT = precision_score(Y_test, DT_predict, average = 'weighted')
precision_KNN = precision_score(Y_test, KNN_predict, average = 'weighted')



# Recall Score
recall_logistic = recall_score(Y_test, log_predict, average = 'weighted')
recall_rf = recall_score(Y_test, rf_predict, average = 'weighted')
recall_xg = recall_score(Y_test, xg_predict, average = 'weighted')
recall_DT = recall_score(Y_test, DT_predict, average = 'weighted')
recall_KNN = recall_score(Y_test, KNN_predict, average = 'weighted')


# F1 Score
f1_logistic = f1_score(Y_test, log_predict, average = 'weighted')
f1_rf = f1_score(Y_test, rf_predict, average = 'weighted')
f1_xg = f1_score(Y_test, xg_predict, average = 'weighted')
f1_DT = f1_score(Y_test, DT_predict, average = 'weighted')
f1_KNN = f1_score(Y_test, KNN_predict, average = 'weighted')


# Use table to compare models
from tabulate import tabulate

data = [
    ['Logistic Regression',accuracy_logistic,precision_logistic, recall_logistic, f1_logistic],
    ['Random Forest',accuracy_rf,precision_rf, recall_rf, f1_rf],
    ['XGBoost',accuracy_xg,precision_xg, recall_xg, f1_xg],
    ['Decision Tree',accuracy_DT,precision_DT, recall_DT, f1_DT],
    ['KNN',accuracy_KNN,precision_KNN, recall_KNN, f1_KNN]
    ]

header = ['Model Name', 'Accuracy', 'Precision', 'Recall','F1-Score']

table = tabulate(data, header, tablefmt ='fancy_grid', floatfmt=('.0%', '.2%', '.2%', '.2%', '.2%'))
print(table)


from sklearn.metrics import confusion_matrix, classification_report

# Confusion Matrix
confusion_matrix_logistic = confusion_matrix(Y_test, log_predict)
confusion_matrix_rf = confusion_matrix(Y_test, rf_predict)
confusion_matrix_xg = confusion_matrix(Y_test, xg_predict)
confusion_matrix_DT = confusion_matrix(Y_test, DT_predict)
confusion_matrix_KNN = confusion_matrix(Y_test, KNN_predict)

# Classification Report
classification_report_logistic = classification_report(Y_test, log_predict)
classification_report_rf = classification_report(Y_test, rf_predict)
classification_report_xg = classification_report(Y_test, xg_predict)
classification_report_DT = classification_report(Y_test, DT_predict)
classification_report_KNN = classification_report(Y_test, KNN_predict)

print("Logistic Regression Confusion Matrix:\n", confusion_matrix_logistic)
print("Logistic Regression Classification Report:\n", classification_report_logistic)

print("Random Forest Confusion Matrix:\n", confusion_matrix_rf)
print("Random Forest Classification Report:\n", classification_report_rf)

print("XGBoost Confusion Matrix:\n", confusion_matrix_xg)
print("XGBoost Classification Report:\n", classification_report_xg)

print("Decision Tree Confusion Matrix:\n", confusion_matrix_DT)
print("Decision Tree Classification Report:\n", classification_report_DT)

print("K-Nearest Neighbors Confusion Matrix:\n", confusion_matrix_KNN)
print("K-Nearest Neighbors Classification Report:\n", classification_report_KNN)

