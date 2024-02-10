# This is the code for Task 1 Cod Alpha Internship
# Task 1: Credit Scoring Model
# The Code is written by Muhammad Mudassir Majeed
# The date is Jan-24.
# Dataset: https://www.kaggle.com/datasets/parisrohan/credit-score-classification
#-----------------------------------------------------------------------------#

import pandas as pd
import seaborn as sns
import numpy as np


# Part 1: Load data
data_train = pd.read_csv('Task 1\data\\train.csv')

# Part 2: Preleminary EDA
pd.set_option('display.max_columns', None)
data_train.head(20)
data_train.info()

# Check Null Values
data_train.isnull().sum()

# Check Duplicates
data_train.duplicated().sum()  # No Duplicates

# Check output class balance
data_train['Credit_Score'].value_counts()
sns.countplot(x= data_train['Credit_Score'], label = 'Counts')
# We have class imbalance

# Part 3: Preprocessing
data_train['Customer_ID'].nunique()
data_train['Payment_of_Min_Amount'].nunique()

# Drop Unnecessary Columns
# Domain Knowledge can help a lot in this case
drop_col = ['ID','Name','SSN']

data_train = data_train.drop(drop_col,axis=1)
data_train.info()

# We will deal with Numeric and String columns seperately

# We can observe from our data that we have outlier and Null values.
# From data inspection we can see that for Columns:
    # Age, Annual_Income,Monthly_Inhand_Salary, Num_Bank Accounts,
    # Num_Credit Card, Interest_Rate, Num_of_Loan, Num_Credit_Inquiries
    # Outstanding_Debt, Total_EMI per month
        # We should cater for Null and Outliers using Mode value for each customer
        # Fisrt convert each Column to Numeric then proceed
    
    # Num_of_Delayed_Payment, changed_Credit_Limit, Credit_Utilization_Ratio,
    # Amount_invested_monthly, Monthly_Balance
        # We Should cater for Null and Outliers using Medain Values
        # First convert each column to Numeric
        
        
# For Columns, Annual Income, Age, Monthly Inhand, Occupation, a good estiamte
# would be to replace problematic values with Mode for a specficic cutomer ID

# First Identify Unique Customers
data_train.info()
customer_unique = data_train['Customer_ID']
customer_unique = customer_unique.drop_duplicates()

# Convert Each Numeric Data column from Object to Numeric to remove any error strings
# A Better Appraoch would be to use regex and look for specfic strings to remove

# For Mode Columns
mode_col = ['Age','Annual_Income','Monthly_Inhand_Salary','Num_Bank_Accounts','Num_Credit_Card',
            'Interest_Rate','Num_of_Loan','Num_Credit_Inquiries','Outstanding_Debt','Total_EMI_per_month']

for column in mode_col:
    data_train[column] = pd.to_numeric(data_train[column], errors='coerce')
    

# For Median Columns
median_col = ['Num_of_Delayed_Payment','Changed_Credit_Limit','Credit_Utilization_Ratio',
              'Amount_invested_monthly','Monthly_Balance']

for column in median_col:
    data_train[column] = pd.to_numeric(data_train[column], errors='coerce')
  
data_train.info()
data_train.head(20)

mode_values = {}
median_values = {}
outlier_mode = {}
outlier_median = {}
count = 0

# Calculate values and apply corrections
for ID in customer_unique:
    
    count = count + 1
    print(count, end='\n')
    print(f'Currently Working for Customer: {ID}')
    
    customer_data = data_train[data_train['Customer_ID'] == ID]

    # Calculate mode values for each customer for each column
    for col in mode_col:
        mode_values[col] = customer_data[col].mode().iloc[0]

    # Calculate median values
    for col in median_col:
        median_values[col] = customer_data[col].median()
    
    # Identify Outliers and replace with Null values
    for col in mode_col:
        Q1 = customer_data[col].quantile(0.25)
        Q3 = customer_data[col].quantile(0.75)
        IQR = Q3 - Q1
        Upper = Q3 + 1.5*IQR
        Lower = Q1 - 1.5*IQR
    
        customer_data[col] = customer_data[col].apply(lambda x: x if (x<Upper and x>Lower) else np.nan) 
    
    # Replace all Null values
    for col in mode_col:
        data_train.loc[data_train['Customer_ID']== ID, col] = customer_data[col].fillna(mode_values[col])
        
    for col in median_col:
        data_train.loc[data_train['Customer_ID']== ID, col] = customer_data[col].fillna(median_values[col])
        
        
data_train.info()
data_train.head(20)
# Now We move to String Columns. Here again I will use mode values for data imputation
# A good appraoch would be to use regex to resolve issues

mode_string_col = ['Occupation','Type_of_Loan','Credit_Mix','Payment_Behaviour']

mode_string ={}
count = 0

for ID in customer_unique:
    
    count = count + 1
    print(count, end='\n')
    print(f'Currently Working for Customer: {ID}')
    
    customer_data = data_train[data_train['Customer_ID']==ID]
    
    # Count Mode values
    for col in mode_string_col:
        mode_series = customer_data[col].mode()
        if not mode_series.empty:
            mode_string[col] = mode_series.iloc[0]
        else:
            # Handle the case when no mode value is found
            mode_string[col] = 'Not Specified'
            
    # If value is other than mode replace with nan
    for col in mode_string_col:
        customer_data.loc[customer_data[col] != mode_string[col], col] = np.nan
        
    # Impute missing values in string columns with mode values
    for col in mode_string:
        data_train.loc[data_train['Customer_ID']== ID, col] = customer_data[col].fillna(mode_string[col])

data_train.info()
data_train.head(50)

# Identify any remaining Null values
# I am going ot drop the column. A better appraoch would be to fill in months

data_train = data_train.drop(['Credit_History_Age'], axis = 1)

# We will also drop columns not neeeded
# Customer_ID, Month

drop_col = ['Customer_ID','Month']
data_train = data_train.drop(drop_col, axis =1)

# Occupation, Type_of_Loan, Credit_Mix, Payment_of_Min_Amount, Payment_Behaviour, Credit_Score
data_train['Occupation'].value_counts() # 16 ordinal categories
data_train['Type_of_Loan'].nunique() # 6260 categories
data_train['Credit_Mix'].value_counts()  # 4 categories
data_train['Payment_of_Min_Amount'].value_counts() # 3 Categories
data_train['Payment_Behaviour'].value_counts()   # 7 categories
data_train['Credit_Score'].value_counts() # 3 categories

# We can segerate type of Loans using , as our seperator. Here I am just going to drop this column
drop_col = ['Type_of_Loan']
data_train = data_train.drop(drop_col, axis =1)


# Split into X and Y
X = data_train.drop(['Credit_Score'], axis =1)
Y = data_train['Credit_Score']
X.info()
Y.info()

# Encoding of Categorical Variables
# Label Encoding for Credit_Mix, Payment_of_Min_Amount, Credit_Score
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

X['Credit_Mix'] = label.fit_transform(X['Credit_Mix'])
X['Payment_of_Min_Amount'] = label.fit_transform(X['Payment_of_Min_Amount'])
Y = label.fit_transform(Y)

# One Hot Encoding for Occupation, Payment_Behaviour
from sklearn.preprocessing import OneHotEncoder
onehot = OneHotEncoder()

# One Hot Encoding 
onehot_encoder = OneHotEncoder(drop='first', sparse=False)
X_encoded = onehot_encoder.fit_transform(X[['Occupation', 'Payment_Behaviour']])

# Concatenate encoded features with original features
X_encoded_df = pd.DataFrame(X_encoded, columns=onehot_encoder.get_feature_names_out(['Occupation', 'Payment_Behaviour']))
X = pd.concat([X.drop(['Occupation', 'Payment_Behaviour'], axis=1), X_encoded_df], axis=1)

# Standardize data
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X = scalar.fit_transform(X)

# Identify Less useful or useless features
# We can perform PCA for feature selection

# Final data set

# Part 4: Model Training
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42,
                                                    stratify=Y)

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

