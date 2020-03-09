# --------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Code starts here
df = pd.read_csv(path)
cols_list = ['INCOME','HOME_VAL','BLUEBOOK','OLDCLAIM','CLM_AMT' ]
df[cols_list] = df[cols_list].astype(str)
for col in cols_list:
    df[col].replace({'\$': '', ',': ''}, regex=True,inplace=True)
print(df.info())
X = df.drop('CLAIM_FLAG', axis=1)
y = df['CLAIM_FLAG']
count =df['CLAIM_FLAG'].value_counts()
print(count)
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 6)

# Code ends here


# --------------
# Code starts here
X_train[cols_list] = X_train[cols_list].astype(float)
X_test[cols_list] = X_test[cols_list].astype(float)
print(X_train.isnull().sum())
print(X_test.isnull().sum())
# Code ends here


# --------------

# drop missing values
X_train.dropna(subset=['YOJ','OCCUPATION'],inplace=True)
X_test.dropna(subset=['YOJ','OCCUPATION'],inplace=True)


y_train=y_train[X_train.index]
y_test=y_test[X_test.index]



# fill missing values with mean
X_train['AGE'].fillna((X_train['AGE'].mean()), inplace=True)
X_test['AGE'].fillna((X_train['AGE'].mean()), inplace=True)

X_train['CAR_AGE'].fillna((X_train['CAR_AGE'].mean()), inplace=True)
X_test['CAR_AGE'].fillna((X_train['CAR_AGE'].mean()), inplace=True)



X_train['INCOME'].fillna((X_train['INCOME'].mean()), inplace=True)
X_test['INCOME'].fillna((X_train['INCOME'].mean()), inplace=True)



X_train['HOME_VAL'].fillna((X_train['HOME_VAL'].mean()), inplace=True)
X_test['HOME_VAL'].fillna((X_train['HOME_VAL'].mean()), inplace=True)


print(X_train.isnull().sum())
print(X_test.isnull().sum())


# --------------
from sklearn.preprocessing import LabelEncoder
columns = ["PARENT1","MSTATUS","GENDER","EDUCATION","OCCUPATION","CAR_USE","CAR_TYPE","RED_CAR","REVOKED"]

# Code starts here
for col in columns:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Code ends here



# --------------
from sklearn.metrics import precision_score 
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression



# code starts here 
model = LogisticRegression(random_state = 6)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

score = accuracy_score(y_test, y_pred)
print(score)
# Code ends here


# --------------
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# code starts here
smote = SMOTE(random_state = 9)
X_train , y_train = smote.fit_sample(X_train , y_train)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Code ends here


# --------------
# Code Starts here
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)
print(score)
# Code ends here


