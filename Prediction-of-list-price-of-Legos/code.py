# --------------
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
# code starts here
df = pd.read_csv(path)
print(df.head(5))
X = df.loc[:, df.columns != 'list_price']
y = df['list_price']
X_train,X_test,y_train,y_test = train_test_split(df,y,test_size=0.2,random_state=6)
# code ends here



# --------------
import matplotlib.pyplot as plt

# code starts here        
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=6)
cols = X_train.columns.values
fig, axes = plt.subplots(nrows=3, ncols=3)
# for i in range(len(cols)):
#     col = cols[ i ]
#     plt.scatter(X_train[col],X_train['list_price'])
print(cols)
i = 0
for row in axes:
    for j in row:
        col = cols[i]
        j.scatter(X_train[col], y_train)
    i+=1
plt.show()
# code ends here



# --------------
# Code starts here
corr = X_train.corr()
# print(corr)

X_train.drop(['play_star_rating','val_star_rating'], 1 ,inplace=True) 

X_test.drop(['play_star_rating','val_star_rating'], 1 ,inplace=True)
# Code ends here


# --------------
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import math
# Code starts here
regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test,y_pred)
r2  = r2_score(y_test, y_pred)
print(r2)
# Code ends here


# --------------
# Code starts here
residual = y_test - y_pred
residual.plot(kind='hist')


# Code ends here


