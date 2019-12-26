import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
le = preprocessing.LabelEncoder()

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])
x = ["21/12", "22/12", "23/12", "24/12", "25/12", "26/12", "27/12", "28/12", "29/12", "30/12"]
y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
X = le.fit_transform(x).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# y = [1, 2, 3, 4, 5]
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(len(y_test), len(y_pred))
print(y_test, y_pred)
y_test, y_pred = np.array(y_test).reshape(-1, 1), np.array(y_pred).reshape(-1, 1)
r_sq = model.score(y_test, y_pred)
print('coefficient of determination:', r_sq)
new_date = ["23/12"]
prep_new_date = pd.get_dummies(new_date)
# prep_new_date = le.fit_transform(new_date).reshape(1, -1)
y_pred = model.predict(prep_new_date)
print('predicted response:', y_pred, sep='\n')