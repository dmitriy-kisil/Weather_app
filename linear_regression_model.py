import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing

df = pd.read_csv('city_data.csv', index_col=0)
# df = df.loc['01/01/2020':]
print(df.head())
le = preprocessing.LabelEncoder()
X = df.index
print(X)
y = df['Kharkiv,Ukraine'].astype(int)
print(y)
X = le.fit_transform(X).reshape(-1, 1)
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
n_steps, n_features = len(df), 1
# define model
model = LinearRegression()
# fit model
model.fit(X_train, y_train)
# model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# y_pred = [int(i) for i in list(y_pred)]
print(len(y_test), len(y_pred))
print(y_test, y_pred)
y_test, y_pred = np.array(y_test, dtype=int).reshape(-1, 1), np.array(y_pred, dtype=int).reshape(-1, 1)
print(y_test, y_pred)
r_sq = model.score(y_test, y_pred)
print('coefficient of determination:', r_sq)
mse = metrics.mean_squared_error(y_test, y_pred)
print('mse:', mse)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('rmse:', rmse)
mae = metrics.mean_absolute_error(y_test, y_pred)
print('mae:', mae)
df.reset_index()
df.rename(columns={'index': 'temp'}, inplace=True)
df['y'] = y
df['y_pred'] = np.array(model.predict(X), dtype=int).reshape(-1, 1)
df.reset_index(inplace=True)
df.rename(columns={'index': 'date'}, inplace=True)
print(df.head())
df.to_csv('city_data2.csv')
