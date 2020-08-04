import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from tensorflow.keras.models import load_model
from joblib import dump, load
from datetime import datetime
from datetime import timedelta
from weather_flask import get_db

# db = get_db()
#
# le = preprocessing.LabelEncoder()
#
# X = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
# y = np.array([5, 20, 14, 32, 22, 38])
# X = ["21/12", "22/12", "23/12", "24/12", "25/12", "26/12", "27/12", "28/12", "29/12", "30/12"]
# y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# X = le.fit_transform(X).reshape(-1, 1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# # y = [1, 2, 3, 4, 5]
# model = LinearRegression()
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# print(len(y_test), len(y_pred))
# print(y_test, y_pred)
# y_test, y_pred = np.array(y_test).reshape(-1, 1), np.array(y_pred).reshape(-1, 1)
# r_sq = model.score(y_test, y_pred)
# print('coefficient of determination:', r_sq)
# new_date = ["23/12"]
# date_format = "%m/%d/%Y"
# today = datetime.now()
# prep_today = datetime.strftime(today, date_format)
# get_date_from_db = db.locations.find_one({"date": prep_today})
# one_day = today + timedelta(days=1)
# prep_one_day = datetime.strftime(one_day, date_format)
# X_one_day = pd.get_dummies(prep_one_day)
# y_pred = model.predict(X_one_day)
# print('predicted response:', y_pred, sep='\n')
# seven_days = [today + timedelta(days=x) for x in range(7)]
# prep_seven_days = [datetime.strftime(i, date_format) for i in seven_days]
# X_seven_days = le.fit_transform(prep_seven_days).reshape(-1, 1)
# y_pred = model.predict(X_seven_days)
# print('predicted response:', y_pred, sep='\n')
# ten_days = [today + timedelta(days=x) for x in range(10)]
# print(ten_days)
# prep_ten_days = [datetime.strftime(i, date_format) for i in ten_days]
# X_ten_days = le.fit_transform(prep_ten_days).reshape(-1, 1)
# y_pred = model.predict(X_ten_days)
# print('predicted response:', y_pred, sep='\n')


def predict_one_day(selected_date, model, X, y):
    today = datetime.strptime(selected_date, date_format)
    one_day = today + timedelta(days=1)
    prep_one_day = datetime.strftime(one_day, date_format)
    X_one_day = np.zeros((1, 1))
    y_pred = model.predict(X_one_day)
    y_pred = [int(i) for i in list(y_pred)]
    date_data, predicted_data = [prep_one_day], y_pred
    print('predicted response:', y_pred, sep='\n')
    return date_data, predicted_data


def predict_seven_days(selected_date, model, X, y):
    today = datetime.strptime(selected_date, date_format)
    seven_days = [today + timedelta(days=x) for x in range(1, 8)]
    prep_seven_days = [datetime.strftime(i, date_format) for i in seven_days]
    X_seven_days = le.fit_transform(prep_seven_days).reshape(-1, 1)
    y_pred = model.predict(X_seven_days)
    y_pred = [int(i) for i in list(y_pred)]
    date_data, predicted_data = prep_seven_days, y_pred
    print('predicted response:', y_pred, sep='\n')
    return date_data, predicted_data


def predict_ten_days(selected_date, model, X, y):
    today = datetime.strptime(selected_date, date_format)
    ten_days = [today + timedelta(days=x) for x in range(1, 11)]
    prep_ten_days = [datetime.strftime(i, date_format) for i in ten_days]
    X_ten_days = le.fit_transform(prep_ten_days).reshape(-1, 1)
    y_pred = model.predict(X_ten_days)
    y_pred = [int(i) for i in list(y_pred)]
    date_data, predicted_data = prep_ten_days, y_pred
    print('predicted response:', y_pred, sep='\n')
    return date_data, predicted_data


if __name__ == "__main__":

    db = get_db()
    le = preprocessing.LabelEncoder()
    directory_for_storing_weights = os.getcwd() + '/weights'
    if not os.path.exists(directory_for_storing_weights):
        os.makedirs(directory_for_storing_weights)

    date_format = "%m/%d/%Y"
    new_date = datetime.strftime(datetime.now(), date_format)
    cities = db.locations.find_one({'date': new_date})['cities']
    print(cities)
    print('Make predictions')
    resulted_dict = {}
    for city in cities:
        path_for_model_city_weights = directory_for_storing_weights + '/' + city + '.joblib'
        if not os.path.exists(path_for_model_city_weights):
            print(f"Could not found weights for a {city} model, create a new one")
            model = LinearRegression()
        else:
            print(f"Found the weights, loads..")
            # model = load_model(path_for_model_city_weights)
            model = load(path_for_model_city_weights)

        city_data = db.locations.find({"cities": city})
        print(city_data)
        city_data_one = db.locations.find_one({"date": new_date, "cities": {"$regex": city}})
        find_index = city_data_one['cities'].index(city)
        print(find_index)
        X = [i["date"] for i in city_data]
        print(X)
        [print(i) for i in city_data]
        l1 = [i[0] for i in city_data]
        print(l1)

        y = [i["temperatures"][find_index] for i in city_data]
        y = []
        for c, v in enumerate(X):
            one_city = db.locations.find_one({"date": v, "cities": city})
            find_index = one_city['cities'].index(city)
            y.append(one_city['temperatures'][find_index])

        # y = [db.locations.find_one({"date": i, "cities": city})['temperatures'][find_index] for i in X]
        print(y)
        X = le.fit_transform(X).reshape(-1, 1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred = [float(i) for i in list(y_pred)]
        print(len(y_test), len(y_pred))
        print(y_test, y_pred)
        y_test, y_pred = np.array(y_test).reshape(-1, 1), np.array(y_pred).reshape(-1, 1)
        r_sq = model.score(y_test, y_pred)
        print('coefficient of determination:', r_sq)
        future_one_day, future_one_day_data = predict_one_day(new_date, model, X, y)
        future_seven_days, future_seven_days_data = predict_seven_days(new_date, model, X, y)
        future_ten_days, future_ten_days_data = predict_ten_days(new_date, model, X, y)
        today = datetime.strptime(new_date, date_format)
        dict_1_day = dict(zip(future_one_day, future_one_day_data))
        dict_7_days = dict(zip(future_seven_days, future_seven_days_data))
        dict_10_days = dict(zip(future_ten_days, future_ten_days_data))
        predicted_temp = {"1_day": dict_1_day,
                          "7_days": dict_7_days,
                          "10_days": dict_10_days}
        resulted_dict[city] = predicted_temp
        # model.save(path_for_model_city_weights)
        dump(model, path_for_model_city_weights)
    print(resulted_dict)
    db.locations.find_one_and_update({"date": new_date}, {"$set": {"predicted_temp": resulted_dict}})
