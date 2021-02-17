import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from joblib import dump, load
from utils import *
from datetime import datetime
from datetime import timedelta
from weather_flask import get_db
from pyowm import OWM
from dotenv import load_dotenv, find_dotenv

load_dotenv()

# Add MongoDB URL:
mongodb_url = os.environ['MONGODB_URL']
# Add tokens for API
openweatherapi_token = os.environ['OPENWEATHERAPI_TOKEN']
# Initialize third-party API
owm = OWM(openweatherapi_token)  # You MUST provide a valid API key


if __name__ == "__main__":

    db = get_db(mongodb_url)
    le = preprocessing.LabelEncoder()

    date_format = "%m/%d/%Y"
    new_date = datetime.strftime(datetime.now(), date_format)
    new_date_info = db.locations.find_one({'date': new_date})
    cities = new_date_info['cities']
    predicted_temp_info = new_date_info['predicted_temp']
    cities_with_predictions = predicted_temp_info.keys()
    print(cities)
    print('Make predictions')
    # resulted_dict = {}
    cities_without_predictions = [city for city in cities if city not in cities_with_predictions]
    for city in cities_without_predictions:
        model = LinearRegression()
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

        y = []
        for c, v in enumerate(X):
            one_city = db.locations.find_one({"date": v, "cities": city})
            find_index = one_city['cities'].index(city)
            y.append(one_city['temperatures'][find_index])

        # y = [db.locations.find_one({"date": i, "cities": city})['temperatures'][find_index] for i in X]
        print(y)
        # from one example cannot use model so create more examples from this one
        one_X, one_y = X[0], y[0]
        X = [one_X for x in range(0, 10)]
        y = [one_y for y in range(0, 10)]
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
        predicted_temp = make_predictions_linear_regression(new_date, date_format, model, le)
        predicted_temp_info[city] = predicted_temp
    print(predicted_temp_info)
    db.locations.find_one_and_update({"date": new_date}, {"$set": {"predicted_temp": predicted_temp_info}})
