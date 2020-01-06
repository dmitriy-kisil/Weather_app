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
import pyowm

# Add tokens for API
openweatherapi_token = os.environ['OPENWEATHERAPI_TOKEN']
# Initialize third-party API
owm = pyowm.OWM(openweatherapi_token)  # You MUST provide a valid API key


def predict_one_day(selected_date, model, X, y):
    today = datetime.strptime(selected_date, date_format)
    one_day = today + timedelta(days=1)
    prep_one_day = datetime.strftime(one_day, date_format)
    X_one_day = pd.get_dummies(prep_one_day)
    y_pred = model.predict(X_one_day)
    y_pred = [round(float(i), 2) for i in list(y_pred)]
    date_data, predicted_data = [prep_one_day], y_pred
    print('predicted response:', y_pred, sep='\n')
    return date_data, predicted_data


def predict_seven_days(selected_date, model, X, y):
    today = datetime.strptime(selected_date, date_format)
    seven_days = [today + timedelta(days=x) for x in range(1, 8)]
    prep_seven_days = [datetime.strftime(i, date_format) for i in seven_days]
    X_seven_days = le.fit_transform(prep_seven_days).reshape(-1, 1)
    y_pred = model.predict(X_seven_days)
    y_pred = [round(float(i), 2) for i in list(y_pred)]
    date_data, predicted_data = prep_seven_days, y_pred
    print('predicted response:', y_pred, sep='\n')
    return date_data, predicted_data


def predict_ten_days(selected_date, model, X, y):
    today = datetime.strptime(selected_date, date_format)
    ten_days = [today + timedelta(days=x) for x in range(1, 11)]
    prep_ten_days = [datetime.strftime(i, date_format) for i in ten_days]
    X_ten_days = le.fit_transform(prep_ten_days).reshape(-1, 1)
    y_pred = model.predict(X_ten_days)
    y_pred = [round(float(i), 2) for i in list(y_pred)]
    date_data, predicted_data = prep_ten_days, y_pred
    print('predicted response:', y_pred, sep='\n')
    return date_data, predicted_data


def get_weather(cities):
    print("Call API to get weather")
    cities_temperatures = []
    for city in cities:
        observation = owm.weather_at_place(city)
        w = observation.get_weather()
        temperature_from_weather = w.get_temperature('celsius')['temp']
        cities_temperatures.append(temperature_from_weather)
    print(f'New temps are: {cities_temperatures}')
    return cities_temperatures


if __name__ == "__main__":

    db = get_db()
    le = preprocessing.LabelEncoder()
    directory_for_storing_weights = os.getcwd() + '/weights'
    if not os.path.exists(directory_for_storing_weights):
        os.makedirs(directory_for_storing_weights)

    date_format = "%m/%d/%Y"
    new_date = datetime.strftime(datetime.now(), date_format)
    # For testing purpose
    # db.locations.delete_one({"date": new_date})

    find_today = db.locations.find_one({'date': new_date})
    if not find_today:
        print('Today is not found, create a new one')
        all_dates = list(db.locations.find({}, {'date': 1, '_id': 0}))
        all_dates = [i['date'] for i in all_dates]
        the_most_recent_date = all_dates[-1]
        previous_day = the_most_recent_date
        get_previous_day = db.locations.find_one({'date': previous_day})
        prev_cities = get_previous_day['cities']
        prev_ips = get_previous_day['ip_addresses']
        prev_number_of_cities = get_previous_day['number_of_cities']
        new_temperatures = get_weather(prev_cities)
        cities = prev_cities
        db.locations.insert_one({"date": new_date, 'cities': prev_cities, 'ip_addresses': prev_ips,
                                 'temperatures': new_temperatures, 'number_of_cities': prev_number_of_cities})
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
            city_data_one = db.locations.find_one({"date": previous_day, "cities": {"$regex": city}})
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
        db.locations.find_one_and_update({"date": new_date}, {'$set': {'predicted_temp': resulted_dict}})
        print('New day created successfully')
    else:
        print('Today is found, nothing to do')
