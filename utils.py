import numpy as np
from datetime import datetime
from datetime import timedelta
from pymongo import MongoClient


def get_db(url):
    client = MongoClient(url)
    db = client.WeatherApp
    return db


def get_weather(owm, cities):
    print("Call API to get weather")
    cities_temperatures = []
    for city in cities:
        observation = owm.weather_at_place(city)
        w = observation.get_weather()
        temperature_from_weather = w.get_temperature('celsius')['temp']
        cities_temperatures.append(temperature_from_weather)
    print(f'New temps are: {cities_temperatures}')
    return cities_temperatures


def predict_one_day(selected_date, date_format, model):
    one_day = selected_date + timedelta(days=1)
    prep_one_day = datetime.strftime(one_day, date_format)
    X_one_day = np.zeros((1, 1))
    y_pred = model.predict(X_one_day)
    y_pred = [int(i) for i in list(y_pred)]
    date_data, predicted_data = [prep_one_day], y_pred
    print('predicted response:', y_pred, sep='\n')
    return date_data, predicted_data


def predict_seven_days(selected_date, date_format, model, le):
    seven_days = [selected_date + timedelta(days=x) for x in range(1, 8)]
    prep_seven_days = [datetime.strftime(i, date_format) for i in seven_days]
    X_seven_days = le.fit_transform(prep_seven_days).reshape(-1, 1)
    y_pred = model.predict(X_seven_days)
    y_pred = [int(i) for i in list(y_pred)]
    date_data, predicted_data = prep_seven_days, y_pred
    print('predicted response:', y_pred, sep='\n')
    return date_data, predicted_data


def predict_ten_days(selected_date, date_format, model, le):
    ten_days = [selected_date + timedelta(days=x) for x in range(1, 11)]
    prep_ten_days = [datetime.strftime(i, date_format) for i in ten_days]
    X_ten_days = le.fit_transform(prep_ten_days).reshape(-1, 1)
    y_pred = model.predict(X_ten_days)
    y_pred = [int(i) for i in list(y_pred)]
    date_data, predicted_data = prep_ten_days, y_pred
    print('predicted response:', y_pred, sep='\n')
    return date_data, predicted_data


def predict_one_day2(selected_date, date_format, model, raw_seq, n_steps_in):
    one_day = selected_date + timedelta(days=1)
    prep_one_day = datetime.strftime(one_day, date_format)
    X_one_day = np.array(raw_seq[-n_steps_in:])
    X_one_day = X_one_day.reshape((1, n_steps_in, 1))
    y_pred = model.predict(X_one_day, verbose=0)
    y_pred = [y_pred[0][0]]
    y_pred = [int(i) for i in list(y_pred)]
    date_data, predicted_data = [prep_one_day], y_pred
    print('predicted response:', y_pred, sep='\n')
    return date_data, predicted_data


def predict_seven_days2(selected_date, date_format, model, raw_seq, n_steps_in):
    seven_days = [selected_date + timedelta(days=x) for x in range(1, 8)]
    prep_seven_days = [datetime.strftime(i, date_format) for i in seven_days]
    X_seven_days = np.array(raw_seq[-n_steps_in:])
    X_seven_days = X_seven_days.reshape((1, n_steps_in, 1))
    y_pred = model.predict(X_seven_days, verbose=0)
    y_pred = y_pred[0][:7]
    y_pred = [int(i) for i in y_pred]
    date_data, predicted_data = prep_seven_days, y_pred
    print('predicted response:', y_pred, sep='\n')
    return date_data, predicted_data


def predict_ten_days2(selected_date, date_format, model, raw_seq, n_steps_in):
    ten_days = [selected_date + timedelta(days=x) for x in range(1, 11)]
    prep_ten_days = [datetime.strftime(i, date_format) for i in ten_days]
    X_ten_days = np.array(raw_seq[-n_steps_in:])
    X_ten_days = X_ten_days.reshape((1, n_steps_in, 1))
    y_pred = model.predict(X_ten_days, verbose=0)
    y_pred = y_pred[0][:10]
    y_pred = [int(i) for i in y_pred]
    date_data, predicted_data = prep_ten_days, y_pred
    print('predicted response:', y_pred, sep='\n')
    return date_data, predicted_data


def make_predictions_linear_regression(selected_date, date_format, model, le):
    today = datetime.strptime(selected_date, date_format)
    future_one_day, future_one_day_data = predict_one_day(today, date_format, model)
    future_seven_days, future_seven_days_data = predict_seven_days(today, date_format, model, le)
    future_ten_days, future_ten_days_data = predict_ten_days(today, date_format, model, le)
    dict_1_day = dict(zip(future_one_day, future_one_day_data))
    dict_7_days = dict(zip(future_seven_days, future_seven_days_data))
    dict_10_days = dict(zip(future_ten_days, future_ten_days_data))
    predicted_temp = {"1_day": dict_1_day,
                      "7_days": dict_7_days,
                      "10_days": dict_10_days}
    return predicted_temp


def make_predictions_rnn(selected_date, date_format, model, raw_seq, n_steps_in):
    today = datetime.strptime(selected_date, date_format)
    future_one_day, future_one_day_data = predict_one_day2(today, date_format, model, raw_seq, n_steps_in)
    future_seven_days, future_seven_days_data = predict_seven_days2(today, date_format, model, raw_seq, n_steps_in)
    future_ten_days, future_ten_days_data = predict_ten_days2(today, date_format, model, raw_seq, n_steps_in)
    dict_1_day = dict(zip(future_one_day, future_one_day_data))
    dict_7_days = dict(zip(future_seven_days, future_seven_days_data))
    dict_10_days = dict(zip(future_ten_days, future_ten_days_data))
    predicted_temp = {"1_day": dict_1_day,
                      "7_days": dict_7_days,
                      "10_days": dict_10_days}
    return predicted_temp
