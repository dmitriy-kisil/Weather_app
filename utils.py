import numpy as np
from datetime import datetime
from datetime import timedelta
from pymongo import MongoClient
import pytz
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
from tensorflow.keras.models import load_model
from joblib import dump, load
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures


def get_db(url):
    client = MongoClient(url)
    db = client.WeatherApp
    return db


def get_weather(owm, cities):
    print("Call API to get weather")
    cities_temperatures = []
    mgr = owm.weather_manager()
    for city in cities:
        weather = mgr.weather_at_place(city).weather
        temperature_from_weather = weather.temperature(unit='celsius')['temp']
        cities_temperatures.append(temperature_from_weather)
    print(f'New temps are: {cities_temperatures}')
    return cities_temperatures


def predict_one_day(selected_date, date_format, model):
    one_day = selected_date + timedelta(days=1)
    prep_one_day = datetime.strftime(one_day, date_format)
    # X_one_day = np.zeros((1, 1))
    month, day, year = prep_one_day.split("/")
    example = pd.DataFrame({"day": [day], "month": [month], "year": [year]})
    X_one_day = example
    y_pred = model.predict(X_one_day)
    y_pred = [int(i) for i in list(y_pred)]
    date_data, predicted_data = [prep_one_day], y_pred
    print('predicted response:', y_pred, sep='\n')
    return date_data, predicted_data


def predict_seven_days(selected_date, date_format, model, le):
    seven_days = [selected_date + timedelta(days=x) for x in range(1, 8)]
    prep_seven_days = [datetime.strftime(i, date_format) for i in seven_days]
    month = [i.split("/")[0] for i in prep_seven_days]
    day = [i.split("/")[1] for i in prep_seven_days]
    year = [i.split("/")[2] for i in prep_seven_days]
    example = pd.DataFrame({"day": day, "month": month, "year": year})
    # X_seven_days = le.fit_transform(prep_seven_days).reshape(-1, 1)
    X_seven_days = example
    y_pred = model.predict(X_seven_days)
    y_pred = [int(i) for i in list(y_pred)]
    date_data, predicted_data = prep_seven_days, y_pred
    print('predicted response:', y_pred, sep='\n')
    return date_data, predicted_data


def predict_ten_days(selected_date, date_format, model, le):
    ten_days = [selected_date + timedelta(days=x) for x in range(1, 11)]
    prep_ten_days = [datetime.strftime(i, date_format) for i in ten_days]
    month = [i.split("/")[0] for i in prep_ten_days]
    day = [i.split("/")[1] for i in prep_ten_days]
    year = [i.split("/")[2] for i in prep_ten_days]
    example = pd.DataFrame({"day": day, "month": month, "year": year})
    # X_ten_days = le.fit_transform(prep_ten_days).reshape(-1, 1)
    X_ten_days = example
    y_pred = model.predict(X_ten_days)
    y_pred = [int(i) for i in list(y_pred)]
    date_data, predicted_data = prep_ten_days, y_pred
    print('predicted response:', y_pred, sep='\n')
    return date_data, predicted_data


def predict_one_day2(selected_date, date_format, model, raw_seq, n_steps_in):
    one_day = selected_date + timedelta(days=1)
    prep_one_day = datetime.strftime(one_day, date_format)
    X_one_day = np.array(raw_seq[-n_steps_in:]).astype(np.float32)
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
    X_seven_days = np.array(raw_seq[-n_steps_in:]).astype(np.float32)
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
    X_ten_days = np.array(raw_seq[-n_steps_in:]).astype(np.float32)
    X_ten_days = X_ten_days.reshape((1, n_steps_in, 1))
    y_pred = model.predict(X_ten_days, verbose=0)
    y_pred = y_pred[0][:10]
    y_pred = [int(i) for i in y_pred]
    date_data, predicted_data = prep_ten_days, y_pred
    print('predicted response:', y_pred, sep='\n')
    return date_data, predicted_data


def predict_one_day_polynomial(selected_date, date_format, degree, model):
    one_day = selected_date + timedelta(days=1)
    prep_one_day = datetime.strftime(one_day, date_format)
    month, day, year = prep_one_day.split("/")
    example = pd.DataFrame({"day": [day], "month": [month], "year": [year]})
    poly_features = PolynomialFeatures(degree=degree)
    # X_one_day = np.zeros((1, 1))
    X_one_day = example
    X_one_day = poly_features.fit_transform(X_one_day)
    y_pred = model.predict(X_one_day)
    y_pred = [int(i) for i in list(y_pred)]
    date_data, predicted_data = [prep_one_day], y_pred
    print('predicted response:', y_pred, sep='\n')
    return date_data, predicted_data


def predict_seven_days_polynomial(selected_date, date_format, degree, model, le):
    seven_days = [selected_date + timedelta(days=x) for x in range(1, 8)]
    prep_seven_days = [datetime.strftime(i, date_format) for i in seven_days]
    poly_features = PolynomialFeatures(degree=degree)
    month = [i.split("/")[0] for i in prep_seven_days]
    day = [i.split("/")[1] for i in prep_seven_days]
    year = [i.split("/")[2] for i in prep_seven_days]
    example = pd.DataFrame({"day": day, "month": month, "year": year})
    # X_seven_days = le.fit_transform(prep_seven_days).reshape(-1, 1)
    X_seven_days = example
    X_seven_days = poly_features.fit_transform(X_seven_days)
    y_pred = model.predict(X_seven_days)
    y_pred = [int(i) for i in list(y_pred)]
    date_data, predicted_data = prep_seven_days, y_pred
    print('predicted response:', y_pred, sep='\n')
    return date_data, predicted_data


def predict_ten_days_polynomial(selected_date, date_format, degree, model, le):
    ten_days = [selected_date + timedelta(days=x) for x in range(1, 11)]
    prep_ten_days = [datetime.strftime(i, date_format) for i in ten_days]
    poly_features = PolynomialFeatures(degree=degree)
    month = [i.split("/")[0] for i in prep_ten_days]
    day = [i.split("/")[1] for i in prep_ten_days]
    year = [i.split("/")[2] for i in prep_ten_days]
    example = pd.DataFrame({"day": day, "month": month, "year": year})
    # X_ten_days = le.fit_transform(prep_ten_days).reshape(-1, 1)
    X_ten_days = example
    X_ten_days = poly_features.fit_transform(X_ten_days)
    y_pred = model.predict(X_ten_days)
    y_pred = [int(i) for i in list(y_pred)]
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


def make_predictions_polynomial_regression(selected_date, date_format, degree, model, le):
    today = datetime.strptime(selected_date, date_format)
    future_one_day, future_one_day_data = predict_one_day_polynomial(today, date_format, degree, model)
    future_seven_days, future_seven_days_data = predict_seven_days_polynomial(today, date_format, degree, model, le)
    future_ten_days, future_ten_days_data = predict_ten_days_polynomial(today, date_format, degree, model, le)
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


def tz_diff(geolocator, tf, cityname):
    # Get lat, long from city name
    location = geolocator.geocode(cityname)
    # Get timezone from coordinates
    latitude, longitude = location.latitude, location.longitude
    # Timezone
    city_timezone = tf.timezone_at(lng=longitude, lat=latitude)
    city_timezone = str(city_timezone)
    print(city_timezone)

    utcnow = pytz.timezone('utc').localize(datetime.utcnow())  # generic time
    here = utcnow.astimezone(pytz.timezone('UTC')).replace(tzinfo=None)
    there = utcnow.astimezone(pytz.timezone(city_timezone)).replace(tzinfo=None)

    offset = relativedelta(here, there)
    print(offset.hours)
    print("The date in " + str(cityname) + " is: " + there.strftime('%A, %m/%d/%y %H:%M:%S %Z %z'))
    return offset.hours


# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out
        # check if we are beyond the sequence
        if out_end_ix > len(sequence):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def predict_for_one_city(db, local_date, resulted_dict, city):
    for city in [city]:
        city_data = db.locations.find({"cities": city})
        le = preprocessing.LabelEncoder()
        print(city_data)
        date_format = "%m/%d/%Y"
        new_date = datetime.strftime(local_date, date_format)
        city_data_one = db.locations.find_one({"date": new_date, "cities": {"$regex": city}})
        find_index = city_data_one['cities'].index(city)
        print(find_index)
        X = [i["date"] for i in city_data]
        print(X)
        # X = X[:1] # test when one example
        [print(i) for i in city_data]
        l1 = [i[0] for i in city_data]
        print(l1)
        y = []
        for c, v in enumerate(X):
            one_city = db.locations.find_one({"date": v, "cities": city})
            find_index = one_city['cities'].index(city)
            y.append(one_city['temperatures'][find_index])
        # y = [db.locations.find_one({"date": i, "cities": city})['temperatures'][find_index] for i in X]
        print(1234)
        print(y)
        # y = y[:1] # test when one example
        # choose a number of time steps
        n_steps_in, n_steps_out = 10, 10
        if len(y) == 1:
            model = LinearRegression()
            one_X, one_y = X[0], y[0]
            X = [one_X for x in range(0, 10)]
            y = [one_y for y in range(0, 10)]
            df = pd.DataFrame({"date": X, "temp": y})
            # X = le.fit_transform(X).reshape(-1, 1)
            df['month'] = df['date'].apply(lambda x: int(x.split("/")[0]))
            df['day'] = df['date'].apply(lambda x: int(x.split("/")[1]))
            df['year'] = df['date'].apply(lambda x: int(x.split("/")[2]))
            df['temp'] = df['temp'].astype(int)

            X = df[['day', 'month', 'year']]
            y = df['temp'].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            model.fit(X_train, y_train)
            predicted_temp = make_predictions_linear_regression(new_date, date_format, model, le)
            resulted_dict[city] = predicted_temp
            break
        if len(y) <= 30:
            print('use linear regression')
            model = LinearRegression()
            print(X)
            df = pd.DataFrame({"date": X, "temp": y})
            # X = le.fit_transform(X).reshape(-1, 1)
            df['month'] = df['date'].apply(lambda x: int(x.split("/")[0]))
            df['day'] = df['date'].apply(lambda x: int(x.split("/")[1]))
            df['year'] = df['date'].apply(lambda x: int(x.split("/")[2]))
            df['temp'] = df['temp'].astype(int)

            X = df[['day', 'month', 'year']]
            y = df['temp'].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            model.fit(X_train, y_train)
            predicted_temp = make_predictions_linear_regression(new_date, date_format, model, le)
            resulted_dict[city] = predicted_temp
            break
        else:
            print('use polynomial regression')
            degree = 3
            model = LinearRegression()
            df = pd.DataFrame({"date": X, "temp": y})
            # X = le.fit_transform(X).reshape(-1, 1)
            df['month'] = df['date'].apply(lambda x: int(x.split("/")[0]))
            df['day'] = df['date'].apply(lambda x: int(x.split("/")[1]))
            df['year'] = df['date'].apply(lambda x: int(x.split("/")[2]))
            df['temp'] = df['temp'].astype(int)

            X = df[['day', 'month', 'year']]
            y = df['temp'].astype(int)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            poly_features = PolynomialFeatures(degree=degree)
            # transforms the existing features to higher degree features.
            X_train_poly = poly_features.fit_transform(X_train)
            model.fit(X_train_poly, y_train)
            predicted_temp = make_predictions_polynomial_regression(new_date, date_format, degree, model, le)
            resulted_dict[city] = predicted_temp
            break
        # # reshape from [samples, timesteps] into [samples, timesteps, features]
        # n_features = 1
        #
        # # define model
        # model = Sequential()
        # model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
        # model.add(RepeatVector(n_steps_out))
        # model.add(LSTM(100, activation='relu', return_sequences=True))
        # model.add(TimeDistributed(Dense(1)))
        # model.compile(optimizer='adam', loss='mse')
        #
        # # define input sequence
        # print(n_steps_in, n_steps_out)
        # print(len(y))
        # if len(y) > (n_steps_in + n_steps_out + n_steps_out):
        #     raw_seq = y[:-n_steps_out]
        #     y_true = y[-n_steps_out:]
        #     print(len(raw_seq))
        #     # split into samples
        #     X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
        #     print(city)
        #     print(type(X), print(X))
        #     X = X.reshape((X.shape[0], X.shape[1], n_features))
        #     # fit model
        #     model.fit(X, y, epochs=5, verbose=0)
        #     # demonstrate prediction
        #     x_input = np.array(raw_seq[-n_steps_in:])
        #     x_input = x_input.astype(np.float32)
        #     x_input = x_input.reshape((1, n_steps_in, n_features))
        #     yhat = model.predict(x_input, verbose=0)
        #     print(yhat)
        #     print(y_true)
        #     yhat = [yhat[0][0]]
        #     y_true = [y_true[0]]
        #     print(mean_absolute_error(y_true, yhat))
        # else:
        #     raw_seq = y[:]
        #     print(len(raw_seq))
        #     # split into samples
        #     X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
        #     print(city)
        #     print(type(X), print(X))
        #     X = X.reshape((X.shape[0], X.shape[1], n_features))
        #     # fit model
        #     model.fit(X, y, epochs=5, verbose=0)
        #     # demonstrate prediction
        #     x_input = np.array(raw_seq[-n_steps_in:])
        #     x_input = x_input.reshape((1, n_steps_in, n_features))
        #     yhat = model.predict(x_input, verbose=0)
        # predicted_temp = make_predictions_rnn(new_date, date_format, model, raw_seq, n_steps_in)
        # today = datetime.strptime(new_date, date_format)
        # resulted_dict[city] = predicted_temp
    print(resulted_dict[city])
    db.locations.find_one_and_update({"date": new_date}, {'$set': {'predicted_temp': resulted_dict}})


def if_future_day_exists(offset):
    previous_date = datetime.now(pytz.timezone('utc')) - timedelta(hours=offset)
    future_date = datetime.now(pytz.timezone('utc')) + timedelta(hours=12)
    future_day, previous_day = int(future_date.strftime('%d')), int(previous_date.strftime('%d'))

    if future_day > previous_day:
        return True
    else:
        return False


def dummy_hour_temp(temp):
    d1 = {}
    for i in range(24):
        d1[str(i) + "_hour"] = temp
    return d1
