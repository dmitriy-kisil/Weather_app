import pandas as pd
import os
from datetime import datetime
from utils import get_db
import pyowm
from dotenv import load_dotenv, find_dotenv
from utils import *
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

load_dotenv()

# Add MongoDB URL:
mongodb_url = os.environ['MONGODB_URL']
# Add tokens for API
openweatherapi_token = os.environ['OPENWEATHERAPI_TOKEN']
# Initialize third-party API
owm = pyowm.OWM(openweatherapi_token)  # You MUST provide a valid API key


if __name__ == "__main__":

    db = get_db(mongodb_url)
    date_format = "%m/%d/%Y"
    new_date = datetime.strftime(datetime.now(), date_format)
    get_previous_day = db.locations.find_one({'date': new_date})
    number_of_cities_to_parse = len(get_previous_day['cities'])
    prev_cities = get_previous_day['cities'][:number_of_cities_to_parse]
    city = prev_cities[1]
    local_date = datetime.now()
    resulted_dict = {}
    for city in [city]:
        city_data = db.locations.find({"cities": city})
        le = preprocessing.LabelEncoder()
        print(city_data)
        date_format = "%m/%d/%Y"
        new_date = datetime.strftime(local_date, date_format)
        if os.path.isfile(city + '.csv'):
            df = pd.read_csv(city + '.csv')
            df = df[df['date'] >= '05/01/2020']
            X, y = df['date'], df['temp']
        else:
            city_data_one = db.locations.find_one({"date": new_date, "cities": {"$regex": city}})
            find_index = city_data_one['cities'].index(city)
            print(find_index)
            X = [i["date"] for i in city_data if i['date'] >= "01/01/2020"]
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
            df = pd.DataFrame({"date": X, "temp": y})
            df.to_csv(city + '.csv')
        # y = y[:1] # test when one example
        # choose a number of time steps
        n_steps_in, n_steps_out = 10, 10
        len_y = len(y)
        if len_y == 1:
            model = LinearRegression()
            one_X, one_y = X[0], y[0]
            X = [one_X for x in range(0, 10)]
            y = [one_y for y in range(0, 10)]
            X = le.fit_transform(X).reshape(-1, 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            model.fit(X_train, y_train)
            predicted_temp = make_predictions_linear_regression(new_date, date_format, model, le)
            resulted_dict[city] = predicted_temp
            break
        if len_y <= n_steps_in:
            degree = 3
            model = LinearRegression()
            X = le.fit_transform(X).reshape(-1, 1)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
            poly_features = PolynomialFeatures(degree=degree)
            # transforms the existing features to higher degree features.
            X_train_poly = poly_features.fit_transform(X_train)
            X_test_poly = poly_features.fit_transform(X_test)
            model.fit(X_train_poly, y_train)
            y_pred = model.predict(X_test_poly)
            y_pred = [float(i) for i in list(y_pred)]
            print(len(y_test), len(y_pred))
            print(y_test, y_pred)
            print("The Explained Variance test: %.2f" % model.score(X_test_poly, y_test))
            print("The Mean Absolute Error test: %.2f degrees celsius" % mean_absolute_error(y_test, y_pred))
            print(
                "The Median Absolute Error test: %.2f degrees celsius" % median_absolute_error(y_test, y_pred))
            print("r2 score test: " + str(r2_score(y_test, y_pred)))
            # y_test, y_pred = np.array(y_test).reshape(-1, 1), np.array(y_pred).reshape(-1, 1)
            # r_sq = model.score(y_test, y_pred)
            # print('coefficient of determination:', r_sq)
            y_test = le.fit_transform(['09/26/2020']).reshape(-1, 1)
            y_pred = model.predict(poly_features.fit_transform(y_test))
            y_pred = [float(i) for i in list(y_pred)]
            print(len(y_test), len(y_pred))
            print(y_test, y_pred)
            # y_test, y_pred = np.array(y_test).reshape(1, -1), np.array(y_pred).reshape(1, -1)
            # r_sq = model.score(y_test, y_pred)
            # print('coefficient of determination:', r_sq)
            predicted_temp = make_predictions_polynomial_regression(new_date, date_format, degree, model, le)
            resulted_dict[city] = predicted_temp
            break
        if len_y <= (n_steps_in + n_steps_out):
            model = LinearRegression()
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
            y_test = le.fit_transform(['09/26/2020']).reshape(-1, 1)
            y_pred = model.predict(y_test)
            y_pred = [float(i) for i in list(y_pred)]
            print(len(y_test), len(y_pred))
            print(y_test, y_pred)
            y_test, y_pred = np.array(y_test).reshape(1, -1), np.array(y_pred).reshape(1, -1)
            r_sq = model.score(y_test, y_pred)
            print('coefficient of determination:', r_sq)
            predicted_temp = make_predictions_linear_regression(new_date, date_format, model, le)
            resulted_dict[city] = predicted_temp
            break
        # reshape from [samples, timesteps] into [samples, timesteps, features]
        n_features = 1

        # define model
        model = Sequential()
        model.add(LSTM(100, activation='relu', input_shape=(n_steps_in, n_features)))
        model.add(RepeatVector(n_steps_out))
        model.add(LSTM(100, activation='relu', return_sequences=True))
        model.add(TimeDistributed(Dense(1)))
        model.compile(optimizer='adam', loss='mse')

        # define input sequence
        print(n_steps_in, n_steps_out)
        print(len(y))
        if len_y > (n_steps_in + n_steps_out + n_steps_out):
            raw_seq = y[:-n_steps_out]
            y_true = y[-n_steps_out:]
            print(len(raw_seq))
            # split into samples
            X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
            print(city)
            print(type(X), print(X))
            X = X.reshape((X.shape[0], X.shape[1], n_features))
            # fit model
            model.fit(X, y, epochs=50, verbose=0)
            # demonstrate prediction
            x_input = np.array(raw_seq[-n_steps_in:])
            x_input = x_input.reshape((1, n_steps_in, n_features))
            yhat = model.predict(x_input, verbose=0)
        else:
            raw_seq = y[:]
            print(len(raw_seq))
            # split into samples
            X, y = split_sequence(raw_seq, n_steps_in, n_steps_out)
            print(city)
            print(type(X), print(X))
            X = X.reshape((X.shape[0], X.shape[1], n_features))
            # fit model
            model.fit(X, y, epochs=50, verbose=0)
            # demonstrate prediction
            x_input = np.array(raw_seq[-n_steps_in:])
            x_input = x_input.reshape((1, n_steps_in, n_features))
            yhat = model.predict(x_input, verbose=0)
            print(yhat)
        predicted_temp = make_predictions_rnn(new_date, date_format, model, raw_seq, n_steps_in)
        today = datetime.strptime(new_date, date_format)
        resulted_dict[city] = predicted_temp
    print(resulted_dict[city])
    # db.locations.find_one_and_update({"date": new_date}, {'$set': {'predicted_temp': resulted_dict}})