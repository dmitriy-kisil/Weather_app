import pandas as pd
import os
import time
from datetime import datetime, timedelta
from utils import get_db
from pyowm import OWM
import pytz
from dotenv import load_dotenv, find_dotenv
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

load_dotenv()

# Add MongoDB URL:
mongodb_url = os.environ['MONGODB_URL']
# Add tokens for API
openweatherapi_token = os.environ['OPENWEATHERAPI_TOKEN']
# Initialize third-party API
owm = OWM(openweatherapi_token)  # You MUST provide a valid API key


def create_example(date):
    month, day, year = date.split('/')
    example = pd.DataFrame({"day": [day for i in range(24)],
                            "month": [month for i in range(24)],
                            "year": [year for i in range(24)],
                            "hour": [i for i in range(24)]
                            })
    return example


def save_hour_temp_from_db_to_csv(db, filename):
    # new_date = datetime.strftime(datetime.now()-timedelta(days=1), date_format)
    # db.locations.delete_one({"date": new_date})
    # from bson.objectid import ObjectId
    # result = db.locations.delete_one({'_id': ObjectId(str)})
    all_dates = list(db.locations.find({}, {'date': 1, '_id': 0}))
    all_dates = [i['date'] for i in all_dates]
    index = 0
    # filter out three days from previous year plus future day which can or can't contain hour_temp for all the cities
    # selected_dates = [i for i in all_dates if i > '09/10/2020'][3:]
    selected_dates = [i for i in all_dates]
    selected_dates = selected_dates[::-1]
    print(selected_dates)
    all_info = db.locations.find({'date': {'$in': selected_dates}}, {'date': 1, 'cities': 1, 'hour_temp': 1})
    start = time.time()
    d = {'city': [], 'day': [], 'month': [], 'year': [], 'hour': [], 'temp': []}
    for item in all_info:
        cities = item['cities']
        selected_date = item['date']
        month, day, year = selected_date.split('/')
        for city in cities:
            if not item.get('hour_temp') or not item['hour_temp'].get(city):
                continue
            hour_temp = item['hour_temp'][city]
            for hour in list(hour_temp.keys()):
                temp = int(hour_temp[hour])
                hour = int(hour.replace("_hour", ""))
                d['city'].append(city)
                d['day'].append(day)
                d['month'].append(month)
                d['year'].append(year)
                d['hour'].append(hour)
                d['temp'].append(temp)
                index += 1
    print(f'Elapsed: {time.time() - start}')
    df = pd.DataFrame({'city': pd.Series(d['city'], dtype='category'),
                       'day': pd.Series(d['day'], dtype='int8'),
                       'month': pd.Series(d['month'], dtype='int8'),
                       'year': pd.Series(d['year'], dtype='int16'),
                       'hour': pd.Series(d['hour'], dtype='int8'),
                       'temp': pd.Series(d['temp'], dtype='int8')})
    return df


def get_polynomial_predictions(resulted_dict, city, df, map_to_int, degree, date_to_predict):
    df1 = df[df['city'] == map_to_int[city]]
    X = df1[['day', 'month', 'year', 'hour']]

    y = df1['temp']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    poly_features = PolynomialFeatures(degree=degree)

    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.fit_transform(X_test)

    # fit the transformed features to Linear Regression
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, y_train)

    # predicting on training data-set
    y_train_predict = poly_model.predict(X_train_poly)

    # make a prediction set using the test set
    y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))
    y_test_predict = [int(i) for i in y_test_predict]
    df2 = create_example(date_to_predict)

    prediction2 = poly_model.predict(poly_features.fit_transform(df2))
    prediction2 = [int(i) for i in prediction2]
    df2['temp_pred'] = prediction2
    print(city)
    # Evaluate the prediction accuracy of the model
    print("The Explained Variance train: %.2f" % poly_model.score(X_train_poly, y_train))
    print("The Mean Absolute Error train: %.2f degrees celsius" % mean_absolute_error(y_train, y_train_predict))
    print("The Median Absolute Error train: %.2f degrees celsius" % median_absolute_error(y_train, y_train_predict))
    print("r2 score train: " + str(r2_score(y_train, y_train_predict)))
    print("The Explained Variance test: %.2f" % poly_model.score(X_test_poly, y_test))
    print("The Mean Absolute Error test: %.2f degrees celsius" % mean_absolute_error(y_test, y_test_predict))
    print("The Median Absolute Error test: %.2f degrees celsius" % median_absolute_error(y_test, y_test_predict))
    print("r2 score test: " + str(r2_score(y_test, y_test_predict)))
    d1 = {}
    for i in range(24):
        d1[str(i) + "_hour"] = prediction2[i]
    print(d1)
    resulted_dict[city] = d1
    return resulted_dict


def get_linear_regression_predictions(resulted_dict, city, df, map_to_int, date_to_predict):
    df1 = df[df['city'] == map_to_int[city]]

    X = df1[['day', 'month', 'year', 'hour']]

    y = df1['temp']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # instantiate the regressor class
    regressor = LinearRegression()

    # fit the build the model by fitting the regressor to the training data
    regressor.fit(X_train, y_train)

    # predicting on training data-set
    y_train_predict = regressor.predict(X_train)

    # predicting on test data-set
    y_test_predict = regressor.predict(X_test)

    df2 = create_example(date_to_predict)

    prediction2 = regressor.predict(df2)
    prediction2 = [int(i) for i in prediction2]
    df2['temp_pred'] = prediction2
    # df2.to_csv('kharkiv_data.csv')
    # print(X_test.head())
    print(city)
    # Evaluate the prediction accuracy of the model
    print("The Explained Variance train: %.2f" % regressor.score(X_train, y_train))
    print("The Mean Absolute Error train: %.2f degrees celsius" % mean_absolute_error(y_train, y_train_predict))
    print("The Median Absolute Error train: %.2f degrees celsius" % median_absolute_error(y_train, y_train_predict))
    print("r2 score train: " + str(r2_score(y_train, y_train_predict)))
    print("The Explained Variance test: %.2f" % regressor.score(X_test, y_test))
    print("The Mean Absolute Error test: %.2f degrees celsius" % mean_absolute_error(y_test, y_test_predict))
    print("The Median Absolute Error test: %.2f degrees celsius" % median_absolute_error(y_test, y_test_predict))
    print("r2 score test: " + str(r2_score(y_test, y_test_predict)))
    d1 = {}
    for i in range(24):
        d1[str(i) + "_hour"] = prediction2[i]
    print(d1)
    resulted_dict[city] = d1
    return resulted_dict


if __name__ == "__main__":
    hours = 12
    hours_added = timedelta(hours=hours)
    future_date_and_time = datetime.now(pytz.timezone('utc')) + hours_added
    print(future_date_and_time)
    filename = "city_data2.csv"
    db = get_db(mongodb_url)
    date_format = "%m/%d/%Y"
    new_date = datetime.strftime(future_date_and_time, date_format)
    df = save_hour_temp_from_db_to_csv(db, filename)
    # df = pd.read_csv(filename)
    cities = df['city'].unique()
    map_to_int = dict(zip(list(cities), list(range(len(cities)))))
    print(map_to_int)
    date_to_predict = new_date
    df['city'] = df['city'].map(map_to_int)
    resulted_dict = {}
    # cities = [cities[1]]
    '''
    for city in cities:
        # X = df[['city', 'day', 'month', 'year', 'hour']]
        df1 = df[df['city'] == map_to_int[city]]
        print(len(df1))
        X = df1[['day', 'month', 'year', 'hour']]

        # y = df['temp'].astype(int)
        y = df1['temp'].astype(int)
        degree = 3
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        poly_features = PolynomialFeatures(degree=degree)
        # # instantiate the regressor class
        # regressor = LinearRegression()

        # fit the build the model by fitting the regressor to the training data
        # regressor.fit(X_train, y_train)
        # transforms the existing features to higher degree features.
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.fit_transform(X_test)

        # fit the transformed features to Linear Regression
        poly_model = LinearRegression()
        poly_model.fit(X_train_poly, y_train)

        # predicting on training data-set
        y_train_predict = poly_model.predict(X_train_poly)

        # predicting on test data-set
        y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))
        # # make a prediction set using the test set
        # prediction = regressor.predict(X_test)
        # prediction = [int(i) for i in prediction]
        # df2 = create_example(date_to_predict)
        # prediction2 = regressor.predict(df2)
        # prediction2 = [int(i) for i in prediction2]
        # df2['temp_pred'] = prediction2
        # make a prediction set using the test set
        y_test_predict = poly_model.predict(poly_features.fit_transform(X_test))
        y_test_predict = [int(i) for i in y_test_predict]
        df2 = create_example(date_to_predict)
        # print(poly_features.fit_transform(df2))
        prediction2 = poly_model.predict(poly_features.fit_transform(df2))
        prediction2 = [int(i) for i in prediction2]
        df2['temp_pred'] = prediction2
        # df2.to_csv('kharkiv_data.csv')
        print(X_test.head())
        print(city)
        # # Evaluate the prediction accuracy of the model
        # print("The Explained Variance: %.2f" % regressor.score(X_test, y_test))
        # print("The Mean Absolute Error: %.2f degrees celsius" % mean_absolute_error(y_test, prediction))
        # print("The Median Absolute Error: %.2f degrees celsius" % median_absolute_error(y_test, prediction))
        print("The Explained Variance train: %.2f" % poly_model.score(X_train_poly, y_train))
        print("The Mean Absolute Error train: %.2f degrees celsius" % mean_absolute_error(y_train, y_train_predict))
        print("The Median Absolute Error train: %.2f degrees celsius" % median_absolute_error(y_train, y_train_predict))
        print("r2 score train: " + str(r2_score(y_train, y_train_predict)))
        print("The Explained Variance test: %.2f" % poly_model.score(X_test_poly, y_test))
        print("The Mean Absolute Error test: %.2f degrees celsius" % mean_absolute_error(y_test, y_test_predict))
        print("The Median Absolute Error test: %.2f degrees celsius" % median_absolute_error(y_test, y_test_predict))
        print("r2 score test: " + str(r2_score(y_test, y_test_predict)))
        d1 = {}
        for i in range(24):
            d1[str(i) + "_hour"] = prediction2[i]
        print(d1)
        resulted_dict[city] = d1
    '''

    for city in cities:
        # resulted_dict = get_linear_regression_predictions(resulted_dict, city, df, map_to_int, date_to_predict)
        resulted_dict = get_polynomial_predictions(resulted_dict, city, df, map_to_int, 3, date_to_predict)
    # print(resulted_dict)
