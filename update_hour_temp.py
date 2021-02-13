import os
from sklearn import preprocessing
from datetime import datetime
from datetime import timedelta
from pyowm import OWM
import pytz
from utils import get_weather
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from dotenv import load_dotenv, find_dotenv
from utils import tz_diff, get_db
from get_city_timezones import get_local_hours, update_temps

load_dotenv()

# Add MongoDB URL:
mongodb_url = os.environ['MONGODB_URL']
# Add tokens for API
openweatherapi_token = os.environ['OPENWEATHERAPI_TOKEN']
# Initialize third-party API
owm = OWM(openweatherapi_token)  # You MUST provide a valid API key
db = get_db(mongodb_url)
tf = TimezoneFinder()
geolocator = Nominatim(user_agent='xxx')


def update_temps_hour():

    date_format = "%m/%d/%Y"
    new_date = datetime.strftime(datetime.now(pytz.timezone('utc')), date_format)
    future_day = datetime.now(pytz.timezone('utc')) + timedelta(hours=12)
    future_day = datetime.strftime(future_day, date_format)
    # For testing purpose
    # db.locations.delete_one({"date": new_date})

    get_previous_day = db.locations.find_one({'date': new_date})
    get_next_day = db.locations.find_one({'date': future_day})
    # all_dates = list(db.locations.find({}, {'date': 1, '_id': 0}))
    # all_dates = [i['date'] for i in all_dates]
    # the_most_recent_date = all_dates[-1]
    # previous_day = the_most_recent_date
    # get_previous_day = db.locations.find_one({'date': previous_day})
    # get_next_day = db.locations.find_one({'date': next_day})
    prev_cities = get_previous_day['cities']
    if get_previous_day.get('predicted_temp'):
        prev_preds = get_previous_day.get('predicted_temp')
    else:
        prev_preds = {}
    new_temperatures = get_weather(owm, prev_cities)
    # For testing:
    # new_temperatures = [23.31, 8.94, 11.23, 14.29, 11.59, 20.34, 3.65, 15.34, 8.9, 8.55, 20.72, 14.91, 19.63, 2.0, 7.53, 20.77, 15.78, 11.71, 13.18, 15.04, 11.0, 18.02, 10.72, 18.24, 10.93, 6.16, 13.6, 18.58, 11.34, 19.68, 13.88, 18.09, 14.67, 5.63, 11.4, 2.0, 6.34, 13.48, 10.06, 9.61, 21.26, -4.34, 23.39, 15.74, 14.45, 14.95, 20.52, 9.49, 28.0, 22.59, 16.54, 19.73, 19.39, 25.0, 10.3, 9.66, 9.01, 8.51, 24.0, 9.2, 20.0, 13.69, 8.55, 14.82, 11.86, 15.53, 26.39, 16.11, 10.58, 19.54, 9.28, 8.74, 7.0, 11.12, 15.56, 22.62, 26.0, 11.79, 19.0]
    new_temperatures = [int(i) for i in new_temperatures]
    cities = prev_cities
    print(cities)
    print(len(cities))
    if not get_previous_day.get('offsets'):
        print("Offsets doesn't found, create a new one")
        prev_offsets = []
        for city in prev_cities:
            city_offset = tz_diff(geolocator, tf, city)
            prev_offsets.append(city_offset)
        print(prev_offsets)
        resulted_dict = dict(zip(prev_cities, prev_offsets))
        print(resulted_dict)
        db.locations.find_one_and_update({"date": new_date}, {'$set': {'offsets': resulted_dict}})
    else:
        print("Offset found")
        prev_offsets = list(get_previous_day['offsets'].values())
        print(prev_offsets)
        # db.locations.find_one_and_update({"date": new_date}, {'$unset': {'offsets': None}})

    local_next_day, local_previous_day, dates_info = get_local_hours(db, prev_offsets, new_temperatures, prev_cities, prev_preds)

    if not get_previous_day.get('hour_temp'):
        print('Create new hour_temp')
        for city in local_previous_day.keys():
            local_date = dates_info[city]
            index = cities.index(city)
            current_temperature = list(local_previous_day[city].values())[-1]
            update_temps(db, local_date, current_temperature, index)
        db.locations.find_one_and_update({"date": new_date}, {'$set': {'hour_temp': local_previous_day}})
    else:
        print('Use existed hour_temp')
        hour_temp = get_previous_day['hour_temp']
        for city in local_previous_day.keys():
            if hour_temp.get(city):
                hour_temp[city].update(local_previous_day[city])
            else:
                hour_temp[city] = local_previous_day[city]
            local_date = dates_info[city]
            index = cities.index(city)
            current_temperature = list(local_previous_day[city].values())[-1]
            update_temps(db, local_date, current_temperature, index)
        db.locations.find_one_and_update({"date": new_date}, {'$set': {'hour_temp': hour_temp}})
        # db.locations.find_one_and_update({"date": new_date}, {'$unset': {'hour_temp': None}})
    # db.locations.find_one_and_update({"date": future_day}, {'$unset': {'hour_temp': None}})
    if local_next_day:
        if not get_next_day.get('hour_temp'):
            print('Create new hour_temp')
            for city in local_next_day.keys():
                local_date = dates_info[city]
                index = cities.index(city)
                current_temperature = list(local_next_day[city].values())[-1]
                update_temps(db, local_date, current_temperature, index)
            db.locations.find_one_and_update({"date": future_day}, {'$set': {'hour_temp': local_next_day}})
        else:
            print('Use existed hour_temp')
            hour_temp = get_next_day['hour_temp']
            print(hour_temp)
            for city in local_next_day.keys():
                if hour_temp.get(city):
                    hour_temp[city].update(local_next_day[city])
                else:
                    hour_temp[city] = local_next_day[city]
                current_temperature = list(local_next_day[city].values())[-1]
                local_date = dates_info[city]
                index = cities.index(city)
                update_temps(db, local_date, current_temperature, index)
                # print(f'for city {city} with new temp {current_temperature} and index {index}')
            db.locations.find_one_and_update({"date": future_day}, {'$set': {'hour_temp': hour_temp}})
            # db.locations.find_one_and_update({"date": new_date}, {'$unset': {'hour_temp': None}})


if __name__ == "__main__":
    update_temps_hour()
