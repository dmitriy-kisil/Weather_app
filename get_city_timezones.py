import pytz
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
import pyowm
from utils import get_db, tz_diff, if_future_day_exists
from dotenv import load_dotenv
from create_new_day3 import predict_for_one_city

load_dotenv()


def update_temps(db, local_date, current_temperature, index):
     date_format = "%m/%d/%Y"
     local_date = datetime.strftime(local_date, date_format)
     local_info = db.locations.find_one({'date': local_date})
     local_temps = local_info['temperatures']
     local_temps[index] = current_temperature
     db.locations.find_one_and_update({"date": local_date}, {'$set': {'temperatures': local_temps}})


def get_local_hours(db, prev_offsets, new_temperatures, prev_cities, prev_preds):
    new_date_hour = int(datetime.now(pytz.timezone('utc')).strftime('%H'))
    new_date_day = int(datetime.now(pytz.timezone('utc')).strftime('%d'))
    next_day_info, previous_day_info = {}, {}
    for index, city_offset in enumerate(prev_offsets):
        local_hour = new_date_hour - city_offset
        local_date = datetime.now(pytz.timezone('utc')) - timedelta(hours=city_offset)
        city_name, current_temperature = prev_cities[index], new_temperatures[index]
        if local_hour < 11 - city_offset:
            local_hour_str = str(local_hour) + '_hour'
            next_day_info[city_name] = {local_hour_str: current_temperature}
        else:
            local_hour_str = str(local_hour) + '_hour'
            previous_day_info[city_name] = {local_hour_str: current_temperature}
            if local_hour_str == '12_hour':
                print(f'Run prediction for {city_name}')
                update_temps(db, local_date, current_temperature, index)
                predict_for_one_city(db, local_date, prev_preds, city_name)
        if not prev_preds.get(city_name):
            print(f'Run prediction for {city_name}')
            update_temps(db, local_date, current_temperature, index)
            predict_for_one_city(db, local_date, prev_preds, city_name)
    return next_day_info, previous_day_info


# Add MongoDB URL:
mongodb_url = os.environ['MONGODB_URL']
# Add tokens for API
openweatherapi_token = os.environ['OPENWEATHERAPI_TOKEN']
# Initialize third-party API
owm = pyowm.OWM(openweatherapi_token)  # You MUST provide a valid API key

geolocator = Nominatim(user_agent='xxx')
tf = TimezoneFinder()
cityname = 'Kharkiv'


if __name__ == "__main__":

    db = get_db(mongodb_url)
    date_format = "%m/%d/%Y"
    new_date = datetime.strftime(datetime.now(), date_format)
    new_date_hour = int(datetime.now(pytz.timezone('utc')).strftime('%H'))
    print(new_date_hour)
    day = db.locations.find_one({'date': new_date})
    cities = day['cities']
    offsets = []
    for city in cities[:]:
        city_offset = tz_diff(geolocator, tf, city)
        offsets.append(city_offset)
    print(offsets)
    local_hours = get_local_hours(new_date_hour, offsets)
    print(local_hours)
