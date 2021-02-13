import pytz
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from pyowm import OWM
from utils import get_db, tz_diff, if_future_day_exists, predict_for_one_city
from dotenv import load_dotenv

load_dotenv()


def update_temps(db, local_date, current_temperature, index):
     date_format = "%m/%d/%Y"
     local_date = datetime.strftime(local_date, date_format)
     local_info = db.locations.find_one({'date': local_date})
     if local_info:
         print("Date is found:" + local_date)
         local_temps = local_info['temperatures']
         local_temps[index] = current_temperature
         db.locations.find_one_and_update({"date": local_date}, {'$set': {'temperatures': local_temps}})
     else:
         print("Cannnot found date " + local_date)


def get_local_hours(db, prev_offsets, new_temperatures, prev_cities, prev_preds):
    utc_date_day = int(datetime.now(pytz.timezone('utc')).strftime('%d'))
    next_day_info, previous_day_info, dates_info = {}, {}, {}
    for index, city_offset in enumerate(prev_offsets):
        local_date = datetime.now(pytz.timezone('utc')) - timedelta(hours=city_offset)
        local_hour = int(local_date.strftime('%H'))
        local_day = int(local_date.strftime('%d'))
        city_name, current_temperature = prev_cities[index], new_temperatures[index]
        if local_day > utc_date_day:
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
        dates_info[city_name] = local_date
    return next_day_info, previous_day_info, dates_info


# Add MongoDB URL:
mongodb_url = os.environ['MONGODB_URL']
# Add tokens for API
openweatherapi_token = os.environ['OPENWEATHERAPI_TOKEN']
# Initialize third-party API
owm = OWM(openweatherapi_token)  # You MUST provide a valid API key

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
