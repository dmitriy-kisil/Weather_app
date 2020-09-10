import os
from sklearn import preprocessing
from datetime import datetime
from datetime import timedelta
import pyowm
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
owm = pyowm.OWM(openweatherapi_token)  # You MUST provide a valid API key
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

    local_next_day, local_previous_day = get_local_hours(db, prev_offsets, new_temperatures, prev_cities, prev_preds)

    if not get_previous_day.get('hour_temp'):
        print('Create new hour_temp')
        db.locations.find_one_and_update({"date": new_date}, {'$set': {'hour_temp': local_previous_day}})
    else:
        print('Use existed hour_temp')
        hour_temp = get_previous_day['hour_temp']
        for city in local_previous_day.keys():
            if hour_temp.get(city):
                hour_temp[city].update(local_previous_day[city])
            else:
                hour_temp[city] = local_previous_day[city]
        db.locations.find_one_and_update({"date": new_date}, {'$set': {'hour_temp': hour_temp}})
        # db.locations.find_one_and_update({"date": new_date}, {'$unset': {'hour_temp': None}})
    # db.locations.find_one_and_update({"date": future_day}, {'$unset': {'hour_temp': None}})
    if local_next_day:
        if not get_next_day.get('hour_temp'):
            print('Create new hour_temp')
            print(local_next_day)
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
            db.locations.find_one_and_update({"date": future_day}, {'$set': {'hour_temp': hour_temp}})
            # db.locations.find_one_and_update({"date": new_date}, {'$unset': {'hour_temp': None}})


if __name__ == "__main__":
    update_temps_hour()
