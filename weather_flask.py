# import the necessary packages
import os
from datetime import datetime
from datetime import timedelta
import flask
from flask_caching import Cache
import ipinfo
import pyowm
from utils import get_db
from create_new_day3 import create_a_new_day, predict_for_one_city
from update_hour_temp import update_temps_hour
import pytz
from pymongo import MongoClient, ReturnDocument
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from utils import tz_diff, if_future_day_exists, dummy_hour_temp
from dotenv import load_dotenv, find_dotenv

load_dotenv()

# initialize our Flask application
app = flask.Flask(__name__)
# Add caching for app
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
# Add MongoDB URL:
mongodb_url = os.environ['MONGODB_URL']
# Add tokens for API
ipinfo_token = os.environ['IPINFO_TOKEN']
openweatherapi_token = os.environ['OPENWEATHERAPI_TOKEN']
# Initialize third-party API
handler = ipinfo.getHandler(ipinfo_token)
owm = pyowm.OWM(openweatherapi_token)  # You MUST provide a valid API key
db = get_db(mongodb_url)
geolocator = Nominatim(user_agent='xxx')
tf = TimezoneFinder()


@app.route("/predict/", methods=["POST"])
@cache.cached(timeout=50)
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.headers.getlist("X-Forwarded-For"):
            ip_address = flask.request.headers.getlist("X-Forwarded-For")[0]
            if "," in ip_address:
                ip_address = ip_address.split(",")[0]
        else:
            ip_address = flask.request.remote_addr
        # If testing from localhost or inside docker-compose, change IP address to a more suitable one
        if ip_address == "127.0.0.1" or ip_address == "172.17.0.1":
            ip_address = "192.162.78.101"  # Ukraine
        date_format = "%m/%d/%Y"
        new_date = datetime.strftime(datetime.now(), date_format)
        print(new_date)
        data['ip'] = ip_address
        print(data['ip'])
        check_date = db.locations.find_one({"date": new_date})
        check_ip_address = check_date
        new_query = None

        ip_address_exists_in_db = False
        if check_ip_address:
            if check_ip_address.get('ip_addresses'):
                if any(ip_address in sl for sl in check_ip_address['ip_addresses']):
                    ip_address_exists_in_db = True
        # ip_address_exists_in_db = False  # for testing
        if ip_address_exists_in_db is True:
            print("Found IP adress")
            for c, v in enumerate(check_ip_address['ip_addresses']):
                if ip_address in v:
                    find_index = c
            data['country'] = check_ip_address['cities'][find_index].split(",")[1]
            data['city'] = check_ip_address['cities'][find_index].split(",")[0]
        else:
            print("Call API to find city from an IP adress")
            details = handler.getDetails(ip_address)
            data['country'] = details.country_name
            data['city'] = details.city
        city_and_country = data['city'] + ',' + data['country']
        offset = tz_diff(geolocator, tf, city_and_country)
        future_day_exists = if_future_day_exists(offset)
        local_date_not_str = datetime.now(pytz.timezone('utc')) - timedelta(hours=offset)
        date_format = "%m/%d/%Y"
        local_date = datetime.strftime(local_date_not_str, date_format)
        local_hour = int(datetime.strftime(local_date_not_str, "%H"))
        local_hour_str = str(local_hour) + '_hour'
        print(local_date)
        check_date = db.locations.find_one({"date": local_date})
        check_if_city_exists = check_date
        city_exists = False
        if check_if_city_exists:
            if check_if_city_exists.get('cities'):
                if city_and_country not in check_if_city_exists['cities']:
                    city_exists = False
                else:
                    city_exists = True
        # city_exists = False  # for testing
        no_ip = True
        if city_exists:
            if check_if_city_exists.get('ip_addresses'):
                if any(ip_address in sl for sl in check_if_city_exists['ip_addresses']):
                    no_ip = False
        print(f"Check for ip: {check_if_city_exists}")
        if city_exists and no_ip:
            print(f"Add current IP: {ip_address} to city: {city_and_country}")
            today = check_if_city_exists
            for c, v in enumerate(today['cities']):
                if city_and_country in v:
                    find_index = c
            print(f"Current index: {find_index}")
            data['temperature'] = today['temperatures'][find_index]
            current_ip = today['ip_addresses'][find_index]
            print(current_ip)
            new_ip = current_ip + [ip_address]
            print(new_ip)
            one = db.locations.find_one_and_update({"date": local_date},
                                                   {'$addToSet': {'ip_addresses.$[element]': ip_address}},
                                                   array_filters=[{"element": {'$eq': current_ip}}],
                                                   upsert=True,
                                                   return_document=ReturnDocument.AFTER)
            # print(one)
            if future_day_exists:
                print('Update next day')
                future_date = datetime.now(pytz.timezone('utc')) + timedelta(hours=12)
                future_date = datetime.strftime(future_date, date_format)
                one = db.locations.find_one_and_update({"date": future_date},
                                                       {'$addToSet': {'ip_addresses.$[element]': ip_address}},
                                                       array_filters=[{"element": {'$eq': current_ip}}],
                                                       upsert=True,
                                                       return_document=ReturnDocument.AFTER)

        if check_date and city_exists:
            print("Nothing to do")
            today = check_if_city_exists
            find_index = today['cities'].index(city_and_country)
            data['temperature'] = today['temperatures'][find_index]
            data_temp = db.locations.find_one({"date": local_date, "cities": {"$regex": city_and_country}})
            data["predict_temp"] = data_temp['predicted_temp'][city_and_country]
            data['hour_temp'] = data_temp['hour_temp'][city_and_country]
        if check_date and not city_exists:
            print("Call API to get weather")
            mgr = owm.weather_manager()
            weather = mgr.weather_at_place(city_and_country).weather
            data['temperature'] = weather.temperature(unit='celsius')['temp']
            hour_temp_for_first_day = dummy_hour_temp(int(data['temperature']))
            db.locations.find_one_and_update({"date": local_date},
                                             {"$set": {"offsets." + city_and_country: offset,
                                                       "hour_temp." + city_and_country:
                                                           hour_temp_for_first_day}})
            db.locations.find_one_and_update({"date": local_date},
                                             {"$push": {
                                                 "cities": city_and_country,
                                                 "temperatures": data['temperature'],
                                                 "ip_addresses": [data['ip']]},
                                                 '$inc': {'number_of_cities': 1}},
                                             upsert=True,
                                             return_document=ReturnDocument.AFTER)
            if future_day_exists:
                print('Update next day')
                future_date = datetime.now(pytz.timezone('utc')) + timedelta(hours=12)
                future_date = datetime.strftime(future_date, date_format)
                hour_temp_for_first_day = dummy_hour_temp(int(data['temperature']))
                db.locations.find_one_and_update({"date": future_date},
                                                 {"$set": {"offsets." + city_and_country: offset,
                                                           "hour_temp." + city_and_country:
                                                               hour_temp_for_first_day}})
                db.locations.find_one_and_update({"date": future_date},
                                                 {"$push": {
                                                     "cities": city_and_country,
                                                     "temperatures": data['temperature'],
                                                     "ip_addresses": [data['ip']]},
                                                     '$inc': {'number_of_cities': 1}},
                                                 upsert=True,
                                                 return_document=ReturnDocument.AFTER)
            print('Added new city!')
            print(db.locations.find_one()['_id'])
            data['id'] = str(db.locations.find_one()['_id'])
            print("Don't found predicted temperatures, create a new one")
            get_previous_day = db.locations.find_one({"date": local_date})
            if get_previous_day.get('predicted_temp'):
                prev_preds = get_previous_day.get('predicted_temp')
            else:
                prev_preds = {}
            predict_for_one_city(db, local_date_not_str, prev_preds, city_and_country)
            data_temp = db.locations.find_one({"date": local_date, "cities": {"$regex": city_and_country}})
            data["predict_temp"] = data_temp['predicted_temp'][city_and_country]
            data['hour_temp'] = data_temp['hour_temp'][city_and_country]
        data.pop("id", None)
        data["today"] = local_date
        # db.locations.delete_one({"date": new_date})  # for testing
        # indicate that the request was a success
        data["success"] = True
    response = flask.jsonify(data)
    response.headers.add("Access-Control-Allow-Origin", "*")
    # return the data dictionary as a JSON response
    return response


@app.route("/update_temp_hour/", methods=["GET"])
def update_temp_hour():
    if flask.request.headers.get("X-Appengine-Cron"):
        update_temps_hour()
        return '200'
    else:
        return '403'


@app.route("/create_new_day/", methods=["GET"])
def create_new_day():
    if flask.request.headers.get("X-Appengine-Cron"):
        create_a_new_day()
        return '200'
    else:
        return '403'


if __name__ == "__main__":
    print("please wait until server has fully started")
    app.run()
