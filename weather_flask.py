# import the necessary packages
import flask
from flask_caching import Cache
import ipinfo
import pyowm
from datetime import datetime
from datetime import timedelta
import os
from pymongo import MongoClient, ReturnDocument
from dotenv import load_dotenv, find_dotenv
load_dotenv()

# See prints from docker console
os.environ['PYTHONUNBUFFERED'] = '1'
# initialize our Flask application
app = flask.Flask(__name__)
# Add caching for app
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
# Add for hot-reload
os.environ['FLASK_APP'] = "app"
os.environ['FLASK_ENV'] = "development"
port = os.environ['PORT']
# Add MongoDB URL:
mongodb_url = os.environ['MONGODB_URL']
# Add tokens for API
ipinfo_token = os.environ['IPINFO_TOKEN']
openweatherapi_token = os.environ['OPENWEATHERAPI_TOKEN']
# Initialize third-party API
handler = ipinfo.getHandler(ipinfo_token)
owm = pyowm.OWM(openweatherapi_token)  # You MUST provide a valid API key


def get_db():
    client = MongoClient(mongodb_url)
    db = client.WeatherApp
    return db


@app.route("/predict/", methods=["POST"])
@cache.cached(timeout=50)
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.headers.getlist("X-Forwarded-For"):
            ip_address = flask.request.headers.getlist("X-Forwarded-For")[0]
        else:
            ip_address = flask.request.remote_addr
        data['ip'] = ip_address
        print(data['ip'])
        # If testing from localhost or inside docker-compose, change IP address to a more suitable one
        if ip_address == "127.0.0.1" or ip_address == "172.17.0.1":
            ip_address = "192.162.78.101"  # Ukraine
            # ip_address = "178.150.147.148"  # another Ukraine address
            # ip_address = "198.16.78.43"  # Netherlands
        data['ip'] = ip_address
        check_ip_address = db.locations.find_one({"ip_addresses": {"$regex": ip_address}})
        if check_ip_address:
            print("Found IP adress")
            find_index = check_ip_address['ip_addresses'].index(ip_address)
            data['country'] = check_ip_address['cities'][find_index].split(",")[1]
            data['city'] = check_ip_address['cities'][find_index].split(",")[0]
        else:
            print("Call API to find city from an IP adress")
            details = handler.getDetails(ip_address)
            data['country'] = details.country_name
            data['city'] = details.city
        city_and_country = data['city'] + ',' + data['country']

        date_format = "%m/%d/%Y"
        new_date = datetime.strftime(datetime.now(), date_format)
        print(new_date)
        check_if_city_exists = db.locations.find_one({"date": new_date, "cities": {"$regex": city_and_country}})
        check_if_city_exists_but_no_ip = db.locations.find_one({"date": new_date,
                                                                "cities": {"$regex": city_and_country},
                                                                "ip_addresses": {"$ne": ip_address}})
        print(f"Check for ip: {check_if_city_exists_but_no_ip}")
        if check_if_city_exists and check_if_city_exists_but_no_ip:
            print(f"Add current IP: {ip_address} to city: {city_and_country}")
            today = db.locations.find_one({"date": new_date, "cities": {"$regex": city_and_country}})
            find_index = today['cities'].index(city_and_country)
            print(f"Current index: {find_index}")
            data['temperature'] = today['temperatures'][find_index]
            current_ip = today['ip_addresses'][find_index]
            print(current_ip)
            new_ip = current_ip + [ip_address]
            print(new_ip)
            one = db.locations.find_one_and_update({"date": new_date},
                                                   {'$push': {'ip_addresses.$[element]': ip_address}},
                                                   array_filters=[{"element": {'$eq': current_ip}}],
                                                   upsert=True,
                                                   return_document=ReturnDocument.AFTER)
            print(one)
            # db.locations.delete_one({"date": new_date})
        check_date = db.locations.find_one({"date": new_date})
        if check_date and check_if_city_exists:
            print("Nothing to do")
            today = db.locations.find_one({"cities": {"$regex": city_and_country}})
            find_index = today['cities'].index(city_and_country)
            data['temperature'] = today['temperatures'][find_index]
            # db.locations.delete_one({"date": new_date})
        elif check_date:
            print("Call API to get weather")
            observation = owm.weather_at_place(city_and_country)
            w = observation.get_weather()
            data['temperature'] = w.get_temperature('celsius')['temp']
            db.locations.find_one_and_update({"date": new_date},
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
        else:
            print("Call API to get weather")
            observation = owm.weather_at_place(city_and_country)
            w = observation.get_weather()
            data['temperature'] = w.get_temperature('celsius')['temp']
            print("Create new date")
            # db.locations.delete_one({"city": "Kharkiv"})
            db.locations.insert_one({"date": new_date, "cities": [city_and_country], "ip_addresses": [[ip_address]],
                                     "temperatures": [data['temperature']], 'number_of_cities': 1})
        data["predict_temp"] = db.locations.find_one({'date': new_date})['predicted_temp'][city_and_country]
        data["today"] = new_date
        # indicate that the request was a success
        data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == "__main__":
    print("please wait until server has fully started")
    db = get_db()

    app.run(host='0.0.0.0', port=port)
