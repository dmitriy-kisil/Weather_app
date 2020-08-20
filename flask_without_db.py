# import the necessary packages
import os
from datetime import datetime
from datetime import timedelta
import flask
from flask_caching import Cache
import ipinfo
import pyowm
from utils import get_db
from pymongo import MongoClient, ReturnDocument
from dotenv import load_dotenv, find_dotenv

load_dotenv()

# initialize our Flask application
app = flask.Flask(__name__)
# Add caching for app
cache = Cache(app, config={'CACHE_TYPE': 'simple'})
port = os.environ['PORT']
# Add MongoDB URL:
mongodb_url = os.environ['MONGODB_URL']
# Add tokens for API
ipinfo_token = os.environ['IPINFO_TOKEN']
openweatherapi_token = os.environ['OPENWEATHERAPI_TOKEN']
# Initialize third-party API
handler = ipinfo.getHandler(ipinfo_token)
owm = pyowm.OWM(openweatherapi_token)  # You MUST provide a valid API key


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
        # If testing from localhost or inside docker-compose, change IP address to a more suitable one
        if ip_address == "127.0.0.1" or ip_address == "172.17.0.1":
            ip_address = "192.162.78.101"  # Ukraine
        date_format = "%m/%d/%Y"
        new_date = datetime.strftime(datetime.now(), date_format)
        data['ip'] = ip_address
        print("Call API to find city from an IP adress")
        details = handler.getDetails(ip_address)
        data['country'] = details.country_name
        data['city'] = details.city
        city_and_country = data['city'] + ',' + data['country']
        mgr = owm.weather_manager()
        weather = mgr.weather_at_place(city_and_country).weather
        dump_dict = weather.temperature(unit='celsius')['temp']
        data['data'] = dump_dict
        data["today"] = new_date
        # indicate that the request was a success
        data["success"] = True
        response = flask.jsonify(data)
        response.headers.add("Access-Control-Allow-Origin", "*")
        # return the data dictionary as a JSON response
        return response


if __name__ == "__main__":
    print("please wait until server has fully started")
    db = get_db(mongodb_url)
    app.run(host='0.0.0.0', port=port)
