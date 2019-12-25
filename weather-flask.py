# import the necessary packages
import flask
import ipinfo
import pyowm
from datetime import datetime
import os
from dotenv import load_dotenv, find_dotenv
load_dotenv()

# initialize our Flask application
app = flask.Flask(__name__)
ipinfo_token = os.environ['IPINFO_TOKEN']
openweatherapi_token = os.environ['OPENWEATHERAPI_TOKEN']

handler = ipinfo.getHandler(ipinfo_token)
owm = pyowm.OWM(openweatherapi_token)  # You MUST provide a valid API key


def get_db():
    from pymongo import MongoClient
    client = MongoClient('localhost:27017')
    db = client.WeatherApp
    return db


@app.route("/predict/", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.headers.getlist("X-Forwarded-For"):
            ip = flask.request.headers.getlist("X-Forwarded-For")[0]
        else:
            ip = flask.request.remote_addr
        data['ip'] = ip
        ip_address = "192.162.78.101"
        data['ip'] = ip_address
        details = handler.getDetails(ip_address)
        data['country'] = details.country_name
        data['city'] = details.city
        data['latitude'] = details.latitude
        data['longitude'] = details.longitude
        observation = owm.weather_at_place(data['city'] + ',' + data['country'])
        w = observation.get_weather()
        data['temperature'] = w.get_temperature('celsius')['temp']
        # indicate that the request was a success
        data["success"] = True
        check_if_city_exists = db.locations.find_one({"city": details.city})
        if not check_if_city_exists:
            date_format = "%m/%d/%Y"
            new_date = datetime.strftime(datetime.now(), date_format)
            print(new_date)
            db.locations.insert_one({"date": new_date, "city": details.city, "temperature": data['temperature']})
            print('Added new city!')
            print(db.locations.find_one()['_id'])
            data['id'] = str(db.locations.find_one()['_id'])
        else:
            print('Remove an old city!')
            db.locations.delete_one({"city": "Kharkiv"})

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == "__main__":
    print("please wait until server has fully started")
    db = get_db()
    app.run(host='0.0.0.0', port='8000')
