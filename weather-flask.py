# import the necessary packages
import flask
import ipinfo
import pyowm
from datetime import datetime
import os
from pymongo import MongoClient, ReturnDocument
from dotenv import load_dotenv, find_dotenv
load_dotenv()

# initialize our Flask application
app = flask.Flask(__name__)
ipinfo_token = os.environ['IPINFO_TOKEN']
openweatherapi_token = os.environ['OPENWEATHERAPI_TOKEN']

handler = ipinfo.getHandler(ipinfo_token)
owm = pyowm.OWM(openweatherapi_token)  # You MUST provide a valid API key


def get_db():
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
        # ip_address = "192.162.78.101"  # Ukraine
        ip_address = "198.16.78.43"  # Netherlands
        data['ip'] = ip_address
        details = handler.getDetails(ip_address)
        data['country'] = details.country_name
        data['city'] = details.city
        data['latitude'] = details.latitude
        data['longitude'] = details.longitude
        city_and_country = data['city'] + ',' + data['country']
        date_format = "%m/%d/%Y"
        new_date = datetime.strftime(datetime.now(), date_format)
        print(new_date)
        check_if_city_exists = db.locations.find_one({"date": new_date, "cities": {"$regex": city_and_country}})
        check_date = db.locations.find_one({"date": new_date})
        if check_date and check_if_city_exists:
            print("Nothing to do")
            today = db.locations.find_one({"cities": {"$regex": city_and_country}})
            find_index = today['cities'].index(city_and_country)
            data['temperature'] = today['temperatures'][find_index]
            # db.locations.delete_one({"date": new_date})
        elif check_date:
            observation = owm.weather_at_place(city_and_country)
            w = observation.get_weather()
            data['temperature'] = w.get_temperature('celsius')['temp']
            db.locations.find_one_and_update({"date": new_date},
                                             {"$push": {
                                                 "cities": city_and_country,
                                                 "temperatures": data['temperature']},
                                             '$inc': {'number_of_cities': 1}},
                                             upsert=True,
                                             return_document=ReturnDocument.AFTER)
            print('Added new city!')
            print(db.locations.find_one()['_id'])
            data['id'] = str(db.locations.find_one()['_id'])
        else:
            observation = owm.weather_at_place(city_and_country)
            w = observation.get_weather()
            data['temperature'] = w.get_temperature('celsius')['temp']
            print("Create new date")
            # db.locations.delete_one({"city": "Kharkiv"})
            db.locations.insert_one({"date": new_date, "cities": [city_and_country],
                                     "temperatures": [data['temperature']], 'number_of_cities': 1})

        # indicate that the request was a success
        data["success"] = True

    # return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__ == "__main__":
    print("please wait until server has fully started")
    db = get_db()
    app.run(host='0.0.0.0', port='8000')
