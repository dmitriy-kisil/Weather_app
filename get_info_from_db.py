import pandas as pd
import os
from datetime import datetime
from weather_flask import get_db
import pyowm

# Add tokens for API
openweatherapi_token = os.environ['OPENWEATHERAPI_TOKEN']
# Initialize third-party API
owm = pyowm.OWM(openweatherapi_token)  # You MUST provide a valid API key


def get_weather(cities):
    print("Call API to get weather")
    cities_temperatures = []
    for city in cities:
        observation = owm.weather_at_place(city)
        w = observation.get_weather()
        temperature_from_weather = w.get_temperature('celsius')['temp']
        cities_temperatures.append(temperature_from_weather)
    print(f'New temps are: {cities_temperatures}')
    return cities_temperatures


if __name__ == "__main__":

    db = get_db()
    date_format = "%m/%d/%Y"
    new_date = datetime.strftime(datetime.now(), date_format)
    all_dates = list(db.locations.find({}, {'date': 1, '_id': 0}))
    all_dates = [i['date'] for i in all_dates]
    get_previous_day = db.locations.find_one({'date': new_date})
    number_of_cities_to_parse = 2
    prev_cities = get_previous_day['cities'][:number_of_cities_to_parse]
    temperatures = {}
    for i in all_dates:
        item = db.locations.find_one({'date': i})
        item_cities = item['cities']
        temperatures_item = []
        for k in prev_cities:
            if k in item_cities:
                find_index = item_cities.index(k)
                temperatures_item.append(item['temperatures'][find_index])
            else:
                temperatures_item.append(None)
        temperatures[i] = temperatures_item
    print(temperatures)
    df = pd.DataFrame.from_dict(
        temperatures, orient='index', columns=get_previous_day['cities'][:number_of_cities_to_parse])
    print(df.head())
    df.to_csv('city_data.csv')
