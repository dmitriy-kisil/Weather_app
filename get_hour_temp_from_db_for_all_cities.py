import pandas as pd
import os
from datetime import datetime
from utils import get_db
import pyowm
from dotenv import load_dotenv, find_dotenv

load_dotenv()

# Add MongoDB URL:
mongodb_url = os.environ['MONGODB_URL']
# Add tokens for API
openweatherapi_token = os.environ['OPENWEATHERAPI_TOKEN']
# Initialize third-party API
owm = pyowm.OWM(openweatherapi_token)  # You MUST provide a valid API key


def create_example(date):
    month, day, year = date.split('/')
    example = pd.DataFrame({"day": [day for i in range(24)],
                            "month": [month for i in range(24)],
                            "year": [year for i in range(24)],
                            "hour": [i for i in range(24)]})
    return example


if __name__ == "__main__":

    db = get_db(mongodb_url)
    date_format = "%m/%d/%Y"
    new_date = datetime.strftime(datetime.now(), date_format)
    # new_date = datetime.strftime(datetime.now()-timedelta(days=1), date_format)
    # db.locations.delete_one({"date": new_date})
    # from bson.objectid import ObjectId
    # result = db.locations.delete_one({'_id': ObjectId(str)})
    all_dates = list(db.locations.find({}, {'date': 1, '_id': 0}))
    all_dates = [i['date'] for i in all_dates]
    get_previous_day = db.locations.find_one({'date': new_date})
    number_of_cities_to_parse = len(get_previous_day['cities'])
    prev_cities = get_previous_day['cities'][:number_of_cities_to_parse]
    # create first dummy row
    df = pd.DataFrame({"city": [0], "day": [0], "month": [0], "year": [0], "hour": [0], "temp": [0]})
    index = 0
    # filter out three days from previous year plus future day which can or can't contain hour_temp for all the cities
    selected_dates = [i for i in all_dates if i > '09/10/2020'][3:-1]
    print(selected_dates)
    for selected_date in selected_dates:
        item = db.locations.find_one({'date': selected_date})
        cities = item['cities']
        month, day, year = selected_date.split('/')
        for city in cities:
            hour_temp = item['hour_temp'][city]
            for hour in list(hour_temp.keys()):
                index += 1
                temp = hour_temp[hour]
                hour = int(hour.replace("_hour", ""))
                new_row = {"city": city, "day": day, "month": month, "year": year, "hour": hour, 'temp': temp}
                new_row = pd.Series(data=new_row, name=index)
                df = df.append(new_row, ignore_index=False)
    # remove first dummy row
    df = df.iloc[1:]
    df.to_csv('city_data2.csv')
