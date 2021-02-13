import pandas as pd
import os
from datetime import datetime
from utils import get_db
from pyowm import OWM
from dotenv import load_dotenv, find_dotenv

load_dotenv()

# Add MongoDB URL:
mongodb_url = os.environ['MONGODB_URL']
# Add tokens for API
openweatherapi_token = os.environ['OPENWEATHERAPI_TOKEN']
# Initialize third-party API
owm = OWM(openweatherapi_token)  # You MUST provide a valid API key


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
