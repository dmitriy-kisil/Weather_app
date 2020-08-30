import pytz
import os
from datetime import datetime
from utils import get_db
import pyowm
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from dotenv import load_dotenv

load_dotenv()


def func(geolocator, tf, cityname):
    # Get lat, long from city name
    location = geolocator.geocode(cityname)
    # Get timezone from coordinates
    latitude, longitude = location.latitude, location.longitude
    # Timezone
    datez = tf.timezone_at(lng=longitude, lat=latitude)
    datez = str(datez)
    globalDate = datetime.now(pytz.timezone(datez))
    print("The date in " + str(cityname) + " is: " + globalDate.strftime('%A, %m/%d/%y %H:%M:%S %Z %z'))

# Add MongoDB URL:
mongodb_url = os.environ['MONGODB_URL']
# Add tokens for API
openweatherapi_token = os.environ['OPENWEATHERAPI_TOKEN']
# Initialize third-party API
owm = pyowm.OWM(openweatherapi_token)  # You MUST provide a valid API key

geolocator = Nominatim(user_agent='xxx')
tf = TimezoneFinder()
cityname = 'Kharkiv'


if __name__ == "__main__":

    db = get_db(mongodb_url)
    date_format = "%m/%d/%Y"
    new_date = datetime.strftime(datetime.now(), date_format)
    day = db.locations.find_one({'date': new_date})
    cities = day['cities']
    for city in cities:
        city = city.split(',')[0]
        func(geolocator, tf, city)
