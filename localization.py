import os
from datetime import datetime
from utils import get_db
from pyowm import OWM
from pyowm.utils.config import get_default_config
import pycountry
import gettext
import babel
from googletrans import Translator
from dotenv import load_dotenv, find_dotenv

load_dotenv()

# Add MongoDB URL:
mongodb_url = os.environ['MONGODB_URL']
# Add tokens for API
openweatherapi_token = os.environ['OPENWEATHERAPI_TOKEN']
# Initialize third-party API
# owm = OWM(openweatherapi_token)  # You MUST provide a valid API key
config_dict = get_default_config()
config_dict['language'] = 'en'  # your language here
owm = OWM(openweatherapi_token, config_dict)

def translate_locale(country_name):
    country = pycountry.countries.get(name=country_name)
    if not country:
        country = [i for i in pycountry.countries if country_name in i.name][0]
    print(country)
    language = babel.Locale.parse('und_' + country.alpha_2).language
    print(language)
    translator = Translator()
    if language == 'en':
        return translator, language
    # translator = gettext.translation('iso3166', pycountry.LOCALES_DIR,
    #                                  languages=[language])
    # translator.install()
    return translator, language


if __name__ == "__main__":
    db = get_db(mongodb_url)
    date_format = "%m/%d/%Y"
    new_date = datetime.strftime(datetime.now(), date_format)
    get_one_day = db.locations.find_one({'date': new_date})
    cities_with_countries = get_one_day['cities']
    cities = [i.split(",")[0] for i in cities_with_countries]
    countries = [i.split(",")[1] for i in cities_with_countries]
    countries_with_fahrenheit = ['United States', 'Belize', 'Palau', 'Bahamas', 'Cayman Islands']
    selected_metric = 'celsius'
    for city_with_country, country, city in zip(cities_with_countries, countries, cities):
        local, language = translate_locale(country)
        if local:
            try:
                translator = gettext.translation('iso3166', pycountry.LOCALES_DIR,
                                                 languages=[language])
                translator.install()
                if city == _(city):
                    translated_city_with_country = local.translate(city + ',' + country, src='en', dest=language).text
                else:
                    translated_city_with_country = _(city) + ',' + _(country)
            except FileNotFoundError:
                translated_city_with_country = local.translate(city + ',' + country, src='en', dest=language).text
            print(translated_city_with_country)
            config_dict = owm.configuration
            config_dict['language'] = language  # your language here
            mgr = owm.weather_manager()
            weather = mgr.weather_at_place(city_with_country).weather
            if country in countries_with_fahrenheit:
                selected_metric = "fahrenheit"
            else:
                selected_metric = "celsius"
            weather = weather.temperature(unit=selected_metric)['temp']
            print(weather)
