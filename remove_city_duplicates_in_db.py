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


if __name__ == "__main__":

    db = get_db(mongodb_url)
    date_format = "%m/%d/%Y"
    new_date = datetime.strftime(datetime.now(), date_format)
    day = db.locations.find_one({'date': new_date})
    all_dates = list(db.locations.find({}, {'date': 1, '_id': 0}))
    all_dates = [i['date'] for i in all_dates]
    # print(all_dates)
    for r in all_dates[-2:]:
        day = db.locations.find_one({'date': r})
        cities, temps, ips = day['cities'], day['temperatures'], day['ip_addresses']
        offsets = day['offsets']
        number_of_records = len(cities)
        print(number_of_records)
        print(temps)
        print(len(temps))
        print(ips)
        print(len(ips))
        only_unique_cities, only_unique_ips, only_unique_temps = [], [], []
        count = 0
        need_to_update = False
        for i in range(0, number_of_records):
            selected_city = cities[i]
            current_ips_for_city, current_temps_for_city = ips[i-count], temps[i-count]
            if selected_city not in only_unique_cities:
                only_unique_cities.append(selected_city)
                only_unique_ips.append((selected_city, current_ips_for_city))
                only_unique_temps.append((selected_city, current_temps_for_city))
            else:
                need_to_update = True
                print(i)
                prev_ips_for_city = [k for k in only_unique_ips if k[0] == selected_city][0][1]
                if prev_ips_for_city != current_ips_for_city:
                    print(f'Append those IPs to which we already have')
                    for d in current_ips_for_city:
                        prev_ips_for_city.append(d)
                print(f'Remove {current_temps_for_city} and {current_ips_for_city} by {selected_city}')
                del temps[i-count]
                del ips[i-count]
                del offsets[selected_city]
                count += 1
        print(only_unique_cities)
        print(temps)
        print(len(temps))
        print(ips)
        print(len(ips))
        if need_to_update:
            db.locations.find_one_and_update({"date": r},
                                             {"$set": {"cities": only_unique_cities,
                                                       "temperatures": temps,
                                                       "ip_addresses": ips,
                                                       "offsets": offsets,
                                                       "number_of_cities": len(only_unique_cities)
                                                       }})
            print(f'Date {r} has been updated')
        else:
            print(f'Date {r} has not been updated')
