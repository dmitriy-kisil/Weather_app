Weather API

Web project with ML.

How to use:
send POST request to https://weather-app-288315.uc.r.appspot.com/predict/.
You will receive JSON with information such as:
1) your IP, 
2) city/country(based upon IP), 
3) weather data for today in city/country(temperature is updated every hour),
4) predictions for the next 24 hours temperatures (the model is launched once a day and gives the result exactly one day ahead),
5) predicted weather data for 1, 7, 10 day(s) for city/country (predict temperature at 12:00 AM in your city).

Show temperature at your location as well as your city, country, and which day is today.

As an experimental option, you could see predictions for weather temperatures for the next 1, 7, 10 days as well as predictions for hour temperature for present day.

If your city not in DB, you will see all predictions are equal to temperature, which you got by the first visit. Each day for each available city in the DB script will update the parameter of temperature. And for each city script will create a machine learning model to predict temperature for the next 1, 7, 10 days.

From 1 to 10 observation LinearRegression model will be used for predictions in your city. After 10 days, a more complex RNN model will be used for predictions in your city.

Also added another ML model, which make prediction for temperatures for each of 24 hours for one day (chose LinearRegression) and update temperatures by hour each hour.

Used in project technologies: Python3, Flask, MongoDB, Docker.
Code hosted previously on Heroku and then moved to Google AppEngine and DB on MLab.
After sending the request, you should receive the result within one minute maximum.

It's backend part. For frontend, please, go [here](https://github.com/Oysiyl/simple-weather-app)
