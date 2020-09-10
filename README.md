Weather API

Web project with ML.

How to use:
send POST request to https://weather-app-288315.uc.r.appspot.com/predict/.
You will receive JSON with information such as:
1) your IP, 
2) city/country(based upon IP), 
3) weather data for today in city/country,
4) predicted weather data for 1, 7, 10 day(s) for city/country.

Show temperature at your location as well as your city, country, and which day is today.

As an experimental option, you could see predictions for weather temperatures for the next 1, 7, 10 days.

If your city not in DB, you will see all predictions are equal to temperature, which you got by the first visit. Each day for each available city in the DB script will update the parameter of temperature. And for each city script will create a machine learning model to predict temperature for the next 1, 7, 10 days.

From 1 to 10 observation LinearRegression model will be used for predictions in your city. After 10 days, a more complex RNN model will be used for predictions in your city.

Code for backend part host on Heroku free dyno, so if you got 'connection was closed because 'full header was received' - just close and open app again. Server starting up time around 30sec - 1min from sleep. After 30 min of inactivity, server will go to sleep again

Used in project technologies: Python3, Flask, MongoDB, Docker.
Code hosted previously on Heroku and then moved to Google AppEngine and DB on MLab.

It's backend part. For frontend, please, go [here](https://github.com/Oysiyl/simple-weather-app)
