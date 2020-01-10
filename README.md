Weather API

Web project with ML.

How to use:
send POST request to https://prediction-weather-app.herokuapp.com/predict/.
You will receive JSON with information such as:
1) your IP, 
2) city/country(based upon IP), 
3) weather data for today in city/country,
4) predicted weather data for 1, 7, 10 day(s) for city/country.
Note: if your city not in BD, you won't receive predictions. Come tomorrow to get it

Used in project technologies: Python3, Flask, MongoDB, Docker.
Code hosted on Heroku and DB on MLab.