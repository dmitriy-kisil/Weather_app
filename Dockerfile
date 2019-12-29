FROM python:3.7
ADD . /weather_app
WORKDIR /weather_app
COPY requirements.txt /
RUN pip3 install -r /requirements.txt
CMD ["python3", "weather-flask.py"]