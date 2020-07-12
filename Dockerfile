ARG CODE_VERSION="3.7-slim"
# Use slim version for smaller size of docker image
FROM python:${CODE_VERSION}
LABEL mantainer="Dmitriy Kisil <email: logart1995@gmail.com>"
ADD . /weather_app
WORKDIR /weather_app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# packages that we need
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . .
CMD ["python3", "clock.py"]