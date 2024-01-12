FROM python:3.11.4

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD gunicorn --bind 0.0.0.0:8050 app:server
