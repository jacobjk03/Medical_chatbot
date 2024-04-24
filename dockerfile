FROM python:3.10-slim-buster

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 8080

WORKDIR /app

COPY . /app

RUN pip3 install -r requirements.txt

CMD ["python", "app.py"]
