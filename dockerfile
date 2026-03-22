FROM python:3.10-slim-buster

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install CPU-only torch first (avoids pulling the 2.5GB GPU build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

EXPOSE 7860

# Use gunicorn for production (1 worker to keep RAM low, 180s timeout for ReAct agent)
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:7860", "--timeout", "180", "--keep-alive", "5", "app:app"]
