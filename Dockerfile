FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

EXPOSE 8080

CMD ["/bin/sh", "-c", "gunicorn ppm_preprocessing.webapp.app:app --bind 0.0.0.0:${PORT:-8080} --timeout 600 --workers 1 --threads 4 --worker-class gthread"]
