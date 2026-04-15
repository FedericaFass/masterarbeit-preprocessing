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

COPY start.sh /start.sh
RUN chmod +x /start.sh

CMD ["/start.sh"]
