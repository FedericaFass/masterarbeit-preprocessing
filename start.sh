#!/bin/sh
exec gunicorn ppm_preprocessing.webapp.app:app \
    --bind 0.0.0.0:${PORT:-8080} \
    --timeout 600 \
    --workers 1 \
    --threads 4 \
    --worker-class gthread
