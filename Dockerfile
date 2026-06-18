# syntax=docker/dockerfile:1
FROM python:3.11-slim

# System libraries required by OpenCV (cv2) at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (far smaller than the default CUDA build), then the rest.
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Application code, templates, trained weights, and the UI background image.
COPY pneumonia_api.py .
COPY templates/ ./templates/
COPY model.pth .
COPY static/background.jpg ./static/background.jpg

EXPOSE 5000

# One worker: the model lives in memory; additional workers each load their own copy.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "pneumonia_api:app"]
