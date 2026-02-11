FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY services/common/requirements.txt /app/common-requirements.txt
COPY services/training/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/common-requirements.txt -r /app/requirements.txt

COPY services/common /app/services/common
COPY services/training /app/services/training
COPY src /app/src

ENV PYTHONPATH=/app/services/common:/app/services/training:/app/src

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8002"]
