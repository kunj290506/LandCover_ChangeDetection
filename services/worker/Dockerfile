FROM python:3.10-slim

WORKDIR /app

COPY services/common/requirements.txt /app/common-requirements.txt
COPY services/worker/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/common-requirements.txt -r /app/requirements.txt

COPY services/common /app/services/common
COPY services/worker /app/services/worker

ENV PYTHONPATH=/app/services/common:/app/services/worker

CMD ["python", "app/worker.py"]
