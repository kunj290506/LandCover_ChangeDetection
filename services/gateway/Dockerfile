FROM python:3.10-slim

WORKDIR /app

COPY services/common/requirements.txt /app/common-requirements.txt
COPY services/gateway/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/common-requirements.txt -r /app/requirements.txt

COPY services/common /app/services/common
COPY services/gateway /app/services/gateway

ENV PYTHONPATH=/app/services/common:/app/services/gateway

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
