FROM python:3.10-slim

WORKDIR /app

COPY services/common/requirements.txt /app/common-requirements.txt
COPY services/inference/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r /app/common-requirements.txt -r /app/requirements.txt

COPY services/common /app/services/common
COPY services/inference /app/services/inference
COPY src /app/src

ENV PYTHONPATH=/app/services/common:/app/services/inference:/app/src

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
