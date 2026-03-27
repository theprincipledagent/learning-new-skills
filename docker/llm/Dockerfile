FROM python:3.12-slim

RUN pip install --no-cache-dir anthropic>=0.39.0

COPY entrypoint.py /app/entrypoint.py

WORKDIR /work

ENTRYPOINT ["python", "/app/entrypoint.py"]
