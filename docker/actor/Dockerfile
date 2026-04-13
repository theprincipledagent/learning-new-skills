FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    'smolagents[litellm]' \
    'litellm[google]' \
    exa-py \
    pandas \
    openpyxl \
    PyPDF2 \
    markdownify \
    requests

COPY entrypoint.py /app/entrypoint.py

RUN useradd -m -s /bin/bash agent
WORKDIR /work
RUN chown agent:agent /work
USER agent

ENTRYPOINT ["python", "/app/entrypoint.py"]
