# API service image (agents + FastAPI). Runs in MOCK backend mode by
# default -- point it at a real vLLM+LMCache server via env vars for
# production (see docker-compose.yml and README.md).
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["python", "scripts/run_server.py"]
