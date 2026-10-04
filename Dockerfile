# Single-service image: builds the React app, then serves it and the API from one FastAPI process.
FROM node:20-slim AS web
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY backend/requirements-serve.txt backend/requirements-serve.txt
RUN pip install --no-cache-dir -r backend/requirements-serve.txt
COPY backend/app backend/app
COPY backend/artifacts backend/artifacts
COPY backend/data/sevir backend/data/sevir
COPY --from=web /web/dist frontend/dist
WORKDIR /app/backend
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
