FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends gcc libpq-dev curl \
    && pip install --no-cache-dir mlflow==2.5.0 \
    && apt-get remove -y gcc curl \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /mlflow
VOLUME /mlflow

EXPOSE 5000

CMD ["mlflow", "server", "--backend-store-uri", "/mlflow/mlflow.db", "--default-artifact-root", "/mlflow/artifacts", "--host", "0.0.0.0", "--port", "5000"]
