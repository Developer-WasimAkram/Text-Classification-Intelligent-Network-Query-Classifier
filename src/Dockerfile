FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    KMP_DUPLICATE_LIB_OK=TRUE \
    OMP_NUM_THREADS=4

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/*.py ./src/
COPY src/train_data_balanced.csv ./src/

# Create data directory for outputs
RUN mkdir -p /app/data

# Mount points:
#   /app/data             — bind-mount to persist embeddings, models, plots

# Default: run full pipeline
WORKDIR /app/src
ENTRYPOINT ["python", "-u", "main.py"]

# Override with args, e.g.: docker run <image> --sample 1000
CMD []
