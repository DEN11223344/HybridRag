FROM python:3.11-slim

# Prevent Python from writing pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV TOKENIZERS_PARALLELISM=false

# Install OS dependencies: curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency requirements
COPY requirements.txt .

# Install CPU PyTorch (pinned to match requirements.txt) using IPv4 curl resume
RUN pip install --no-cache-dir --timeout 120 --retries 15 --upgrade pip && \
  curl -4 -fL --retry 30 --retry-delay 2 --retry-all-errors -C - \
    -o /tmp/torch-2.2.2+cpu-cp311-cp311-linux_x86_64.whl \
    "https://download.pytorch.org/whl/cpu/torch-2.2.2%2Bcpu-cp311-cp311-linux_x86_64.whl" && \
  pip install --no-cache-dir --timeout 120 --retries 15 --resume-retries 50 /tmp/torch-2.2.2+cpu-cp311-cp311-linux_x86_64.whl && \
  rm -f /tmp/torch-2.2.2+cpu-cp311-cp311-linux_x86_64.whl

# Install Python packages
RUN pip install --no-cache-dir --timeout 120 --retries 15 --resume-retries 50 -r requirements.txt

# Copy application source code
COPY . .

# Ensure storage directories exist
RUN mkdir -p data/uploads data/extracted data/processed data/qdrant_storage

EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
