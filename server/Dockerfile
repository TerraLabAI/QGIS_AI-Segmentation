FROM python:3.11-slim

WORKDIR /app

# System deps for torch
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download SAM2 checkpoint at build time (~323MB)
RUN mkdir -p /app/checkpoints && \
    curl -L -o /app/checkpoints/sam2.1_hiera_base_plus.pt \
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"

# App code
COPY main.py .

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
