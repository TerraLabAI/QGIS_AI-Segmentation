FROM ghcr.io/lilien86/sam3-api-base:v1

WORKDIR /app

RUN pip install --no-cache-dir httpx>=0.27.0

COPY main.py .
COPY supabase_auth.py .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
