FROM python:3.11-slim

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full project
COPY . .

# Install the package in editable mode so imports resolve
RUN pip install --no-cache-dir -e .

# Create non-root user (HF Spaces security requirement)
RUN useradd -m -u 1000 appuser
USER appuser

# HF Spaces requires port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Environment defaults (can be overridden at runtime)
ENV TASK_ID=task_easy
ENV PORT=7860

# Entrypoint — single worker (concurrent sessions disabled by design)
CMD ["uvicorn", "spacecraft_anomaly_env.server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--log-level", "info"]
