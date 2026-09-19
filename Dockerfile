# STAGE 1: BUILDER - Full python image with build tools used to install dependencies
FROM python:3.11-slim AS builder

# System dependencies for building python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv in the builder
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create and activate vitural env
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy dependency spefication first (Docker layer caching):
# If pyproject.toml doesn't change, this layer is cached — faster rebuilds
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src/ src/

# Install the package and its dependencies
RUN uv pip install --no-cache .


## STAGE 2: PRODUCTION - minimal image no build rools, Only what's needed to run the API
FROM python:3.11-slim AS production

# Secuirty: don't run as root
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --no-create-home appuser

# Copy the vitural env from builder stage
COPY --from=builder /opt/venv /opt/venv

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV LOG_FORMAT=json
ENV LOG_LEVEL=INFO

WORKDIR /app

# Copy app code 
COPY src/ src/
COPY api/ api/
COPY models/ models/

# Give ownership to non-root user
RUN chown -R appuser:appuser /app

USER appuser

# Expose the API port
EXPOSE 8000

# Health check - if fails 3 times, container is marked unhealthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Start the API server
CMD ["uvicorn","api.main:app", \
     "--host","0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "warning"]