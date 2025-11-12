# Multi-stage build for optimized image
FROM python:3.12-slim AS builder

# Set working directory
WORKDIR /build

# Copy project files
COPY pyproject.toml ./

# Create virtual environment and install dependencies with moderate cleanup
RUN python -m venv /opt/venv && \
    . /opt/venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e . && \
    # Cleanup unnecessary files (but keep module dependencies)
    find /opt/venv -type f -name '*.pyc' -delete && \
    find /opt/venv -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type d -name 'tests' -exec rm -rf {} + 2>/dev/null || true && \
    find /opt/venv -type f -name '*.md' -delete && \
    find /opt/venv -type d -name 'examples' -not -path '*/google/adk/examples*' -exec rm -rf {} + 2>/dev/null || true

# Final stage
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH"

# Install minimal runtime dependencies (no curl, use Python for health checks)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code (excluding unnecessary files via .dockerignore)
COPY . .

# Expose port
EXPOSE 10001

# Health check using Python instead of curl
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:10001/.well-known/agent-card.json').read()" || exit 1

# Run the application
CMD ["python", "__main__.py"]