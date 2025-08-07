# Optimized Dockerfile for Zyppts v10 - Fly.io Deployment
FROM python:3.11-slim

# Set environment variables for Fly.io optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PLATFORM=fly

# Install system dependencies in one layer (optimized for Fly.io)
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    --no-install-suggests \
    build-essential \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    # Image processing dependencies
    libjpeg62-turbo \
    libpng16-16 \
    libtiff-dev \
    # PDF processing dependencies
    poppler-utils \
    # SVG processing dependencies
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    # SSL and networking
    libffi-dev \
    libssl3 \
    # XML processing
    libxml2 \
    libxslt1.1 \
    # Utilities
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with Fly.io optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create non-root user and storage directories
RUN useradd --create-home --shell /bin/bash app && \
    mkdir -p /app/data/{uploads,outputs,cache,temp} && \
    mkdir -p /app/Backend/logs && \
    chown -R app:app /app

# Copy application code
COPY . .

# Change ownership
RUN chown -R app:app /app

# Switch to non-root user
USER app

# Expose port
EXPOSE 8080

# Health check optimized for Fly.io
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start command optimized for Fly.io (4 workers for 2GB RAM)
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "--worker-class", "sync", "--timeout", "120", "--keep-alive", "5", "--max-requests", "1000", "--max-requests-jitter", "100", "--worker-connections", "1000", "Backend:create_app()"]
