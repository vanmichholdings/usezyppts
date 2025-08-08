# Optimized Dockerfile for Zyppts v10 - Fly.io Deployment
FROM python:3.11-slim

# Set environment variables for Fly.io optimization
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PLATFORM=fly \
    PYTHONPATH=/app

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
    mkdir -p /app/data/{uploads,outputs,cache,temp,db} && \
    mkdir -p /app/Backend/logs && \
    mkdir -p /app/Backend/logs/sessions && \
    mkdir -p /app/Frontend/templates && \
    mkdir -p /app/Frontend/static && \
    chown -R app:app /app

# Copy application code
COPY . .

# Make startup script executable
RUN chmod +x start_app.py

# Change ownership and ensure database directory permissions
RUN chown -R app:app /app && \
    chmod -R 777 /app/data && \
    chmod -R 755 /app/data/db

# Switch to non-root user
USER app

# Expose port
EXPOSE 8080

# Health check optimized for Fly.io
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Start command - use startup script for proper initialization
CMD ["python", "start_app.py"]
