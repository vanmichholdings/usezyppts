import os
import multiprocessing

# Server socket - Fly.io uses PORT environment variable
bind = f"0.0.0.0:{os.environ.get('PORT', '8080')}"
backlog = 2048

# Worker processes - Optimized for Fly.io
workers = min(multiprocessing.cpu_count() * 2 + 1, 4)  # Optimized for Fly.io
worker_class = 'sync'
worker_connections = 1000
timeout = 120
keepalive = 2

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 50

# Preload app for better memory usage
preload_app = True

# Logging
accesslog = '-'
errorlog = '-'
loglevel = 'info'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'zyppts-logo-processor'

# Server mechanics
daemon = False
pidfile = None
user = None
group = None
tmp_upload_dir = None

# SSL (handled by Fly.io)
keyfile = None
certfile = None

# Memory and performance tuning - Fly.io optimized
worker_tmp_dir = '/tmp'  # Use /tmp for Fly.io compatibility
max_requests_per_child = 1000

# Graceful timeout
graceful_timeout = 30

# Enable stats for monitoring
enable_stdio_inheritance = True
