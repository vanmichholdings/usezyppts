import os
import multiprocessing

# Server socket - Railway uses PORT environment variable
bind = f"0.0.0.0:{os.environ.get('PORT', '10000')}"
backlog = 2048

# Worker processes - Railway has different resource constraints
workers = min(multiprocessing.cpu_count() * 2 + 1, 4)  # Reduced for Railway
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

# SSL (handled by Railway)
keyfile = None
certfile = None

# Memory and performance tuning - Railway optimized
worker_tmp_dir = '/tmp'  # Use /tmp for Railway compatibility
max_requests_per_child = 1000

# Graceful timeout
graceful_timeout = 30

# Enable stats for monitoring
enable_stdio_inheritance = True 