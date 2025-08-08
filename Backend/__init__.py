"""
Flask application factory and extension initialization
"""

import os
import logging
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_mail import Mail
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Try to import Flask-Caching, but make it optional
try:
    from flask_caching import Cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    Cache = None

from flask_session import Session
import urllib.parse
import redis

def parse_redis_url(redis_url):
    """Parse Redis URL to extract connection details"""
    try:
        parsed = urllib.parse.urlparse(redis_url)
        return {
            'host': parsed.hostname or 'localhost',
            'port': parsed.port or 6379,
            'password': parsed.password,
            'db': int(parsed.path.lstrip('/')) if parsed.path else 0
        }
    except Exception:
        return {
            'host': 'localhost',
            'port': 6379,
            'password': None,
            'db': 0
        }

def test_redis_connection():
    """Test Redis connection and return status"""
    try:
        redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
        r = redis.from_url(redis_url)
        r.ping()
        return True, "Connected"
    except ImportError:
        return False, "Redis library not available"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

# Initialize Flask extensions
db = SQLAlchemy()
login_manager = LoginManager()
mail = Mail()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["1000 per day", "100 per hour"],  # Increased limits
    storage_uri=os.environ.get('REDIS_URL', 'redis://localhost:6379/0'),  # Explicitly use db 0
    storage_options={"db": 0}  # Ensure db 0 is used
)
if CACHE_AVAILABLE:
    cache = Cache()
else:
    cache = None
session = Session()

# Import the main create_app function from app_config
from .app_config import create_app
