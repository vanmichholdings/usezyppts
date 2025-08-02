"""
Render-specific configuration for production deployment
"""

import os
import logging
from datetime import timedelta

class RenderConfig:
    """Production configuration optimized for Render deployment"""
    
    # Database configuration for Render
    DATABASE_URL = os.environ.get('DATABASE_URL')
    if DATABASE_URL and DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    
    # Redis configuration for Render
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
    
    # Admin tracking configuration
    ADMIN_TRACKING_ENABLED = True
    ADMIN_REAL_TIME_UPDATES = True
    ADMIN_DATA_RETENTION_DAYS = 90  # Keep admin data for 90 days
    
    # Performance optimization for Render
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_recycle': 300,
        'pool_timeout': 20,
        'max_overflow': 20
    }
    
    # Session configuration for production
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    # Admin session timeout (shorter for production)
    ADMIN_SESSION_TIMEOUT = 1800  # 30 minutes
    
    # Rate limiting for production
    ADMIN_RATE_LIMIT = 200  # requests per hour
    ADMIN_MAX_LOGIN_ATTEMPTS = 3
    ADMIN_LOCKOUT_DURATION = 1800  # 30 minutes
    
    # Logging configuration for Render
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    
    # Admin tracking intervals
    ADMIN_METRICS_UPDATE_INTERVAL = 300  # 5 minutes
    ADMIN_DATA_CLEANUP_INTERVAL = 86400  # 24 hours
    
    # Real-time tracking settings
    ENABLE_REAL_TIME_ANALYTICS = True
    ENABLE_LIVE_USER_TRACKING = True
    ENABLE_SUBSCRIPTION_MONITORING = True
    ENABLE_SYSTEM_HEALTH_MONITORING = True
    
    # Data export settings
    ADMIN_EXPORT_ENABLED = True
    ADMIN_EXPORT_FORMATS = ['csv', 'json']
    ADMIN_EXPORT_MAX_RECORDS = 10000
    
    # Security settings for production
    ADMIN_IP_WHITELIST = os.environ.get('ADMIN_IP_WHITELIST', '').split(',')
    ADMIN_ALLOWED_EMAILS = os.environ.get('ADMIN_ALLOWED_EMAILS', '').split(',')
    ADMIN_SECRET_KEY = os.environ.get('ADMIN_SECRET_KEY', 'change-this-in-production')
    
    # Monitoring and alerting
    ENABLE_ADMIN_ALERTS = True
    ADMIN_ALERT_EMAIL = os.environ.get('ADMIN_ALERT_EMAIL')
    ENABLE_PERFORMANCE_MONITORING = True
    
    @classmethod
    def get_database_config(cls):
        """Get database configuration for Render"""
        return {
            'SQLALCHEMY_DATABASE_URI': cls.DATABASE_URL,
            'SQLALCHEMY_TRACK_MODIFICATIONS': False,
            'SQLALCHEMY_ENGINE_OPTIONS': cls.SQLALCHEMY_ENGINE_OPTIONS
        }
    
    @classmethod
    def get_redis_config(cls):
        """Get Redis configuration for Render"""
        return {
            'REDIS_URL': cls.REDIS_URL,
            'SESSION_TYPE': 'redis',
            'CACHE_TYPE': 'redis',
            'CACHE_REDIS_URL': cls.REDIS_URL
        }
    
    @classmethod
    def get_admin_config(cls):
        """Get admin configuration for Render"""
        return {
            'ADMIN_TRACKING_ENABLED': cls.ADMIN_TRACKING_ENABLED,
            'ADMIN_REAL_TIME_UPDATES': cls.ADMIN_REAL_TIME_UPDATES,
            'ADMIN_DATA_RETENTION_DAYS': cls.ADMIN_DATA_RETENTION_DAYS,
            'ADMIN_SESSION_TIMEOUT': cls.ADMIN_SESSION_TIMEOUT,
            'ADMIN_RATE_LIMIT': cls.ADMIN_RATE_LIMIT,
            'ADMIN_MAX_LOGIN_ATTEMPTS': cls.ADMIN_MAX_LOGIN_ATTEMPTS,
            'ADMIN_LOCKOUT_DURATION': cls.ADMIN_LOCKOUT_DURATION,
            'ADMIN_IP_WHITELIST': cls.ADMIN_IP_WHITELIST,
            'ADMIN_ALLOWED_EMAILS': cls.ADMIN_ALLOWED_EMAILS,
            'ADMIN_SECRET_KEY': cls.ADMIN_SECRET_KEY
        } 