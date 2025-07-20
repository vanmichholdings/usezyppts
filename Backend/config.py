"""
Configuration settings for the logo processing application.
"""

import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False') == 'True'
    
    # Security settings
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'False') == 'True'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(days=31)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 5,
        'pool_recycle': 3600,
        'pool_timeout': 30
    }
    
    # Email configuration
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME', 'zyppts@gmail.com')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD', 'stnvdfagbjeazski')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER', 'team@zyppts.com')
    
    # File upload configuration
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
    FRONTEND_DIR = os.path.join(BASE_DIR, 'Frontend')
    UPLOAD_FOLDER = os.path.join(BACKEND_DIR, 'uploads')
    OUTPUT_FOLDER = os.path.join(BACKEND_DIR, 'outputs')
    CACHE_FOLDER = os.path.join(BACKEND_DIR, 'cache')
    TEMP_FOLDER = os.path.join(BACKEND_DIR, 'temp')
    TEMPLATES_FOLDER = os.path.join(FRONTEND_DIR, 'templates')
    STATIC_FOLDER = os.path.join(FRONTEND_DIR, 'static')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    
    # Brand Colors
    BRAND_COLORS = {
        'primary': {
            'hex': '#1B1464',  # Deep Blue
            'rgb': (27, 20, 100)
        },
        'secondary': {
            'hex': '#FFB81C',  # Gold
            'rgb': (255, 184, 28)
        }
    }
    
    # Rate limiting with Redis
    RATELIMIT_DEFAULT = "200 per day"
    RATELIMIT_STORAGE_URL = "redis://localhost:6379/0"
    RATELIMIT_STRATEGY = "fixed-window"
    RATELIMIT_ENABLED = True
    
    # Session configuration with Redis
    SESSION_TYPE = 'redis'
    SESSION_REDIS = {
        'host': 'localhost',
        'port': 6379,
        'db': 1,
        'prefix': 'session:',
        'socket_timeout': 5,
        'socket_connect_timeout': 5,
        'retry_on_timeout': True
    }
    
    # Cache Configuration with Redis
    CACHE_TYPE = 'redis'
    CACHE_REDIS_HOST = 'localhost'
    CACHE_REDIS_PORT = 6379
    CACHE_REDIS_DB = 0
    CACHE_DEFAULT_TIMEOUT = 300
    CACHE_KEY_PREFIX = 'zyppts_'
    CACHE_OPTIONS = {
        'socket_timeout': 5,
        'socket_connect_timeout': 5,
        'retry_on_timeout': True
    }
    
    # Image Processing Optimization
    IMAGE_PROCESSING_CONFIG = {
        'use_gpu': True,
        'max_image_dimension': 8192,  # Increased from 4096
        'jpeg_quality': 95,  # Increased from 85
        'png_compression': 6,
        'webp_quality': 95,  # Increased from 80
        'svg_simplify_tolerance': 0.1,
        'enable_multithreading': True,
        'thread_count': 4,
        'resize_method': 'lanczos',  # Added explicit resize method
        'cache_size': 1000,  # Added cache size
        'cache_timeout': 3600  # Added cache timeout
    }

    # Advanced Performance Optimization Settings
    PERFORMANCE_CONFIG = {
        # Parallel Processing
        'enable_multiprocessing': True,
        'enable_priority_queue': True,
        'enable_adaptive_workers': True,
        'max_workers': 16,  # Increased for better parallelization
        'process_workers': 8,  # For CPU-intensive tasks
        'batch_size': 20,  # Increased batch size
        'task_timeout': 120,  # Increased timeout
        
        # Memory Management
        'memory_threshold': 0.75,  # 75% memory usage threshold
        'predictive_cleanup': True,
        'aggressive_cleanup': True,
        'cleanup_interval': 180,  # 3 minutes
        
        # Caching Strategy
        'enable_advanced_caching': True,
        'cache_size': 2000,  # Increased cache size
        'cache_timeout': 3600,  # 1 hour
        'lru_cache_size': 1000,
        
        # Vector Processing
        'vector_trace_optimization': {
            'max_dimension': 2000,
            'max_colors': 3,
            'kmeans_iterations': 100,
            'contour_epsilon': 0.01,
            'min_area_threshold': 50,
            'parallel_workers': 4,
            'enable_caching': True,
            'timeout_seconds': 30
        },
        
        # Social Media Processing
        'social_media_optimization': {
            'parallel_platforms': True,
            'max_concurrent_platforms': 8,
            'preview_generation': True,
            'batch_processing': True
        }
    }

    # Performance Optimization Settings
    CACHE_ENABLED = True
    CACHE_TIMEOUT = 3600  # 1 hour
    MAX_CONCURRENT_PROCESSES = 8  # Increased from 4
    BATCH_SIZE = 20  # Increased from 10
    COMPRESSION_QUALITY = 95  # Increased from 85
    ENABLE_PROGRESSIVE_LOADING = True
    CHUNK_SIZE = 8192  # Added chunk size
    BUFFER_SIZE = 65536  # Added buffer size
    
    # Memory Management
    MEMORY_LIMIT = '6GB'  # Increased from 2GB
    CLEANUP_INTERVAL = 1800  # 30 minutes (reduced from 1 hour)
    TEMP_FILE_TTL = 1800  # 30 minutes (reduced from 1 hour)
    
    # Advanced Caching
    ADVANCED_CACHE_CONFIG = {
        'enable_redis': True,
        'enable_file_cache': True,
        'enable_memory_cache': True,
        'cache_layers': ['memory', 'file', 'redis'],
        'cache_strategy': 'lru',
        'cache_compression': True,
        'cache_encryption': False
    }
    
    # Task Scheduling
    TASK_SCHEDULER_CONFIG = {
        'enable_priority_queue': True,
        'enable_task_preemption': True,
        'max_queue_size': 1000,
        'task_timeout': 300,  # 5 minutes
        'retry_attempts': 3,
        'retry_delay': 1
    }
    
    # Resource Monitoring
    RESOURCE_MONITORING = {
        'enable_monitoring': True,
        'monitoring_interval': 30,  # 30 seconds
        'alert_thresholds': {
            'cpu_usage': 80,
            'memory_usage': 85,
            'disk_usage': 90
        },
        'auto_optimization': True
    }
    
    # Subscription Plans
    SUBSCRIPTION_PLANS = {
        'free': {
            'price': 0,
            'features': [
                '5 Projects',
                '2 Team Members',
                'Basic Analytics',
                'Community Support'
            ]
        },
        'pro': {
            'price': 29,
            'features': [
                'Unlimited Projects',
                '5 Team Members',
                'Advanced Analytics',
                'Priority Support',
                'Custom Branding'
            ]
        },
        'studio': {
            'price': 79,
            'features': [
                'Unlimited Projects',
                '15 Team Members',
                'Premium Analytics',
                '24/7 Priority Support',
                'Custom Branding',
                'API Access'
            ]
        },
        'enterprise': {
            'price': 199,
            'features': [
                'Unlimited Everything',
                'Unlimited Team Members',
                'Enterprise Analytics',
                'Dedicated Support',
                'Custom Branding',
                'API Access',
                'SLA Guarantee',
                'Custom Integrations'
            ]
        }
    }
    
    # Beta test configuration
    BETA_TEST_CREDITS = 500
    BETA_FEATURES = ['basic_formats', 'social_media', 'advanced_effects', 'color_variations', 'api_access']
    
    # Authentication settings
    LOGIN_REQUIRED_FOR_GENERATION = True
    PREVIEW_ALLOWED_WITHOUT_LOGIN = True
    
    # Matrix Operation Settings
    MATRIX_OPERATIONS = {
        'cholesky': {
            'discard_threshold': 1e-6,  # Decreased from 1e-4
            'shift': 1e-1,  # Increased from 1e-4
            'max_iterations': 100,
            'tolerance': 1e-10
        },
        'numerical_stability': {
            'min_eigenvalue': 1e-12,
            'condition_number_threshold': 1e8,
            'regularization_factor': 1e-6
        }
    }
    
    # Redis Configuration
    REDIS_HOST = 'localhost'
    REDIS_PORT = 6379
    REDIS_DB = 0
    REDIS_PASSWORD = None
    REDIS_SOCKET_TIMEOUT = 120.0
    REDIS_SOCKET_CONNECT_TIMEOUT = 10.0
    REDIS_RETRY_ON_TIMEOUT = True
    REDIS_SOCKET_KEEPALIVE = True
    REDIS_MAX_CONNECTIONS = 1000

    # Celery Configuration
    CELERY_BROKER_URL = f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'
    CELERY_RESULT_BACKEND = f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'
    CELERY_REDIS_MAX_CONNECTIONS = REDIS_MAX_CONNECTIONS
    CELERY_REDIS_SOCKET_TIMEOUT = REDIS_SOCKET_TIMEOUT
    CELERY_REDIS_SOCKET_CONNECT_TIMEOUT = REDIS_SOCKET_CONNECT_TIMEOUT
    CELERY_REDIS_RETRY_ON_TIMEOUT = REDIS_RETRY_ON_TIMEOUT
    CELERY_REDIS_SOCKET_KEEPALIVE = REDIS_SOCKET_KEEPALIVE
    
    @classmethod
    def init_app(cls, app):
        """Initialize application directories"""
        # Create required directories with proper permissions
        for directory in [cls.UPLOAD_FOLDER, cls.OUTPUT_FOLDER, 
                         cls.CACHE_FOLDER, cls.TEMP_FOLDER]:
            os.makedirs(directory, exist_ok=True)
            os.chmod(directory, 0o750)  # rwxr-x---
            
            # Set ownership if possible
            try:
                import pwd
                uid = pwd.getpwnam(os.environ.get('USER', '')).pw_uid
                os.chown(directory, uid, -1)
            except (ImportError, KeyError):
                pass 