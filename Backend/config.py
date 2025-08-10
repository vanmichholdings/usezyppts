"""
Configuration settings for the logo processing application.
"""

import os
from datetime import timedelta
from dotenv import load_dotenv
import redis # Added missing import for redis

# Load environment variables from .env file in root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_FILE = os.path.join(BASE_DIR, '.env')
load_dotenv(ENV_FILE)

class Config:
    # Flask configuration
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev')
    DEBUG = os.environ.get('FLASK_DEBUG', 'False') == 'True'
    
    @staticmethod
    def get_safe_engine_options(database_uri):
        """
        Get SQLAlchemy engine options that are safe for the specific database type.
        SQLite doesn't support connection pooling, so we exclude pool_size and pool_timeout.
        """
        if not database_uri or database_uri.startswith('sqlite'):
            # SQLite configuration - no connection pooling
            return {
                'pool_pre_ping': True,  # Safe for SQLite
                'echo': False  # Set to True for SQL debugging
            }
        else:
            # PostgreSQL/MySQL configuration - full pooling support
            return {
                'pool_size': 10,
                'pool_recycle': 3600,
                'pool_timeout': 30,
                'pool_pre_ping': True,
                'echo': False  # Set to True for SQL debugging
            }
    
    # Security settings
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'False') == 'True'
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = timedelta(days=31)
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Database configuration for production - SIMPLIFIED APPROACH
    DATABASE_URL = os.environ.get('DATABASE_URL')
    
    # For Fly.io, use PostgreSQL for persistent data storage
    if os.environ.get('PLATFORM') == 'fly':
        # Use PostgreSQL if DATABASE_URL is provided, otherwise fallback to SQLite file
        if DATABASE_URL and DATABASE_URL.startswith('postgresql://'):
            SQLALCHEMY_DATABASE_URI = DATABASE_URL
            SQLALCHEMY_TRACK_MODIFICATIONS = False
            SQLALCHEMY_ENGINE_OPTIONS = {
                'pool_size': 10,
                'pool_recycle': 3600,
                'pool_timeout': 30,
                'pool_pre_ping': True,
                'echo': False
            }
        else:
            # Fallback to file-based SQLite for persistence
            SQLALCHEMY_DATABASE_URI = 'sqlite:////app/data/db/app.db'
            SQLALCHEMY_TRACK_MODIFICATIONS = False
            SQLALCHEMY_ENGINE_OPTIONS = {
                'pool_pre_ping': True,
                'echo': False
            }
        
        # File storage still uses mounted volume
        DATA_PATH = '/app/data'
        UPLOAD_FOLDER = '/app/data/uploads'
        OUTPUT_FOLDER = '/app/data/outputs'
        CACHE_FOLDER = '/app/data/cache'
        TEMP_FOLDER = '/app/data/temp'
        
    else:
        # Local development with file-based SQLite
        DATA_PATH = os.path.join(BASE_DIR, 'data')
        DB_PATH = os.path.join(DATA_PATH, 'db')
        
        # Create directories if needed
        try:
            os.makedirs(DB_PATH, exist_ok=True)
            os.chmod(DB_PATH, 0o777)
        except OSError as e:
            print(f"Warning: Could not create database directory {DB_PATH}: {e}")
        
        DB_FILE_PATH = os.path.join(DB_PATH, "app.db")
        SQLALCHEMY_DATABASE_URI = DATABASE_URL or f'sqlite:///{DB_FILE_PATH}'
        SQLALCHEMY_TRACK_MODIFICATIONS = False
        
        UPLOAD_FOLDER = os.path.join(DATA_PATH, 'uploads')
        OUTPUT_FOLDER = os.path.join(DATA_PATH, 'outputs')
        CACHE_FOLDER = os.path.join(DATA_PATH, 'cache')
        TEMP_FOLDER = os.path.join(DATA_PATH, 'temp')
    
    # SQLAlchemy engine options will be set dynamically in init_app method
    SQLALCHEMY_ENGINE_OPTIONS = {}
    
    # Email configuration
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 587
    MAIL_USE_TLS = True
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER', 'Zyppts HQ <zyppts@gmail.com>')
    
    # Admin email configuration
    ADMIN_ALERT_EMAIL = os.environ.get('ADMIN_ALERT_EMAIL', 'mike@usezyppts.com,zyppts@gmail.com')
    SITE_URL = os.environ.get('SITE_URL', 'https://usezyppts.com')
    
    # File upload configuration for production
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
    FRONTEND_DIR = os.path.join(BASE_DIR, 'Frontend')

    # Use data directory for persistent storage (outside version control)
    # Ensure consistent path handling for Fly.io vs local development
    if os.environ.get('PLATFORM') == 'fly':
        # On Fly.io, always use the mounted volume path for file storage
        UPLOAD_FOLDER = '/app/data/uploads'
        OUTPUT_FOLDER = '/app/data/outputs'
        CACHE_FOLDER = '/app/data/cache'
        TEMP_FOLDER = '/app/data/temp'
    else:
        # Local development
        UPLOAD_FOLDER = os.path.join(DATA_PATH, 'uploads')
        OUTPUT_FOLDER = os.path.join(DATA_PATH, 'outputs')
        CACHE_FOLDER = os.path.join(DATA_PATH, 'cache')
        TEMP_FOLDER = os.path.join(DATA_PATH, 'temp')
        
    TEMPLATES_FOLDER = os.path.join(FRONTEND_DIR, 'templates')
    STATIC_FOLDER = os.path.join(FRONTEND_DIR, 'static')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'svg', 'pdf', 'webp'}
    
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
    
    # Redis Configuration for production
    REDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379')
    
    # Parse Redis URL to get connection details
    import urllib.parse
    try:
        parsed_redis = urllib.parse.urlparse(REDIS_URL)
        REDIS_HOST = parsed_redis.hostname or 'localhost'
        REDIS_PORT = parsed_redis.port or 6379
        REDIS_PASSWORD = parsed_redis.password
        REDIS_DB = int(parsed_redis.path.lstrip('/')) if parsed_redis.path else 0
    except Exception:
        # Fallback to environment variables or defaults
        REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
        REDIS_PORT = int(os.environ.get('REDIS_PORT', 6379))
        REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD')
        REDIS_DB = int(os.environ.get('REDIS_DB', 0))
    
    # Rate limiting with Redis
    RATELIMIT_DEFAULT = "100 per hour"  # Reduced for production
    RATELIMIT_STORAGE_URL = f"{REDIS_URL}/0"
    RATELIMIT_STRATEGY = "fixed-window"
    RATELIMIT_ENABLED = True
    
    # Session configuration with Redis
    SESSION_TYPE = 'redis'
    try:
        SESSION_REDIS = redis.from_url(REDIS_URL, db=0)  # Use db 0 for sessions
    except Exception as e:
        print(f"Warning: Could not configure Redis sessions: {e}")
        SESSION_TYPE = 'filesystem'
        SESSION_FILE_DIR = os.path.join(BASE_DIR, 'data', 'sessions')
        SESSION_FILE_THRESHOLD = 500
        SESSION_FILE_MODE = 0o600
        if not os.path.exists(SESSION_FILE_DIR):
            os.makedirs(SESSION_FILE_DIR, mode=0o700)
    
    # Cache Configuration with Redis
    CACHE_TYPE = 'redis'
    CACHE_REDIS_HOST = REDIS_HOST
    CACHE_REDIS_PORT = REDIS_PORT
    CACHE_REDIS_PASSWORD = REDIS_PASSWORD
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
            'monthly_credits': 3,  # 3 logo credits per month
            'features': [
                '5 Projects',
                '2 Team Members',
                'Basic Analytics',
                'Community Support'
            ]
        },
        'pro': {
            'price': 9.99,
            'monthly_credits': -1,  # Unlimited uploads for basic + effects
            'stripe_price_id': 'price_1RnCQDI1902kkwjouP5vvijE',
            'stripe_annual_price_id': 'price_1Rr9JxI1902kkwjoIOBPETYv',  # Pro Annual
            'features': [
                'Unlimited Projects',
                '5 Team Members',
                'Advanced Analytics',
                'Priority Support',
                'Custom Branding'
            ]
        },
        'studio': {
            'price': 29.99,
            'monthly_credits': -1,  # Unlimited, with social + batch
            'stripe_price_id': 'price_1RnCRWI1902kkwjoq18LY3eB',
            'stripe_annual_price_id': 'price_1Rr9MII1902kkwjomfiuG44B',  # Studio Annual
            'features': [
                'Unlimited Projects',
                '15 Team Members',
                'Premium Analytics',
                '24/7 Priority Support',
                'Custom Branding',
                'API Access',
                'Batch Processing'  # New feature for Studio plan
            ]
        },
        'enterprise': {
            'price': 199,
            'monthly_credits': -1,  # Unlimited logo credits
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
    
    # Stripe Configuration - Use environment variables
    STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY')
    STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY')
    STRIPE_WEBHOOK_SECRET = os.environ.get('STRIPE_WEBHOOK_SECRET')
    
    # Celery Configuration (uses Redis)
    CELERY_BROKER_URL = f'{REDIS_URL}/{REDIS_DB}'
    CELERY_RESULT_BACKEND = f'{REDIS_URL}/{REDIS_DB}'
    CELERY_REDIS_MAX_CONNECTIONS = 1000
    CELERY_REDIS_SOCKET_TIMEOUT = 120.0
    CELERY_REDIS_SOCKET_CONNECT_TIMEOUT = 10.0
    CELERY_REDIS_RETRY_ON_TIMEOUT = True
    CELERY_REDIS_SOCKET_KEEPALIVE = True
    
    # Platform-specific optimizations
    PLATFORM = os.environ.get('PLATFORM', 'fly')
    
    # Fly.io optimizations (2GB RAM limit)
    if PLATFORM == 'fly':
        # Memory optimizations for Fly.io - CRITICAL FOR STABILITY
        MEMORY_LIMIT = '1.5GB'  # Reduced from 2GB to leave headroom
        MAX_CONCURRENT_PROCESSES = 2  # Reduced from 4 to prevent memory issues
        BATCH_SIZE = 5  # Reduced from 10 to limit memory usage
        
        # Performance optimizations for Fly.io - MEMORY CONSCIOUS
        PERFORMANCE_CONFIG = {
            # Parallel Processing - REDUCED FOR MEMORY
            'enable_multiprocessing': True,
            'enable_priority_queue': True,
            'enable_adaptive_workers': True,
            'max_workers': 4,  # REDUCED from 8 - critical for Fly.io 2GB limit
            'process_workers': 2,  # REDUCED from 4 - prevent memory exhaustion
            'batch_size': 5,  # REDUCED from 10 - smaller batches
            'task_timeout': 90,  # REDUCED from 120 - quicker timeouts
            
            # Memory Management - AGGRESSIVE FOR FLY.IO
            'memory_threshold': 0.6,  # REDUCED from 0.7 - more conservative
            'predictive_cleanup': True,
            'aggressive_cleanup': True,
            'cleanup_interval': 120,  # REDUCED from 180 - more frequent cleanup
            
            # Caching Strategy - REDUCED FOR MEMORY
            'enable_advanced_caching': True,
            'cache_size': 500,  # REDUCED from 1000 - half the cache size
            'cache_timeout': 1800,  # REDUCED from 3600 - shorter cache lifetime
            'lru_cache_size': 250,  # REDUCED from 500 - quarter the LRU cache
            
            # Vector Processing - MEMORY OPTIMIZED
            'vector_trace_optimization': {
                'max_dimension': 1500,  # REDUCED from 2000 - smaller max dimension
                'max_colors': 2,  # REDUCED from 3 - fewer colors to process
                'kmeans_iterations': 50,  # REDUCED from 100 - fewer iterations
                'contour_epsilon': 0.02,  # INCREASED from 0.01 - less precision for speed
                'min_area_threshold': 100,  # INCREASED from 50 - filter smaller areas
                'parallel_workers': 1,  # REDUCED from 2 - single worker only
                'enable_caching': True,
                'timeout_seconds': 20  # REDUCED from 30 - quicker timeouts
            },
            
            # Social Media Processing - MEMORY OPTIMIZED
            'social_media_optimization': {
                'parallel_platforms': False,  # DISABLED - process sequentially
                'max_concurrent_platforms': 1,  # REDUCED from 4 - one at a time
                'preview_generation': True,
                'batch_processing': False  # DISABLED - no batch processing
            }
        }
        
        # Image Processing optimizations for Fly.io - MEMORY FOCUSED
        IMAGE_PROCESSING_CONFIG = {
            'use_gpu': False,  # DISABLED - may not be available on Fly.io
            'max_image_dimension': 2048,  # REDUCED from 4096 - half the max size
            'jpeg_quality': 85,  # REDUCED from 95 - smaller files
            'png_compression': 9,  # INCREASED from 6 - better compression
            'webp_quality': 85,  # REDUCED from 95 - smaller files
            'svg_simplify_tolerance': 0.2,  # INCREASED from 0.1 - more simplification
            'enable_multithreading': False,  # DISABLED - save memory
            'thread_count': 1,  # REDUCED from 2 - single thread only
            'resize_method': 'nearest',  # CHANGED from lanczos - faster, less memory
            'cache_size': 100,  # REDUCED from 500 - much smaller cache
            'cache_timeout': 1800  # REDUCED from 3600 - shorter cache
        }
        
        # Cache optimizations for Fly.io - MINIMAL MEMORY USAGE
        CACHE_SIZE = 200  # REDUCED from 1000 - much smaller
        CACHE_TIMEOUT = 1800  # REDUCED from 3600 - shorter lifetime
        LRU_CACHE_SIZE = 100  # REDUCED from 500 - much smaller
        
        # Task scheduling optimizations - MEMORY CONSCIOUS
        TASK_SCHEDULER_CONFIG = {
            'enable_priority_queue': True,
            'enable_task_preemption': True,
            'max_queue_size': 50,  # REDUCED from 500 - much smaller queue
            'task_timeout': 60,  # REDUCED from 300 - quicker timeouts
            'retry_attempts': 2,  # REDUCED from 3 - fewer retries
            'retry_delay': 2  # INCREASED from 1 - longer delays between retries
        }
        
        # Resource monitoring for Fly.io - AGGRESSIVE MONITORING
        RESOURCE_MONITORING = {
            'enable_monitoring': True,
            'monitoring_interval': 15,  # REDUCED from 30 - more frequent monitoring
            'alert_thresholds': {
                'cpu_usage': 70,  # REDUCED from 80 - earlier CPU alerts
                'memory_usage': 70,  # REDUCED from 80 - earlier memory alerts
                'disk_usage': 80  # REDUCED from 85 - earlier disk alerts
            },
            'auto_optimization': True
        }
        
        # Celery optimizations for Fly.io - MEMORY CONSCIOUS
        CELERY_REDIS_MAX_CONNECTIONS = 100  # REDUCED from 500 - much fewer connections
        
    else:
        # Default configuration for other platforms
        MEMORY_LIMIT = '6GB'
        MAX_CONCURRENT_PROCESSES = 8
        BATCH_SIZE = 20
        
        # Default performance config
        PERFORMANCE_CONFIG = {
            # Parallel Processing
            'enable_multiprocessing': True,
            'enable_priority_queue': True,
            'enable_adaptive_workers': True,
            'max_workers': 16,
            'process_workers': 8,
            'batch_size': 20,
            'task_timeout': 120,
            
            # Memory Management
            'memory_threshold': 0.75,
            'predictive_cleanup': True,
            'aggressive_cleanup': True,
            'cleanup_interval': 180,
            
            # Caching Strategy
            'enable_advanced_caching': True,
            'cache_size': 2000,
            'cache_timeout': 3600,
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
        
        # Default image processing config
        IMAGE_PROCESSING_CONFIG = {
            'use_gpu': True,
            'max_image_dimension': 8192,
            'jpeg_quality': 95,
            'png_compression': 6,
            'webp_quality': 95,
            'svg_simplify_tolerance': 0.1,
            'enable_multithreading': True,
            'thread_count': 4,
            'resize_method': 'lanczos',
            'cache_size': 1000,
            'cache_timeout': 3600
        }
        
        # Default cache config
        CACHE_SIZE = 2000
        CACHE_TIMEOUT = 3600
        LRU_CACHE_SIZE = 1000
        
        # Default task scheduling
        TASK_SCHEDULER_CONFIG = {
            'enable_priority_queue': True,
            'enable_task_preemption': True,
            'max_queue_size': 1000,
            'task_timeout': 300,
            'retry_attempts': 3,
            'retry_delay': 1
        }
        
        # Default resource monitoring
        RESOURCE_MONITORING = {
            'enable_monitoring': True,
            'monitoring_interval': 30,
            'alert_thresholds': {
                'cpu_usage': 80,
                'memory_usage': 85,
                'disk_usage': 90
            },
            'auto_optimization': True
        }
        
        # Default Celery config
        CELERY_REDIS_MAX_CONNECTIONS = 1000

    @classmethod
    def init_app(cls, app):
        """Initialize application directories and configure database engine options"""
        # Set engine options based on the actual database URI
        db_uri = app.config.get('SQLALCHEMY_DATABASE_URI', '')
        app.config['SQLALCHEMY_ENGINE_OPTIONS'] = cls.get_safe_engine_options(db_uri)
        
        # Log the configuration for debugging
        app.logger.info(f"Database URI: {db_uri}")
        app.logger.info(f"Engine options: {app.config['SQLALCHEMY_ENGINE_OPTIONS']}")
        
        # Create required directories with proper permissions
        directories_to_create = []
        
        if os.environ.get('PLATFORM') == 'fly':
            # On Fly.io, use the mounted volume paths
            directories_to_create = [
                '/app/data/uploads',
                '/app/data/outputs', 
                '/app/data/cache',
                '/app/data/temp'
            ]
        else:
            # Local development
            directories_to_create = [
                cls.UPLOAD_FOLDER,
                cls.OUTPUT_FOLDER, 
                cls.CACHE_FOLDER,
                cls.TEMP_FOLDER
            ]
        
        for directory in directories_to_create:
            try:
                os.makedirs(directory, exist_ok=True)
                os.chmod(directory, 0o750)  # rwxr-x---
                app.logger.info(f"Created directory: {directory}")
            except OSError as e:
                app.logger.error(f"Error creating directory {directory}: {e}") 