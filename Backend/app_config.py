"""
App configuration for the new Frontend/Backend structure
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, redirect, url_for, render_template
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
from werkzeug.middleware.proxy_fix import ProxyFix

# Make Redis import more robust with proper error handling
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError as e:
    REDIS_AVAILABLE = False
    redis = None
    print(f"Warning: Redis library not available - {e}")

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
        if not REDIS_AVAILABLE or redis is None:
            return False, "Redis library not available"
        
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

# Initialize limiter with fallback if Redis is not available
if REDIS_AVAILABLE:
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=["100 per hour", "500 per day"],
        storage_uri=os.environ.get('REDIS_URL', 'redis://localhost:6379/0')  # Use Redis for rate limiting
    )
else:
    # Fallback to memory storage for rate limiting
    limiter = Limiter(
        key_func=get_remote_address,
        default_limits=["100 per hour", "500 per day"],
        storage_uri="memory://"
    )

if CACHE_AVAILABLE:
    cache = Cache()
else:
    cache = None
session = Session()

def create_app():
    """Create and configure the Flask application with new structure"""
    
    # Get paths for new structure
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(backend_dir)
    frontend_dir = os.path.join(project_root, 'Frontend')
    
    # Set up template and static folders
    templates_folder = os.path.join(frontend_dir, 'templates')
    static_folder = os.path.join(frontend_dir, 'static')
    
    # Create Flask app with correct paths
    app = Flask(__name__,
                template_folder=templates_folder,
                static_folder=static_folder)
    
    # Load configuration
    try:
        from .config import Config
    except ImportError:
        from config import Config
    app.config.from_object(Config)
    
    # Ensure app directories exist based on config
    try:
        Config.init_app(app)
    except Exception as e:
        app.logger.warning(f"Config.init_app failed: {e}")
    
    # Respect proxy headers (X-Forwarded-For, X-Forwarded-Proto) on Fly.io
    if os.environ.get('PLATFORM') == 'fly':
        app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
    
    # Override paths in config for new structure
    app.config['TEMPLATES_FOLDER'] = templates_folder
    app.config['STATIC_FOLDER'] = static_folder
    
    # Configure logging for production
    if not app.debug and not app.testing:
        # Create logs directory
        logs_dir = os.path.join(backend_dir, 'logs')
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir, mode=0o750)
        
        # Set up file logging
        file_handler = RotatingFileHandler(
            os.path.join(logs_dir, 'zyppts.log'),
            maxBytes=10240000,  # 10MB
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Zyppts startup')
    
    # Configure sessions to use Redis
    redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
    redis_config = parse_redis_url(redis_url)
    
    redis_available, redis_status = test_redis_connection()

    if redis_available:
        try:
            app.config['SESSION_TYPE'] = 'redis'
            app.config['SESSION_REDIS'] = redis.from_url(redis_url, db=0)  # Changed from db=1 to db=0
            app.logger.info(f"Redis sessions configured: {redis_status}")
        except Exception as e:
            app.logger.warning(f"Redis session config failed, using filesystem fallback: {e}")
            redis_available = False
    
    if not redis_available:
        # Fallback to filesystem sessions
        app.config['SESSION_TYPE'] = 'filesystem'
        session_dir = os.path.join(backend_dir, 'logs', 'sessions')
        app.config['SESSION_FILE_DIR'] = session_dir
        app.config['SESSION_FILE_THRESHOLD'] = 500
        app.config['SESSION_FILE_MODE'] = 0o600
        try:
            if not os.path.exists(session_dir):
                os.makedirs(session_dir, mode=0o700)
            app.logger.warning(f"Redis not available, using filesystem sessions: {redis_status}")
        except Exception as e:
            app.logger.error(f"Failed to create session directory: {e}")
            # Use default Flask session if all else fails
            app.config['SESSION_TYPE'] = 'null'
    
    # Initialize SQLAlchemy with app
    db.init_app(app)
    
    # Log database configuration for debugging
    db_uri = app.config.get('SQLALCHEMY_DATABASE_URI', 'Not configured')
    engine_options = app.config.get('SQLALCHEMY_ENGINE_OPTIONS', {})
    db_type = 'SQLite' if 'sqlite' in db_uri.lower() else 'Other'
    app.logger.info(f"Database Type: {db_type}")
    app.logger.info(f"Database URI: {db_uri}")
    app.logger.info(f"Engine Options: {engine_options}")
    
    # Initialize Flask-Login
    login_manager.init_app(app)
    login_manager.login_view = 'main.login'
    login_manager.login_message = 'Please log in to access this page.'
    login_manager.login_message_category = 'info'
    login_manager.session_protection = 'strong'
    
    # User loader function
    @login_manager.user_loader
    def load_user(user_id):
        if user_id is None:
            return None
        try:
            from .models import User
        except ImportError:
            try:
                from models import User
            except ImportError:
                # Return None if models are not available
                return None
        return User.query.get(int(user_id))
    
    # Initialize other extensions
    mail.init_app(app)
    limiter.init_app(app)
    session.init_app(app)
    
    # Configure email settings
    app.config.setdefault('MAIL_SERVER', 'smtp.gmail.com')
    app.config.setdefault('MAIL_PORT', 587)
    app.config.setdefault('MAIL_USE_TLS', True)
    app.config.setdefault('MAIL_USERNAME', os.environ.get('MAIL_USERNAME'))
    app.config.setdefault('MAIL_PASSWORD', os.environ.get('MAIL_PASSWORD'))
    app.config.setdefault('MAIL_DEFAULT_SENDER', os.environ.get('MAIL_DEFAULT_SENDER', 'Zyppts HQ <zyppts@gmail.com>'))
    app.config.setdefault('ADMIN_ALERT_EMAIL', os.environ.get('ADMIN_ALERT_EMAIL', os.environ.get('MAIL_USERNAME')))
    
    # Log email configuration status
    if app.config.get('MAIL_USERNAME') and app.config.get('MAIL_PASSWORD'):
        app.logger.info("Email configuration: ✅ Configured")
    else:
        app.logger.warning("Email configuration: ⚠️ Missing MAIL_USERNAME or MAIL_PASSWORD environment variables")
    
    # Initialize caching if available
    if CACHE_AVAILABLE:
        cache.init_app(app)
        if redis_available and REDIS_AVAILABLE and redis is not None:
            # Configure Redis cache
            app.config['CACHE_TYPE'] = 'redis'
            app.config['CACHE_REDIS_HOST'] = redis_config['host']
            app.config['CACHE_REDIS_PORT'] = redis_config['port']
            app.config['CACHE_REDIS_PASSWORD'] = redis_config['password']
            app.config['CACHE_REDIS_DB'] = 0  # Use separate DB for cache
            app.config['CACHE_DEFAULT_TIMEOUT'] = 300
            app.config['CACHE_KEY_PREFIX'] = 'zyppts_'
            app.logger.info("Redis cache configured")
        else:
            # Fallback to memory cache
            app.config['CACHE_TYPE'] = 'simple'
            app.config['CACHE_DEFAULT_TIMEOUT'] = 300
            app.logger.warning("Redis not available, using memory cache")
    
    # Create database tables
    with app.app_context():
        try:
            db_uri = app.config.get('SQLALCHEMY_DATABASE_URI', '')
            if ':memory:' in db_uri:
                app.logger.info("Database: Using in-memory SQLite")
            elif 'postgresql://' in db_uri:
                app.logger.info("Database: Using PostgreSQL")
            else:
                app.logger.info("Database: Using file-based SQLite")
            
            # Import models to ensure they're registered with SQLAlchemy
            try:
                app.logger.info("Attempting to import models...")
                from .models import User, Subscription, LogoUpload, LogoVariation, UserAnalytics, UserSession, UserMetrics, SubscriptionAnalytics
                app.logger.info("Models imported successfully")
            except ImportError as e:
                app.logger.error(f"Failed to import models with relative import: {e}")
                try:
                    app.logger.info("Attempting fallback model import...")
                    from models import User, Subscription, LogoUpload, LogoVariation, UserAnalytics, UserSession, UserMetrics, SubscriptionAnalytics
                    app.logger.info("Models imported successfully (fallback)")
                except ImportError as e2:
                    app.logger.error(f"Failed to import models with fallback import: {e2}")
                    app.logger.error("Continuing without models - basic tables will still be created")
                    # Set models to None to prevent errors later
                    User = None
                    Subscription = None
            
            # Create all tables first
            db.create_all()
            app.logger.info("Database tables created successfully")
            
            # Force commit to ensure tables are persisted
            db.session.commit()
            
            # Test basic connection
            from sqlalchemy import text
            db.session.execute(text('SELECT 1'))
            db.session.commit()
            app.logger.info("Database connection test successful")
            
            # Now try to seed users - with better error handling
            try:
                from datetime import datetime
                
                # Check if models were imported successfully
                if User is None or Subscription is None:
                    app.logger.warning("Models not available, skipping user seeding")
                else:
                    # Check if users table exists by trying to query it
                    try:
                        # Use raw SQL to check if table exists (works for both SQLite and PostgreSQL)
                        if 'postgresql://' in db_uri:
                            result = db.session.execute(text("SELECT tablename FROM pg_tables WHERE tablename = 'users'"))
                        else:
                            result = db.session.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='users'"))
                        
                        table_exists = result.fetchone() is not None
                        
                        if table_exists:
                            user_count = User.query.count()
                            app.logger.info(f"Found {user_count} existing users in database")
                            
                            if user_count == 0:
                                app.logger.info("No users found. Seeding default admin and owner account...")
                                admin_user = User(
                                    username='admin',
                                    email='admin@usezyppts.com',
                                    is_admin=True,
                                    is_active=True,
                                    is_beta=True,
                                    created_at=datetime.utcnow()
                                )
                                admin_user.set_password('admin123')
                                owner_user = User(
                                    username='mike',
                                    email='mike@usezyppts.com',
                                    is_admin=True,
                                    is_active=True,
                                    is_beta=True,
                                    created_at=datetime.utcnow()
                                )
                                owner_user.set_password('admin123')
                                admin_subscription = Subscription(
                                    user=admin_user,
                                    plan='enterprise',
                                    status='active',
                                    monthly_credits=-1,
                                    start_date=datetime.utcnow(),
                                    billing_cycle='annual'
                                )
                                owner_subscription = Subscription(
                                    user=owner_user,
                                    plan='enterprise',
                                    status='active',
                                    monthly_credits=-1,
                                    start_date=datetime.utcnow(),
                                    billing_cycle='annual'
                                )
                                db.session.add_all([admin_user, owner_user, admin_subscription, owner_subscription])
                                db.session.commit()
                                app.logger.info("Seeded default users: admin & mike")
                            else:
                                app.logger.info("Users already exist, skipping seeding")
                        else:
                            app.logger.warning("Users table does not exist, skipping user seeding")
                            
                    except Exception as table_error:
                        app.logger.warning(f"Could not query users table: {table_error}")
                        # Continue without seeding - tables will be created on first use
                    
            except Exception as seed_error:
                app.logger.error(f"Failed to seed default users: {seed_error}")
            
            # Initialize promo codes
            try:
                # First, ensure the database schema is up to date
                try:
                    # Check if promo code columns exist in users table
                    if 'postgresql://' in db_uri:
                        # PostgreSQL
                        result = db.session.execute(text("""
                            SELECT column_name FROM information_schema.columns 
                            WHERE table_name = 'users' AND column_name IN ('promo_code_used', 'promo_code_applied')
                        """))
                        existing_columns = [row[0] for row in result.fetchall()]
                    else:
                        # SQLite
                        result = db.session.execute(text("PRAGMA table_info(users)"))
                        existing_columns = [row[1] for row in result.fetchall()]
                    
                    # Add missing columns
                    if 'promo_code_used' not in existing_columns:
                        try:
                            if 'postgresql://' in db_uri:
                                db.session.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS promo_code_used VARCHAR(50)"))
                            else:
                                db.session.execute(text("ALTER TABLE users ADD COLUMN promo_code_used VARCHAR(50)"))
                            app.logger.info("✅ Added promo_code_used column to users table")
                        except Exception as e:
                            app.logger.warning(f"Could not add promo_code_used column: {e}")
                    
                    if 'promo_code_applied' not in existing_columns:
                        try:
                            if 'postgresql://' in db_uri:
                                db.session.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS promo_code_applied BOOLEAN DEFAULT FALSE"))
                            else:
                                db.session.execute(text("ALTER TABLE users ADD COLUMN promo_code_applied BOOLEAN DEFAULT FALSE"))
                            app.logger.info("✅ Added promo_code_applied column to users table")
                        except Exception as e:
                            app.logger.warning(f"Could not add promo_code_applied column: {e}")
                    
                    # Create promo_codes table if it doesn't exist
                    try:
                        if 'postgresql://' in db_uri:
                            # PostgreSQL
                            result = db.session.execute(text("""
                                SELECT EXISTS (
                                    SELECT FROM information_schema.tables 
                                    WHERE table_name = 'promo_codes'
                                )
                            """))
                            table_exists = result.fetchone()[0]
                        else:
                            # SQLite
                            result = db.session.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='promo_codes'"))
                            table_exists = result.fetchone() is not None
                        
                        if not table_exists:
                            if 'postgresql://' in db_uri:
                                # PostgreSQL
                                db.session.execute(text("""
                                    CREATE TABLE IF NOT EXISTS promo_codes (
                                        id SERIAL PRIMARY KEY,
                                        code VARCHAR(50) UNIQUE NOT NULL,
                                        description VARCHAR(200),
                                        discount_percentage INTEGER DEFAULT 0,
                                        discount_amount FLOAT DEFAULT 0.0,
                                        max_uses INTEGER,
                                        current_uses INTEGER DEFAULT 0,
                                        is_active BOOLEAN DEFAULT TRUE,
                                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                                        expires_at TIMESTAMP,
                                        early_access BOOLEAN DEFAULT FALSE,
                                        bonus_credits INTEGER DEFAULT 0,
                                        plan_upgrade VARCHAR(20)
                                    )
                                """))
                            else:
                                # SQLite
                                db.session.execute(text("""
                                    CREATE TABLE IF NOT EXISTS promo_codes (
                                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                                        code VARCHAR(50) UNIQUE NOT NULL,
                                        description VARCHAR(200),
                                        discount_percentage INTEGER DEFAULT 0,
                                        discount_amount FLOAT DEFAULT 0.0,
                                        max_uses INTEGER,
                                        current_uses INTEGER DEFAULT 0,
                                        is_active BOOLEAN DEFAULT TRUE,
                                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                                        expires_at DATETIME,
                                        early_access BOOLEAN DEFAULT FALSE,
                                        bonus_credits INTEGER DEFAULT 0,
                                        plan_upgrade VARCHAR(20)
                                    )
                                """))
                            
                            db.session.commit()
                            app.logger.info("✅ Created promo_codes table")
                        else:
                            app.logger.info("✅ Promo_codes table already exists")
                    
                    except Exception as e:
                        app.logger.warning(f"Could not create promo_codes table: {e}")
                    
                    # Create EARLYZYPPTS promo code if it doesn't exist
                    try:
                        result = db.session.execute(text("SELECT id FROM promo_codes WHERE code = 'EARLYZYPPTS'"))
                        existing_code = result.fetchone()
                        
                        if not existing_code:
                            if 'postgresql://' in db_uri:
                                # PostgreSQL
                                db.session.execute(text("""
                                    INSERT INTO promo_codes (
                                        code, description, early_access, bonus_credits, 
                                        plan_upgrade, max_uses, is_active, expires_at,
                                        discount_percentage, discount_amount
                                    ) VALUES (
                                        'EARLYZYPPTS', 
                                        'Early Access - Pro features with free credits',
                                        TRUE, 0, NULL, NULL, TRUE, NULL, 0, 0.0
                                    )
                                """))
                            else:
                                # SQLite
                                db.session.execute(text("""
                                    INSERT INTO promo_codes (
                                        code, description, early_access, bonus_credits, 
                                        plan_upgrade, max_uses, is_active, expires_at,
                                        discount_percentage, discount_amount
                                    ) VALUES (
                                        'EARLYZYPPTS', 
                                        'Early Access - Pro features with free credits',
                                        1, 0, NULL, NULL, 1, NULL, 0, 0.0
                                    )
                                """))
                            
                            db.session.commit()
                            app.logger.info("✅ EARLYZYPPTS promo code created successfully")
                        else:
                            app.logger.info("✅ EARLYZYPPTS promo code already exists")
                    
                    except Exception as e:
                        app.logger.warning(f"Could not create EARLYZYPPTS promo code: {e}")
                
                except Exception as schema_error:
                    app.logger.warning(f"Database schema migration failed: {schema_error}")
                
                # Try to import and create promo codes
                try:
                    from utils.promo_codes import create_early_zyppts_promo_code
                    promo_code = create_early_zyppts_promo_code()
                    if promo_code:
                        app.logger.info("✅ EARLYZYPPTS promo code initialized")
                    else:
                        app.logger.warning("⚠️ Failed to initialize EARLYZYPPTS promo code")
                except ImportError as e:
                    app.logger.warning(f"⚠️ Could not import promo_codes module: {e}")
                except Exception as promo_error:
                    app.logger.error(f"Failed to initialize promo codes: {promo_error}")
            except Exception as promo_error:
                app.logger.error(f"Failed to initialize promo codes: {promo_error}")
            
        except Exception as e:
            app.logger.error(f"Database initialization failed: {e}")
            # Don't crash the app - continue without database for now
            pass
    
    # Register blueprints
    try:
        from .routes import bp as main_bp
    except ImportError:
        try:
            from routes import bp as main_bp
        except ImportError:
            # Create a dummy blueprint if routes are not available
            from flask import Blueprint
            main_bp = Blueprint('main', __name__)
            @main_bp.route('/')
            def home():
                return "Application is running"
    
    app.register_blueprint(main_bp)
    
    # Register admin routes if available
    try:
        from .admin_routes import admin_bp
        app.register_blueprint(admin_bp, url_prefix='/admin')
    except ImportError:
        app.logger.warning("Admin routes not available: attempted relative import with no known parent package")
        try:
            from admin_routes import admin_bp
            app.register_blueprint(admin_bp, url_prefix='/admin')
        except ImportError:
            app.logger.warning("Admin routes not available")
    
    # Error handlers
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        return "Internal Server Error", 500
    
    @app.errorhandler(404)
    def not_found_error(error):
        return "Page Not Found", 404
    
    @app.errorhandler(UnicodeDecodeError)
    def handle_session_corruption(error):
        # Clear corrupted session
        session.clear()
        return "Session error occurred. Please try again.", 400
    
    return app 