"""
App configuration for the new Frontend/Backend structure
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_mail import Mail
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_caching import Cache
from flask_session import Session

# Initialize Flask extensions
db = SQLAlchemy()
login_manager = LoginManager()
mail = Mail()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per hour", "500 per day"],
    storage_uri=os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
)
cache = Cache()
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
    from config import Config
    app.config.from_object(Config)
    
    # Override paths in config for new structure
    app.config['TEMPLATES_FOLDER'] = templates_folder
    app.config['STATIC_FOLDER'] = static_folder
    app.config['UPLOAD_FOLDER'] = os.path.join(backend_dir, 'uploads')
    app.config['OUTPUT_FOLDER'] = os.path.join(backend_dir, 'outputs')
    app.config['CACHE_FOLDER'] = os.path.join(backend_dir, 'cache')
    app.config['TEMP_FOLDER'] = os.path.join(backend_dir, 'temp')
    
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
    
    # Configure Redis connections for production
    import redis
    redis_configured = False
    
    try:
        # Get Redis configuration from environment
        redis_url = os.environ.get('REDIS_URL')
        if redis_url:
            # Parse Redis URL for production (Render format)
            app.logger.info(f"Configuring Redis with URL: {redis_url}")
            
            # Configure Flask-Session for Redis
            app.config['SESSION_TYPE'] = 'redis'
            app.config['SESSION_REDIS'] = redis.from_url(f"{redis_url}/1", decode_responses=False)
            app.config['SESSION_PERMANENT'] = False
            app.config['SESSION_USE_SIGNER'] = True
            app.config['SESSION_KEY_PREFIX'] = 'zyppts:'
            
            # Configure Flask-Limiter for Redis  
            app.config['RATELIMIT_STORAGE_URL'] = f"{redis_url}/0"
            
            # Configure Flask-Cache for Redis
            app.config['CACHE_TYPE'] = 'redis'
            app.config['CACHE_REDIS_URL'] = f"{redis_url}/0"
            app.config['CACHE_DEFAULT_TIMEOUT'] = 300
            
            # Test connection
            test_redis = redis.from_url(redis_url)
            test_redis.ping()
            redis_configured = True
            app.logger.info("✅ Redis connections configured successfully")
            
    except Exception as redis_error:
        app.logger.warning(f"⚠️ Redis not available ({redis_error}), falling back to filesystem storage")
        redis_configured = False
    
    # Fallback to filesystem if Redis is not available
    if not redis_configured:
        app.logger.info("🔧 Using filesystem sessions and simple cache")
        app.config['SESSION_TYPE'] = 'filesystem'
        app.config['SESSION_FILE_DIR'] = os.path.join(backend_dir, 'sessions')
        app.config['SESSION_PERMANENT'] = False
        app.config['SESSION_USE_SIGNER'] = True
        app.config['SESSION_KEY_PREFIX'] = 'zyppts:'
        
        # Simple cache configuration (remove Redis-specific options)
        app.config['CACHE_TYPE'] = 'simple'
        app.config['CACHE_DEFAULT_TIMEOUT'] = 300
        # Remove Redis-specific options that don't work with SimpleCache
        app.config.pop('CACHE_REDIS_URL', None)
        app.config.pop('CACHE_REDIS_HOST', None)
        app.config.pop('CACHE_REDIS_PORT', None)
        app.config.pop('CACHE_REDIS_PASSWORD', None)
        app.config.pop('CACHE_REDIS_DB', None)
        app.config.pop('CACHE_OPTIONS', None)
        
        # Create sessions directory if it doesn't exist
        sessions_dir = os.path.join(backend_dir, 'sessions')
        if not os.path.exists(sessions_dir):
            os.makedirs(sessions_dir, mode=0o750)
    
    # Initialize SQLAlchemy with app
    db.init_app(app)
    
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
        from models import User
        return User.query.get(int(user_id))
    
    # Initialize other extensions
    mail.init_app(app)
    limiter.init_app(app)
    cache.init_app(app)
    session.init_app(app)
    
    # Add error handlers for production
    @app.errorhandler(500)
    def internal_error(error):
        db.session.rollback()
        app.logger.error(f'Server Error: {error}')
        return "Internal server error occurred. Please try again later.", 500
    
    @app.errorhandler(404)
    def not_found_error(error):
        return "Page not found.", 404
    
    # Add session error handling
    @app.errorhandler(UnicodeDecodeError)
    def handle_session_corruption(error):
        """Handle corrupted session data gracefully"""
        try:
            from flask import session as flask_session
            flask_session.clear()
            return redirect(url_for('main.home'))
        except Exception as e:
            app.logger.error(f"Session corruption error: {str(e)}")
            return "Session error occurred. Please clear your browser cookies and try again.", 500
    
    # Initialize app configuration
    Config.init_app(app)
    
    # Create database tables
    with app.app_context():
        try:
            db.create_all()
            app.logger.info("Database tables created successfully")
        except Exception as e:
            app.logger.error(f"Error creating database tables: {e}")
    
    # Register blueprints
    from routes import bp as main_bp
    app.register_blueprint(main_bp)
    
    # Register admin blueprint if available
    try:
        from admin_routes import admin_bp
        app.register_blueprint(admin_bp)
        app.logger.info("Admin routes registered")
    except ImportError:
        app.logger.info("Admin routes not available")
    
    # Initialize scheduled tasks for email notifications
    try:
        from utils.scheduled_tasks import start_scheduled_tasks
        start_scheduled_tasks()
        app.logger.info("📧 Email scheduling system initialized")
    except Exception as e:
        app.logger.error(f"Failed to initialize email scheduling: {e}")
    
    return app 