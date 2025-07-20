"""
App configuration for the new Frontend/Backend structure
"""

import os
from flask import Flask
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
    default_limits=["200 per day", "50 per hour"]
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
    
    # Initialize SQLAlchemy with app
    db.init_app(app)
    
    # Set up logging
    logs_dir = os.path.join(backend_dir, 'logs')
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir, mode=0o750)
    
    file_handler = logging.FileHandler(os.path.join(logs_dir, 'zyppts.log'))
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    )
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    
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
    
    # Configure Redis
    app.config['CACHE_TYPE'] = 'redis'
    app.config['CACHE_REDIS_HOST'] = 'localhost'
    app.config['CACHE_REDIS_PORT'] = 6379
    app.config['CACHE_DEFAULT_TIMEOUT'] = 300
    app.config['RATELIMIT_STORAGE_URL'] = 'redis://localhost:6379/0'
    
    # Initialize app configuration
    Config.init_app(app)
    
    # Create database tables
    with app.app_context():
        db.create_all()
    
    # Register blueprints
    from routes import bp as main_bp
    app.register_blueprint(main_bp)
    
    return app

import logging 