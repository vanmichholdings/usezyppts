#!/usr/bin/env python3
"""
Startup script for Zyppts v10 - ensures proper initialization before starting the app
"""

import os
import sys
import time
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_imports():
    """Check that all critical imports work"""
    logger.info("Checking critical imports...")
    
    try:
        # Test Flask and core dependencies
        import flask
        logger.info("✅ Flask imported successfully")
        
        # Test SQLAlchemy
        import sqlalchemy
        logger.info("✅ SQLAlchemy imported successfully")
        
        # Test Redis
        import redis
        logger.info("✅ Redis imported successfully")
        
        # Test our app modules
        sys.path.insert(0, '/app')
        from Backend.app_config import create_app
        logger.info("✅ App config imported successfully")
        
        # Test models
        from Backend.models import User, Subscription
        logger.info("✅ Models imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error during import check: {e}")
        return False

def test_app_creation():
    """Test that the app can be created successfully"""
    logger.info("Testing app creation...")
    
    try:
        from Backend.app_config import create_app
        
        # Create the app
        app = create_app()
        logger.info("✅ App created successfully")
        
        # Test basic functionality
        with app.app_context():
            # Test database connection
            from Backend.app_config import db
            from sqlalchemy import text
            db.session.execute(text('SELECT 1'))
            logger.info("✅ Database connection test passed")
            
            # Test Redis connection
            import redis
            redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379')
            r = redis.from_url(redis_url)
            r.ping()
            logger.info("✅ Redis connection test passed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ App creation failed: {e}")
        return False

def main():
    """Main startup function"""
    logger.info("🚀 Starting Zyppts v10...")
    
    # Check environment
    logger.info(f"Platform: {os.environ.get('PLATFORM', 'unknown')}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Check imports
    if not check_imports():
        logger.error("❌ Import check failed - exiting")
        sys.exit(1)
    
    # Test app creation
    if not test_app_creation():
        logger.error("❌ App creation test failed - exiting")
        sys.exit(1)
    
    logger.info("✅ All startup checks passed - app is ready to start!")
    
    # Start the actual app using Gunicorn
    import subprocess
    
    # Gunicorn command optimized for Fly.io
    cmd = [
        "gunicorn",
        "--bind", "0.0.0.0:8080",
        "--workers", "2",
        "--worker-class", "sync",
        "--timeout", "90",
        "--keep-alive", "2",
        "--max-requests", "500",
        "--max-requests-jitter", "50",
        "--worker-tmp-dir", "/tmp",
        "--preload",  # Preload the app to avoid import issues
        "Backend.app_config:create_app()"
    ]
    
    logger.info(f"Starting Gunicorn with command: {' '.join(cmd)}")
    
    try:
        # Start Gunicorn
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Gunicorn failed to start: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal - shutting down gracefully")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 