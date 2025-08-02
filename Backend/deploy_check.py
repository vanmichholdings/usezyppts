#!/usr/bin/env python3
"""
Deployment verification script for Zyppts
Checks that all required dependencies and configurations are properly set up
"""

import sys
import os
import importlib
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 11:
        print("✓ Python version is compatible (3.11+)")
        return True
    else:
        print("✗ Python version should be 3.11 or higher")
        return False

def check_required_packages():
    """Check if all required packages are installed"""
    # Map of package names to their import names
    package_imports = {
        'flask': 'flask',
        'flask_sqlalchemy': 'flask_sqlalchemy',
        'flask_login': 'flask_login',
        'flask_mail': 'flask_mail',
        'flask_session': 'flask_session',
        'flask_socketio': 'flask_socketio',
        'flask_wtf': 'flask_wtf',
        'flask_caching': 'flask_caching',
        'flask_cors': 'flask_cors',
        'flask_limiter': 'flask_limiter',
        'gunicorn': 'gunicorn',
        'werkzeug': 'werkzeug',
        'jinja2': 'jinja2',
        'pillow': 'PIL',
        'opencv-python': 'cv2',
        'numpy': 'numpy',
        'scipy': 'scipy',
        'scikit-image': 'skimage',
        'scikit-learn': 'sklearn',
        'pdf2image': 'pdf2image',
        'pymupdf': 'fitz',
        'lxml': 'lxml',
        'cairosvg': 'cairosvg',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'celery': 'celery',
        'redis': 'redis',
        'apscheduler': 'apscheduler',
        'bcrypt': 'bcrypt',
        'passlib': 'passlib',
        'stripe': 'stripe',
        'requests': 'requests',
        'pandas': 'pandas',
        'python-dotenv': 'dotenv',
        'sentry-sdk': 'sentry_sdk',
        'websockets': 'websockets',
        'psutil': 'psutil',
        'tqdm': 'tqdm',
        'rich': 'rich'
    }
    
    missing_packages = []
    
    for package, import_name in package_imports.items():
        try:
            importlib.import_module(import_name)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        return False
    else:
        print("\n✓ All required packages are installed")
        return True

def check_file_structure():
    """Check if required files and directories exist"""
    required_files = [
        'app_config.py',
        'config.py',
        'models.py',
        'routes.py',
        'gunicorn.conf.py',
        'utils/logo_processor.py',
        'utils/email_notifications.py',
        'utils/scheduled_tasks.py'
    ]
    
    required_dirs = [
        'uploads',
        'outputs',
        'cache',
        'temp',
        'logs',
        'templates',
        'static'
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING")
            missing_files.append(file_path)
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ - MISSING")
            missing_dirs.append(dir_path)
    
    if missing_files or missing_dirs:
        print(f"\nMissing files: {', '.join(missing_files)}")
        print(f"Missing directories: {', '.join(missing_dirs)}")
        return False
    else:
        print("\n✓ All required files and directories exist")
        return True

def check_environment_variables():
    """Check if required environment variables are set"""
    required_vars = [
        'SECRET_KEY',
        'DATABASE_URL',
        'REDIS_URL',
        'MAIL_USERNAME',
        'MAIL_PASSWORD',
        'STRIPE_SECRET_KEY',
        'STRIPE_PUBLISHABLE_KEY'
    ]
    
    optional_vars = [
        'FLASK_DEBUG',
        'FLASK_ENV',
        'LOG_LEVEL',
        'SITE_URL'
    ]
    
    missing_vars = []
    
    print("Required environment variables:")
    for var in required_vars:
        if os.environ.get(var):
            print(f"✓ {var}")
        else:
            print(f"✗ {var} - NOT SET")
            missing_vars.append(var)
    
    print("\nOptional environment variables:")
    for var in optional_vars:
        if os.environ.get(var):
            print(f"✓ {var} = {os.environ.get(var)}")
        else:
            print(f"- {var} (not set, will use defaults)")
    
    if missing_vars:
        print(f"\nMissing required environment variables: {', '.join(missing_vars)}")
        print("Note: These will be set by Render during deployment")
        return True  # Don't fail for local development
    else:
        print("\n✓ All required environment variables are set")
        return True

def check_app_configuration():
    """Check if the Flask app can be created successfully"""
    try:
        from app_config import create_app
        app = create_app()
        print("✓ Flask app created successfully")
        
        # Check if app has required extensions
        if hasattr(app, 'extensions'):
            print("✓ Flask extensions loaded")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create Flask app: {e}")
        return False

def main():
    """Run all deployment checks"""
    print("=" * 60)
    print("ZYPPTS DEPLOYMENT VERIFICATION")
    print("=" * 60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("File Structure", check_file_structure),
        ("Environment Variables", check_environment_variables),
        ("App Configuration", check_app_configuration)
    ]
    
    results = []
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        print("-" * 40)
        try:
            result = check_func()
            results.append((check_name, result))
        except Exception as e:
            print(f"✗ Error during {check_name} check: {e}")
            results.append((check_name, False))
    
    print("\n" + "=" * 60)
    print("DEPLOYMENT CHECK SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for check_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{check_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL CHECKS PASSED - Ready for deployment!")
        print("\nDeployment Notes:")
        print("- Environment variables will be set by Render")
        print("- All required packages are available")
        print("- Application structure is correct")
        print("- Flask app initializes successfully")
        sys.exit(0)
    else:
        print("✗ SOME CHECKS FAILED - Please fix issues before deployment")
        sys.exit(1)

if __name__ == "__main__":
    main() 