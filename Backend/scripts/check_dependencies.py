#!/usr/bin/env python3
"""
Dependency Check Script for ZYPPTS
Checks if all required packages are installed and working
"""

import sys
import importlib
from pathlib import Path

def check_package(package_name, import_name=None, required=True):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, '__version__', 'Unknown')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError as e:
        if required:
            print(f"❌ {package_name}: Not installed - {e}")
            return False
        else:
            print(f"⚠️  {package_name}: Not installed (optional) - {e}")
            return True
    except Exception as e:
        print(f"⚠️  {package_name}: Import error - {e}")
        return True

def main():
    """Check all dependencies"""
    print("🔍 ZYPPTS Dependency Check")
    print("=" * 50)
    
    # Core Flask dependencies
    print("\n📦 Core Flask Dependencies:")
    print("-" * 30)
    flask_deps = [
        ("Flask", "flask"),
        ("Flask-SQLAlchemy", "flask_sqlalchemy"),
        ("Flask-Login", "flask_login"),
        ("Flask-WTF", "flask_wtf"),
        ("Flask-Mail", "flask_mail"),
        ("Flask-Limiter", "flask_limiter"),
        ("Flask-Caching", "flask_caching"),
        ("Flask-Session", "flask_session"),
        ("Werkzeug", "werkzeug"),
    ]
    
    flask_results = []
    for package, import_name in flask_deps:
        flask_results.append(check_package(package, import_name))
    
    # Database dependencies
    print("\n🗄️ Database Dependencies:")
    print("-" * 30)
    db_deps = [
        ("redis", "redis"),
        ("psycopg2-binary", "psycopg2"),
        ("SQLAlchemy-Utils", "sqlalchemy_utils"),
    ]
    
    db_results = []
    for package, import_name in db_deps:
        db_results.append(check_package(package, import_name))
    
    # Image Processing dependencies
    print("\n🖼️ Image Processing Dependencies:")
    print("-" * 30)
    image_deps = [
        ("Pillow", "PIL"),
        ("numpy", "numpy"),
        ("opencv-contrib-python", "cv2"),
        ("scikit-image", "skimage"),
    ]
    
    image_results = []
    for package, import_name in image_deps:
        image_results.append(check_package(package, import_name))
    
    # Scientific Computing dependencies
    print("\n🔬 Scientific Computing Dependencies:")
    print("-" * 30)
    sci_deps = [
        ("scipy", "scipy"),
        ("scikit-learn", "sklearn"),
    ]
    
    sci_results = []
    for package, import_name in sci_deps:
        sci_results.append(check_package(package, import_name))
    
    # AI/ML dependencies
    print("\n🤖 AI/ML Dependencies:")
    print("-" * 30)
    ai_deps = [
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("rembg", "rembg"),
        ("onnxruntime", "onnxruntime"),
    ]
    
    ai_results = []
    for package, import_name in ai_deps:
        ai_results.append(check_package(package, import_name))
    
    # PDF and SVG dependencies
    print("\n📄 PDF and SVG Dependencies:")
    print("-" * 30)
    pdf_deps = [
        ("reportlab", "reportlab"),
        ("svgwrite", "svgwrite"),
        ("lxml", "lxml"),
        ("shapely", "shapely"),
        ("svg.path", "svg.path"),
        ("CairoSVG", "cairosvg"),
        ("svglib", "svglib"),
        ("svgpathtools", "svgpathtools"),
        ("pdf2image", "pdf2image"),
        ("PyMuPDF", "fitz"),
    ]
    
    pdf_results = []
    for package, import_name in pdf_deps:
        pdf_results.append(check_package(package, import_name))
    
    # Vector Tracing dependencies
    print("\n🎨 Vector Tracing Dependencies:")
    print("-" * 30)
    vector_deps = [
        ("vtracer", "vtracer"),
        ("psd-tools", "psd_tools"),
    ]
    
    vector_results = []
    for package, import_name in vector_deps:
        vector_results.append(check_package(package, import_name))
    
    # Payment Processing dependencies
    print("\n💳 Payment Processing Dependencies:")
    print("-" * 30)
    payment_deps = [
        ("stripe", "stripe"),
    ]
    
    payment_results = []
    for package, import_name in payment_deps:
        payment_results.append(check_package(package, import_name))
    
    # Utility dependencies
    print("\n🛠️ Utility Dependencies:")
    print("-" * 30)
    util_deps = [
        ("psutil", "psutil"),
        ("python-dotenv", "dotenv"),
        ("gunicorn", "gunicorn"),
        ("email-validator", "email_validator"),
        ("tqdm", "tqdm"),
        ("requests", "requests"),
    ]
    
    util_results = []
    for package, import_name in util_deps:
        util_results.append(check_package(package, import_name))
    
    # Monitoring dependencies
    print("\n📊 Monitoring Dependencies:")
    print("-" * 30)
    monitor_deps = [
        ("prometheus_client", "prometheus_client"),
        ("statsd", "statsd"),
    ]
    
    monitor_results = []
    for package, import_name in monitor_deps:
        monitor_results.append(check_package(package, import_name))
    
    # Production dependencies
    print("\n🚀 Production Dependencies:")
    print("-" * 30)
    prod_deps = [
        ("sentry-sdk", "sentry_sdk"),
        ("structlog", "structlog"),
        ("python-json-logger", "pythonjsonlogger"),
        ("healthcheck", "healthcheck"),
        ("redis-py-cluster", "rediscluster"),
        ("APScheduler", "apscheduler"),
    ]
    
    prod_results = []
    for package, import_name in prod_deps:
        prod_results.append(check_package(package, import_name))
    
    # Development dependencies (optional)
    print("\n🔧 Development Dependencies (Optional):")
    print("-" * 40)
    dev_deps = [
        ("pytest", "pytest"),
        ("black", "black"),
        ("flake8", "flake8"),
    ]
    
    dev_results = []
    for package, import_name in dev_deps:
        dev_results.append(check_package(package, import_name, required=False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Dependency Check Summary:")
    print("=" * 50)
    
    all_results = (
        flask_results + db_results + image_results + sci_results + 
        ai_results + pdf_results + vector_results + payment_results + 
        util_results + monitor_results + prod_results
    )
    
    required_results = [r for r in all_results if r is not None]
    failed_required = [r for r in required_results if not r]
    
    print(f"✅ Required packages installed: {len(required_results) - len(failed_required)}/{len(required_results)}")
    
    if failed_required:
        print(f"❌ Missing required packages: {len(failed_required)}")
        print("\n📦 To install missing packages:")
        print("1. Create a virtual environment:")
        print("   python3 -m venv venv")
        print("   source venv/bin/activate")
        print("2. Install requirements:")
        print("   pip install -r requirements.txt")
    else:
        print("🎉 All required dependencies are installed!")
        print("\n✅ Your ZYPPTS environment is ready!")
    
    # Check Python version
    print(f"\n🐍 Python Version: {sys.version}")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✅ Running in virtual environment")
    else:
        print("⚠️  Not running in virtual environment (recommended for development)")

if __name__ == "__main__":
    main() 