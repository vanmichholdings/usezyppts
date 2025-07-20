#!/usr/bin/env python3

import os
import sys

def test_structure():
    print("🔍 Testing Complete Structure...")
    
    # Check directories - now we're in Backend/tests, so go up two levels
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    backend_dir = os.path.join(project_root, 'Backend')
    frontend_dir = os.path.join(project_root, 'Frontend')
    
    print(f"Project root: {project_root}")
    print(f"Backend dir: {backend_dir}")
    print(f"Frontend dir: {frontend_dir}")
    
    # Check if directories exist
    if os.path.exists(backend_dir):
        print("✅ Backend directory exists")
    else:
        print("❌ Backend directory missing")
        return False
    
    if os.path.exists(frontend_dir):
        print("✅ Frontend directory exists")
    else:
        print("❌ Frontend directory missing")
        return False
    
    # Check Frontend contents
    templates_dir = os.path.join(frontend_dir, 'templates')
    static_dir = os.path.join(frontend_dir, 'static')
    
    if os.path.exists(templates_dir):
        print(f"✅ Templates directory exists: {templates_dir}")
        templates = os.listdir(templates_dir)
        print(f"   Found {len(templates)} template files")
    else:
        print("❌ Templates directory missing")
        return False
    
    if os.path.exists(static_dir):
        print(f"✅ Static directory exists: {static_dir}")
        static_items = os.listdir(static_dir)
        print(f"   Found {len(static_items)} static items")
    else:
        print("❌ Static directory missing")
        return False
    
    return True

def test_imports():
    print("\n🔍 Testing Imports...")
    
    try:
        # Add Backend to path - now we're in Backend/tests, so go up one level
        current_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(current_dir)
        sys.path.insert(0, backend_dir)
        
        # Test basic imports
        import numpy as np
        print("✅ NumPy imported")
        
        import cv2
        print("✅ OpenCV imported")
        
        from PIL import Image
        print("✅ Pillow imported")
        
        import torch
        print("✅ PyTorch imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_flask_app():
    print("\n🔍 Testing Flask App...")
    
    try:
        # Add Backend to path - now we're in Backend/tests, so go up one level
        current_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(current_dir)
        sys.path.insert(0, backend_dir)
        
        from app_config import create_app
        
        app = create_app()
        print("✅ Flask app created successfully")
        
        # Test routes
        with app.app_context():
            routes = list(app.url_map._rules)
            print(f"✅ Found {len(routes)} routes")
            
            # Check for key routes
            route_names = [str(rule) for rule in routes]
            key_routes = ['/', '/logo_processor', '/login', '/register']
            
            for route in key_routes:
                if any(route in r for r in route_names):
                    print(f"✅ Found route: {route}")
                else:
                    print(f"⚠️  Missing route: {route}")
        
        return True
        
    except Exception as e:
        print(f"❌ Flask app error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("🚀 Zyppts Structure Test")
    print("=" * 50)
    
    # Test structure
    if not test_structure():
        print("❌ Structure test failed")
        return
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed")
        return
    
    # Test Flask app
    if not test_flask_app():
        print("❌ Flask app test failed")
        return
    
    print("\n🎉 All tests passed!")
    print("💡 The app is ready to run with: python start_app.py")
    print("💡 Or manually: cd Backend && python run.py")

if __name__ == "__main__":
    main() 