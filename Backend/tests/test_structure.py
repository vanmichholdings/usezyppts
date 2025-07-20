#!/usr/bin/env python3

import sys
import os

# Add the Backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Backend'))

try:
    print("🔍 Testing new structure...")
    
    # Test config paths
    from Backend.config import Config
    print(f"✅ Config loaded")
    print(f"   Templates: {Config.TEMPLATES_FOLDER}")
    print(f"   Static: {Config.STATIC_FOLDER}")
    print(f"   Upload: {Config.UPLOAD_FOLDER}")
    
    # Check if directories exist
    if os.path.exists(Config.TEMPLATES_FOLDER):
        print(f"✅ Templates directory exists: {Config.TEMPLATES_FOLDER}")
    else:
        print(f"❌ Templates directory missing: {Config.TEMPLATES_FOLDER}")
    
    if os.path.exists(Config.STATIC_FOLDER):
        print(f"✅ Static directory exists: {Config.STATIC_FOLDER}")
    else:
        print(f"❌ Static directory missing: {Config.STATIC_FOLDER}")
    
    # Test Flask app creation
    print("🔍 Testing Flask app creation...")
    from Backend.__init__ import create_app
    
    app = create_app()
    print("✅ Flask app created successfully")
    
    # Test routes registration
    print("🔍 Testing routes...")
    with app.app_context():
        print(f"✅ App context created")
        print(f"   Blueprints: {list(app.blueprints.keys())}")
        print(f"   Routes: {len(app.url_map._rules)}")
    
    print("🎉 Structure test passed!")
    print("💡 The app should now work with the new Frontend/Backend structure")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc() 