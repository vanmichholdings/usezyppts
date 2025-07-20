#!/usr/bin/env python3

import sys
import os

# Add the Backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Backend'))

try:
    print("🔍 Testing imports...")
    
    # Test basic imports
    import numpy as np
    print("✅ NumPy imported successfully")
    
    import cv2
    print("✅ OpenCV imported successfully")
    
    from PIL import Image
    print("✅ Pillow imported successfully")
    
    import torch
    print("✅ PyTorch imported successfully")
    
    # Test Flask app creation
    print("🔍 Testing Flask app creation...")
    from Backend.__init__ import create_app
    
    print("✅ Flask app creation successful")
    
    # Create the app
    app = create_app()
    print("✅ Flask app created successfully")
    
    print("🎉 All tests passed! The app is ready to run.")
    print("💡 Run: cd Backend && python run.py")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're in the virtual environment: source venv/bin/activate")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("💡 Check the error message above for details") 