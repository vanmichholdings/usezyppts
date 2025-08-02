#!/usr/bin/env python3
"""
Simple test for vector tracing to isolate the boolean array issue
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from utils.logo_processor import LogoProcessor

def test_vector_trace():
    """Test vector tracing with a simple image"""
    print("🧪 Testing vector tracing...")
    
    # Test image path
    test_image = "tests/test_results_mercyevol/original_mercyevol.png"
    
    if not os.path.exists(test_image):
        print(f"❌ Test image not found: {test_image}")
        return False
    
    try:
        # Create processor
        processor = LogoProcessor()
        
        # Test vector tracing
        result = processor.generate_vector_trace(test_image)
        
        print(f"✅ Vector tracing result: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vector_trace()
    sys.exit(0 if success else 1) 