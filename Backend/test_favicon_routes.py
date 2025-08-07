#!/usr/bin/env python3
"""
Test script to verify favicon routes are working correctly
"""

import os
import sys
import requests
from datetime import datetime

def test_favicon_routes():
    """Test all favicon routes"""
    
    # Test URL
    base_url = "http://localhost:5003"
    
    print("🔍 Testing Favicon Routes")
    print("=" * 50)
    
    # Test favicon routes
    favicon_routes = [
        ('/favicon.ico', 'favicon.ico'),
        ('/favicon.png', 'favicon.png'),
        ('/apple-touch-icon.png', 'apple-touch-icon.png'),
        ('/site.webmanifest', 'site.webmanifest'),
        ('/safari-pinned-tab.svg', 'safari-pinned-tab.svg'),
        ('/static/images/favicon/favicon-32x32.png', 'favicon-32x32.png (static)'),
        ('/static/images/favicon/favicon-16x16.png', 'favicon-16x16.png (static)'),
        ('/static/images/favicon/safari-pinned-tab.svg', 'safari-pinned-tab.svg (static)')
    ]
    
    for route, description in favicon_routes:
        try:
            print(f"\nTesting {description}...")
            response = requests.get(f"{base_url}{route}", timeout=5)
            print(f"   Status Code: {response.status_code}")
            
            if response.status_code == 200:
                print(f"   ✅ {description} loads successfully")
                print(f"   📏 Content Length: {len(response.content)} bytes")
                print(f"   📋 Content Type: {response.headers.get('Content-Type', 'Not specified')}")
                
                # Check cache headers
                cache_control = response.headers.get('Cache-Control', 'Not set')
                print(f"   🕒 Cache Control: {cache_control}")
                
            elif response.status_code == 404:
                print(f"   ❌ {description} not found (404)")
            else:
                print(f"   ⚠️  {description} returned status {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"   ❌ Cannot connect to server for {description}")
        except Exception as e:
            print(f"   ❌ Error testing {description}: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Favicon Test Summary:")
    print("   - All routes should return 200 status codes")
    print("   - Favicon files should have appropriate content types")
    print("   - Cache headers should be set for performance")
    print("\n🔧 If you see 404 errors:")
    print("   1. Check that favicon files exist in Frontend/static/images/favicon/")
    print("   2. Verify the Flask app is running on port 5003")
    print("   3. Check that static folder is properly configured")
    
    return True

if __name__ == "__main__":
    test_favicon_routes() 