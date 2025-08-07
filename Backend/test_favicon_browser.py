#!/usr/bin/env python3
"""
Quick test to check favicon accessibility
"""

import requests
import os

def test_favicon_accessibility():
    """Test if favicon routes are accessible"""
    
    base_url = "http://localhost:5003"
    
    print("🔍 Testing Favicon Browser Accessibility")
    print("=" * 50)
    
    # Test the main favicon routes
    test_routes = [
        ('/favicon.ico', 'Main favicon'),
        ('/favicon.png', 'PNG favicon'),
        ('/apple-touch-icon.png', 'Apple touch icon'),
        ('/site.webmanifest', 'Web manifest'),
        ('/safari-pinned-tab.svg', 'Safari pinned tab')
    ]
    
    for route, description in test_routes:
        try:
            print(f"\nTesting {description}...")
            response = requests.get(f"{base_url}{route}", timeout=5)
            
            if response.status_code == 200:
                print(f"   ✅ {description} accessible")
                print(f"   📏 Size: {len(response.content)} bytes")
                print(f"   📋 Type: {response.headers.get('Content-Type', 'Unknown')}")
            else:
                print(f"   ❌ {description} returned {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Error accessing {description}: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Browser Favicon Test:")
    print("1. Open your browser to http://localhost:5003")
    print("2. Check the browser tab - you should see the Zyppts favicon")
    print("3. If not visible, try:")
    print("   - Hard refresh (Ctrl+F5 or Cmd+Shift+R)")
    print("   - Clear browser cache")
    print("   - Check browser developer tools (F12) > Network tab")
    print("   - Look for favicon requests and their status codes")
    
    return True

if __name__ == "__main__":
    test_favicon_accessibility() 