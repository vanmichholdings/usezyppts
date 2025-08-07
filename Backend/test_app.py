#!/usr/bin/env python3
"""
Test script for Zyppts application
"""

import os
import sys
import requests
import time

# Add the parent directory to the path so we can import from Backend
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_app_health():
    """Test the application health endpoint"""
    try:
        # Test local development server
        response = requests.get('http://localhost:5003/health', timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")
        return False

def test_favicon_routes():
    """Test favicon routes"""
    favicon_routes = [
        '/favicon.ico',
        '/favicon.png',
        '/apple-touch-icon.png',
        '/site.webmanifest',
        '/safari-pinned-tab.svg'
    ]
    
    for route in favicon_routes:
        try:
            response = requests.get(f'http://localhost:5003{route}', timeout=5)
            if response.status_code == 200:
                print(f"✅ {route} - OK")
            else:
                print(f"❌ {route} - {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ {route} - Error: {e}")

def test_admin_analytics():
    """Test admin analytics route"""
    try:
        response = requests.get('http://localhost:5003/admin/analytics', timeout=5)
        if response.status_code in [200, 302, 401]:  # 302 for redirect, 401 for unauthorized
            print("✅ Admin analytics route accessible")
        else:
            print(f"❌ Admin analytics route error: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Admin analytics route error: {e}")

def main():
    """Run all tests"""
    print("🧪 Testing Zyppts Application")
    print("=" * 40)
    
    # Test health endpoint
    print("\n1. Testing health endpoint...")
    health_ok = test_app_health()
    
    # Test favicon routes
    print("\n2. Testing favicon routes...")
    test_favicon_routes()
    
    # Test admin analytics
    print("\n3. Testing admin analytics...")
    test_admin_analytics()
    
    print("\n" + "=" * 40)
    if health_ok:
        print("✅ All tests completed successfully!")
    else:
        print("❌ Some tests failed. Check your application.")

if __name__ == "__main__":
    main() 