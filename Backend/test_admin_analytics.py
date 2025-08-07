#!/usr/bin/env python3
"""
Test script to verify admin analytics route functionality
"""

import os
import sys
import requests
from datetime import datetime

def test_admin_analytics():
    """Test the admin analytics endpoint"""
    
    # Test URL
    base_url = "http://localhost:5003"
    analytics_url = f"{base_url}/admin/analytics"
    
    print("🔍 Testing Admin Analytics Route")
    print("=" * 50)
    
    try:
        # Test 1: Check if server is running
        print("1. Testing server connectivity...")
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            print("   ✅ Server is running")
        else:
            print(f"   ❌ Server returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to server. Is it running?")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test 2: Check analytics endpoint
    print("\n2. Testing analytics endpoint...")
    try:
        response = requests.get(analytics_url, timeout=10)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Analytics page loads successfully")
            print("   📊 Page content length:", len(response.text))
            
            # Check for key content
            if "Analytics Dashboard" in response.text:
                print("   ✅ Analytics dashboard content found")
            else:
                print("   ⚠️  Analytics dashboard content not found")
                
        elif response.status_code == 302:
            print("   🔄 Redirect detected (likely to login)")
            print("   ℹ️  This is expected if not logged in as admin")
            
        elif response.status_code == 403:
            print("   🚫 Access forbidden (not admin user)")
            print("   ℹ️  This is expected if not logged in as admin")
            
        elif response.status_code == 404:
            print("   ❌ Analytics route not found")
            print("   🔧 Check if admin blueprint is registered")
            
        else:
            print(f"   ❌ Unexpected status code: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error testing analytics: {e}")
        return False
    
    # Test 3: Check admin dashboard
    print("\n3. Testing admin dashboard...")
    try:
        dashboard_url = f"{base_url}/admin/"
        response = requests.get(dashboard_url, timeout=10)
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ Admin dashboard loads successfully")
        elif response.status_code == 302:
            print("   🔄 Redirect to login (expected)")
        else:
            print(f"   ⚠️  Dashboard status: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Error testing dashboard: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Test Summary:")
    print("   - If you see 302 redirects, you need to log in as an admin user")
    print("   - If you see 404 errors, there's a routing issue")
    print("   - If you see 200 responses, the analytics page is working!")
    print("\n🔧 To access admin analytics:")
    print("   1. Log in with an admin account")
    print("   2. Navigate to: http://localhost:5003/admin/analytics")
    print("   3. Or visit: http://localhost:5003/admin/")
    
    return True

if __name__ == "__main__":
    test_admin_analytics() 