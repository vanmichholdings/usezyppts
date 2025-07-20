#!/usr/bin/env python3
"""
Test script to simulate authenticated user flow for zip file download.
"""

import os
import sys
import tempfile
import shutil
import requests
import json
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_authenticated_flow():
    """Test the complete authenticated flow for zip file download."""
    
    print("=== AUTHENTICATED FLOW TEST ===\n")
    
    # Create a session to maintain cookies
    session = requests.Session()
    
    # Step 1: Test if Flask app is running
    print("1. Checking Flask application status...")
    try:
        response = session.get('http://127.0.0.1:5003/', timeout=5)
        if response.status_code == 200:
            print("   ✓ Flask application is running")
        else:
            print(f"   ✗ Flask application returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("   ✗ Flask application is not running on port 5003")
        return False
    except Exception as e:
        print(f"   ✗ Error connecting to Flask app: {e}")
        return False
    
    # Step 2: Create a test user account
    print("\n2. Creating test user account...")
    try:
        # Register a test user
        register_data = {
            'username': 'testuser_zip',
            'email': 'testuser_zip@example.com',
            'password': 'testpassword123'
        }
        
        response = session.post(
            'http://127.0.0.1:5003/register',
            data=register_data,
            timeout=10
        )
        
        print(f"   Register response status: {response.status_code}")
        
        # If registration fails, try to login (user might already exist)
        if response.status_code != 200:
            print("   Registration failed, trying login...")
            login_data = {
                'email': 'testuser_zip@example.com',
                'password': 'testpassword123'
            }
            
            response = session.post(
                'http://127.0.0.1:5003/login',
                data=login_data,
                timeout=10
            )
            
            print(f"   Login response status: {response.status_code}")
        
        # Check if we're now authenticated
        response = session.get('http://127.0.0.1:5003/logo_processor', timeout=5)
        if 'Login Required' in response.text:
            print("   ✗ Still not authenticated")
            return False
        else:
            print("   ✓ Successfully authenticated")
            
    except Exception as e:
        print(f"   ✗ Error with authentication: {e}")
        return False
    
    # Step 3: Create a test image file
    print("\n3. Creating test image file...")
    try:
        from PIL import Image
        
        # Create a simple test image
        img = Image.new('RGB', (200, 200), color='green')
        test_image_path = '/tmp/test_logo_auth.png'
        img.save(test_image_path)
        
        print(f"   ✓ Created test image: {test_image_path}")
        
    except Exception as e:
        print(f"   ✗ Error creating test image: {e}")
        return False
    
    # Step 4: Submit the logo processor form
    print("\n4. Submitting logo processor form...")
    try:
        # Prepare form data
        with open(test_image_path, 'rb') as f:
            files = {'logo': ('test_logo.png', f, 'image/png')}
            data = {
                'effect_vector': 'on',  # Enable vector trace
                'session_id': 'test_session_auth_123'
            }
            
            # Submit the form
            response = session.post(
                'http://127.0.0.1:5003/logo_processor',
                files=files,
                data=data,
                timeout=60  # Longer timeout for processing
            )
        
        print(f"   Response status: {response.status_code}")
        print(f"   Response headers: {dict(response.headers)}")
        
        # Check if response is JSON
        if 'application/json' in response.headers.get('content-type', ''):
            try:
                json_response = response.json()
                print(f"   JSON response: {json.dumps(json_response, indent=2)}")
                
                # Check if processing was successful
                if json_response.get('success'):
                    print("   ✓ Processing was successful")
                    
                    # Check for upload_id and files
                    upload_id = json_response.get('upload_id')
                    files_list = json_response.get('files', [])
                    
                    if upload_id:
                        print(f"   ✓ Upload ID: {upload_id}")
                    else:
                        print("   ✗ No upload ID in response")
                    
                    if files_list:
                        print(f"   ✓ Files generated: {files_list}")
                        
                        # Check for zip file
                        zip_files = [f for f in files_list if f.endswith('.zip')]
                        if zip_files:
                            print(f"   ✓ Zip file found: {zip_files[0]}")
                            
                            # Test zip download
                            print("\n5. Testing zip file download...")
                            zip_download_url = f"http://127.0.0.1:5003/download_zip/{upload_id}"
                            print(f"   Download URL: {zip_download_url}")
                            
                            download_response = session.get(zip_download_url, timeout=10)
                            print(f"   Download response status: {download_response.status_code}")
                            print(f"   Download response headers: {dict(download_response.headers)}")
                            
                            if download_response.status_code == 200:
                                # Save the downloaded zip file
                                downloaded_zip_path = '/tmp/downloaded_auth_test.zip'
                                with open(downloaded_zip_path, 'wb') as f:
                                    f.write(download_response.content)
                                
                                zip_size = os.path.getsize(downloaded_zip_path)
                                print(f"   ✓ Zip file downloaded successfully: {downloaded_zip_path} (size: {zip_size} bytes)")
                                
                                # Test zip file contents
                                import zipfile
                                with zipfile.ZipFile(downloaded_zip_path, 'r') as zipf:
                                    file_list = zipf.namelist()
                                    print(f"   ✓ Downloaded zip contains: {file_list}")
                                
                                # Clean up
                                os.remove(downloaded_zip_path)
                                
                            else:
                                print(f"   ✗ Download failed with status {download_response.status_code}")
                                if download_response.text:
                                    print(f"   Error response: {download_response.text}")
                        else:
                            print("   ✗ No zip file in generated files list")
                    else:
                        print("   ✗ No files generated")
                else:
                    print(f"   ✗ Processing failed: {json_response.get('error', 'Unknown error')}")
                    
            except json.JSONDecodeError as e:
                print(f"   ✗ Invalid JSON response: {e}")
                print(f"   Response text: {response.text[:500]}...")
        else:
            print(f"   ✗ Response is not JSON: {response.text[:500]}...")
            
    except Exception as e:
        print(f"   ✗ Error during form submission: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up test image
        if os.path.exists(test_image_path):
            os.remove(test_image_path)
    
    print("\n=== AUTHENTICATED FLOW TEST COMPLETE ===")
    return True

if __name__ == "__main__":
    print("Starting authenticated flow test...")
    
    # Run the test
    test_authenticated_flow()
    
    print("\nTest complete.") 