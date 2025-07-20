#!/usr/bin/env python3
"""
Test script to simulate the actual web application flow for zip file download.
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

def test_web_application_flow():
    """Test the complete web application flow for zip file download."""
    
    print("=== WEB APPLICATION FLOW TEST ===\n")
    
    # Step 1: Test if Flask app is running
    print("1. Checking Flask application status...")
    try:
        response = requests.get('http://127.0.0.1:5003/', timeout=5)
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
    
    # Step 2: Create a test image file
    print("\n2. Creating test image file...")
    try:
        from PIL import Image
        
        # Create a simple test image
        img = Image.new('RGB', (200, 200), color='blue')
        test_image_path = '/tmp/test_logo_web.png'
        img.save(test_image_path)
        
        print(f"   ✓ Created test image: {test_image_path}")
        
    except Exception as e:
        print(f"   ✗ Error creating test image: {e}")
        return False
    
    # Step 3: Simulate form submission
    print("\n3. Simulating form submission...")
    try:
        # Prepare form data
        files = {'logo': open(test_image_path, 'rb')}
        data = {
            'effect_vector': 'on',  # Enable vector trace
            'session_id': 'test_session_123'
        }
        
        # Submit the form
        response = requests.post(
            'http://127.0.0.1:5003/logo_processor',
            files=files,
            data=data,
            timeout=30
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
                            print("\n4. Testing zip file download...")
                            zip_download_url = f"http://127.0.0.1:5003/download_zip/{upload_id}"
                            print(f"   Download URL: {zip_download_url}")
                            
                            download_response = requests.get(zip_download_url, timeout=10)
                            print(f"   Download response status: {download_response.status_code}")
                            print(f"   Download response headers: {dict(download_response.headers)}")
                            
                            if download_response.status_code == 200:
                                # Save the downloaded zip file
                                downloaded_zip_path = '/tmp/downloaded_test.zip'
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
    
    print("\n=== WEB APPLICATION FLOW TEST COMPLETE ===")
    return True

def test_download_endpoints_directly():
    """Test download endpoints directly to see if they work."""
    print("\n=== DIRECT DOWNLOAD ENDPOINT TEST ===\n")
    
    # Test the download endpoints with a fake upload ID
    test_upload_id = "fake_upload_123"
    
    # Test individual file download
    print("1. Testing individual file download endpoint...")
    try:
        download_url = f"http://127.0.0.1:5003/download/{test_upload_id}/test_file.zip"
        response = requests.get(download_url, timeout=5)
        print(f"   Individual download response: {response.status_code}")
        if response.status_code != 404:  # Should be 404 for fake upload
            print(f"   Unexpected response: {response.text[:200]}...")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test zip download endpoint
    print("\n2. Testing zip download endpoint...")
    try:
        zip_download_url = f"http://127.0.0.1:5003/download_zip/{test_upload_id}"
        response = requests.get(zip_download_url, timeout=5)
        print(f"   Zip download response: {response.status_code}")
        if response.status_code != 404:  # Should be 404 for fake upload
            print(f"   Unexpected response: {response.text[:200]}...")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n=== DIRECT DOWNLOAD ENDPOINT TEST COMPLETE ===")

if __name__ == "__main__":
    print("Starting web application flow test...")
    
    # Run the tests
    test_web_application_flow()
    test_download_endpoints_directly()
    
    print("\nAll tests complete.") 