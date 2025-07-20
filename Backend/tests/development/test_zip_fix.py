#!/usr/bin/env python3
"""
Test script to verify zip file creation fix.
"""

import os
import sys
import tempfile
import shutil
import time
import requests
import json

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_zip_creation_fix():
    """Test that zip files are now properly created and accessible."""
    
    print("=== TESTING ZIP FILE CREATION FIX ===\n")
    
    # Test 1: Check if Flask app is running
    print("1. Testing Flask application connectivity...")
    try:
        response = requests.get('http://127.0.0.1:5003/', timeout=5)
        if response.status_code == 200:
            print("   ✓ Flask application is running")
        else:
            print(f"   ✗ Flask application returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("   ✗ Flask application is not running on port 5003")
        print("   Please start the Flask application first with: python run.py")
        return False
    except Exception as e:
        print(f"   ✗ Error connecting to Flask app: {e}")
        return False
    
    # Test 2: Test logo processor endpoint
    print("\n2. Testing logo processor endpoint...")
    try:
        # Create a test image
        from PIL import Image
        test_image_path = tempfile.mktemp(suffix='.png')
        img = Image.new('RGB', (100, 100), color='blue')
        img.save(test_image_path)
        
        # Prepare form data
        with open(test_image_path, 'rb') as f:
            files = {'logo': ('test_logo.png', f, 'image/png')}
            data = {
                'effect_vector': 'on',  # Enable vector trace
                'session_id': 'test_session_' + str(int(time.time()))
            }
            
            print("   Sending request to logo processor...")
            response = requests.post('http://127.0.0.1:5003/logo_processor', 
                                   files=files, data=data, timeout=30)
        
        # Clean up test image
        os.unlink(test_image_path)
        
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Logo processor returned success: {result.get('success', False)}")
            print(f"   ✓ Files generated: {result.get('files', [])}")
            print(f"   ✓ Upload ID: {result.get('upload_id', 'N/A')}")
            
            # Check if zip file is in the list
            files_list = result.get('files', [])
            zip_files = [f for f in files_list if f.endswith('.zip')]
            if zip_files:
                print(f"   ✓ Zip file found: {zip_files[0]}")
                
                # Test 3: Try to download the zip file
                print("\n3. Testing zip file download...")
                upload_id = result.get('upload_id')
                zip_filename = zip_files[0]
                
                download_url = f"http://127.0.0.1:5003/download/{upload_id}/{zip_filename}"
                print(f"   Downloading from: {download_url}")
                
                download_response = requests.get(download_url, timeout=10)
                if download_response.status_code == 200:
                    print(f"   ✓ Zip file downloaded successfully (size: {len(download_response.content)} bytes)")
                    
                    # Save the zip file temporarily to verify it's valid
                    temp_zip_path = tempfile.mktemp(suffix='.zip')
                    with open(temp_zip_path, 'wb') as f:
                        f.write(download_response.content)
                    
                    # Verify zip file is valid
                    import zipfile
                    try:
                        with zipfile.ZipFile(temp_zip_path, 'r') as zipf:
                            file_list = zipf.namelist()
                            print(f"   ✓ Zip file is valid and contains {len(file_list)} files: {file_list}")
                    except Exception as e:
                        print(f"   ✗ Zip file is corrupted: {e}")
                        return False
                    finally:
                        os.unlink(temp_zip_path)
                    
                    return True
                else:
                    print(f"   ✗ Failed to download zip file: {download_response.status_code}")
                    return False
            else:
                print("   ✗ No zip file found in the response")
                return False
        else:
            print(f"   ✗ Logo processor failed: {response.status_code}")
            print(f"   Response headers: {dict(response.headers)}")
            print(f"   Response content (first 500 chars): {response.text[:500]}")
            return False
            
    except Exception as e:
        print(f"   ✗ Error testing logo processor: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_zip_creation_fix()
    if success:
        print("\n=== TEST PASSED ===")
        print("✓ Zip file creation and download is working correctly!")
    else:
        print("\n=== TEST FAILED ===")
        print("✗ There are still issues with zip file creation.")
    
    print("\nTest complete.") 