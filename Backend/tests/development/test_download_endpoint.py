#!/usr/bin/env python3
"""
Test script to verify download endpoint functionality.
"""

import os
import sys
import requests
import tempfile
import zipfile

def test_download_endpoint():
    """Test the download endpoint functionality."""
    
    print("=== TESTING DOWNLOAD ENDPOINT ===\n")
    
    # Test with a known upload ID from the logs
    upload_id = "b60940f9-1dc4-4e44-9a37-87b10a20a616"
    zip_filename = "evolelon_processed.zip"
    
    print(f"1. Testing download endpoint for upload: {upload_id}")
    print(f"   Zip file: {zip_filename}")
    
    # First, let's check if the file exists on disk
    file_path = f"uploads/{upload_id}/outputs/{zip_filename}"
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        print(f"   ✓ File exists on disk: {file_path} (size: {file_size} bytes)")
    else:
        print(f"   ✗ File not found on disk: {file_path}")
        return False
    
    # Test the download endpoint
    download_url = f"http://127.0.0.1:5003/download/{upload_id}/{zip_filename}"
    print(f"\n2. Testing download URL: {download_url}")
    
    try:
        # Make the request
        response = requests.get(download_url, timeout=10)
        
        print(f"   Response status: {response.status_code}")
        print(f"   Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            content_length = len(response.content)
            print(f"   ✓ Download successful (content length: {content_length} bytes)")
            
            # Check if the content looks like a zip file
            if response.content.startswith(b'PK'):
                print("   ✓ Content appears to be a valid ZIP file (starts with PK)")
                
                # Save and verify the zip file
                temp_zip_path = tempfile.mktemp(suffix='.zip')
                with open(temp_zip_path, 'wb') as f:
                    f.write(response.content)
                
                try:
                    with zipfile.ZipFile(temp_zip_path, 'r') as zipf:
                        file_list = zipf.namelist()
                        print(f"   ✓ ZIP file is valid and contains {len(file_list)} files:")
                        for file in file_list:
                            print(f"      - {file}")
                    
                    # Clean up
                    os.unlink(temp_zip_path)
                    return True
                    
                except Exception as e:
                    print(f"   ✗ ZIP file verification failed: {e}")
                    if os.path.exists(temp_zip_path):
                        os.unlink(temp_zip_path)
                    return False
            else:
                print(f"   ✗ Content doesn't appear to be a ZIP file")
                print(f"   First 100 bytes: {response.content[:100]}")
                return False
                
        elif response.status_code == 302:
            print("   ✗ Got redirect response (likely to login page)")
            print(f"   Location header: {response.headers.get('Location', 'None')}")
            return False
        else:
            print(f"   ✗ Unexpected status code: {response.status_code}")
            print(f"   Response content: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"   ✗ Error testing download: {e}")
        return False

if __name__ == "__main__":
    success = test_download_endpoint()
    if success:
        print("\n=== TEST PASSED ===")
        print("✓ Download endpoint is working correctly!")
    else:
        print("\n=== TEST FAILED ===")
        print("✗ There are issues with the download endpoint.")
    
    print("\nTest complete.") 