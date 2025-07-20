#!/usr/bin/env python3
"""
Comprehensive debug script for zip file export issue.
"""

import os
import sys
import tempfile
import shutil
import zipfile
import requests
import json
import time
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_zip_creation_process():
    """Test the entire zip creation process step by step."""
    
    print("=== ZIP FILE EXPORT DEBUG TEST ===\n")
    
    # Step 1: Test basic zip creation
    print("1. Testing basic zip file creation...")
    temp_dir = tempfile.mkdtemp()
    test_files = []
    
    try:
        # Create test files
        for i in range(3):
            filename = f"test_file_{i}.txt"
            file_path = os.path.join(temp_dir, filename)
            with open(file_path, 'w') as f:
                f.write(f"Test content {i}")
            test_files.append(filename)
        
        # Create zip file
        zip_filename = "test_processed.zip"
        zip_path = os.path.join(temp_dir, zip_filename)
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename in test_files:
                file_path = os.path.join(temp_dir, filename)
                zipf.write(file_path, filename)
        
        # Verify zip file
        if os.path.exists(zip_path):
            zip_size = os.path.getsize(zip_path)
            print(f"   ✓ Zip file created successfully: {zip_path} (size: {zip_size} bytes)")
            
            # Test zip file contents
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                file_list = zipf.namelist()
                print(f"   ✓ Zip contains {len(file_list)} files: {file_list}")
        else:
            print("   ✗ Zip file was not created")
            return False
            
    except Exception as e:
        print(f"   ✗ Error creating zip file: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir)
    
    # Step 2: Test logo processor zip creation
    print("\n2. Testing logo processor zip creation...")
    try:
        from zyppts.utils.logo_processor import LogoProcessor
        
        # Create test directories
        test_upload_id = "test_upload_debug"
        base_dir = tempfile.mkdtemp()
        upload_dir = os.path.join(base_dir, test_upload_id)
        
        dirs = {
            'upload': os.path.join(upload_dir, 'uploads'),
            'output': os.path.join(upload_dir, 'outputs'),
            'cache': os.path.join(upload_dir, 'cache'),
            'temp': os.path.join(upload_dir, 'temp')
        }
        
        # Create directories
        for path in dirs.values():
            os.makedirs(path, exist_ok=True)
        
        # Create a test image
        from PIL import Image
        test_image_path = os.path.join(dirs['upload'], 'test_logo.png')
        img = Image.new('RGB', (100, 100), color='red')
        img.save(test_image_path)
        
        # Initialize processor
        processor = LogoProcessor(
            cache_folder=dirs['cache'],
            upload_folder=dirs['upload'],
            output_folder=dirs['output'],
            temp_folder=dirs['temp']
        )
        
        # Test vector trace processing
        options = {'vector_trace': True}
        result = processor.process_logo(file_path=test_image_path, options=options)
        
        print(f"   Processing result: {result.get('success', False)}")
        print(f"   Outputs: {list(result.get('outputs', {}).keys())}")
        
        # Check if vector trace output exists
        vector_output = result.get('outputs', {}).get('vector_trace', {})
        if vector_output and vector_output.get('status') == 'success':
            output_paths = vector_output.get('output_paths', {})
            print(f"   Vector trace output paths: {output_paths}")
            
            # Check if files exist
            for format_key, format_path in output_paths.items():
                if format_path and os.path.exists(format_path):
                    print(f"   ✓ {format_key}: {format_path} (exists)")
                else:
                    print(f"   ✗ {format_key}: {format_path} (missing)")
        else:
            print(f"   ✗ Vector trace processing failed: {vector_output}")
        
        # Test zip creation with actual files
        output_files = []
        original_name = "test_logo"
        
        # Add vector trace files to output
        if vector_output and vector_output.get('status') == 'success':
            output_paths = vector_output.get('output_paths', {})
            for format_key, format_path in output_paths.items():
                if format_path and os.path.exists(format_path):
                    dest_filename = os.path.basename(format_path)
                    dest_path = os.path.join(dirs['output'], dest_filename)
                    shutil.copy2(format_path, dest_path)
                    output_files.append(dest_filename)
                    print(f"   ✓ Copied {format_key} to output: {dest_filename}")
        
        # Create zip file
        zip_filename = f"{original_name}_processed.zip"
        zip_path = os.path.join(dirs['output'], zip_filename)
        
        print(f"   Creating zip file: {zip_path}")
        print(f"   Files to include: {output_files}")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename in output_files:
                file_path = os.path.join(dirs['output'], filename)
                if os.path.exists(file_path):
                    zipf.write(file_path, filename)
                    print(f"   ✓ Added {filename} to zip")
                else:
                    print(f"   ✗ File not found: {file_path}")
        
        # Verify final zip
        if os.path.exists(zip_path):
            zip_size = os.path.getsize(zip_path)
            print(f"   ✓ Final zip file created: {zip_path} (size: {zip_size} bytes)")
            
            # Test zip contents
            with zipfile.ZipFile(zip_path, 'r') as zipf:
                file_list = zipf.namelist()
                print(f"   ✓ Zip contains: {file_list}")
        else:
            print(f"   ✗ Final zip file not found: {zip_path}")
        
    except Exception as e:
        print(f"   ✗ Error in logo processor test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'base_dir' in locals():
            shutil.rmtree(base_dir)
    
    # Step 3: Test web application endpoints
    print("\n3. Testing web application endpoints...")
    try:
        # Test if Flask app is running
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
    
    # Step 4: Test download endpoints
    print("\n4. Testing download endpoints...")
    try:
        # Test download endpoint structure
        download_url = "http://127.0.0.1:5003/download/test_upload/test_file.zip"
        response = requests.get(download_url, timeout=5)
        print(f"   Download endpoint test: {response.status_code}")
        
        # Test zip download endpoint
        zip_download_url = "http://127.0.0.1:5003/download_zip/test_upload"
        response = requests.get(zip_download_url, timeout=5)
        print(f"   Zip download endpoint test: {response.status_code}")
        
    except Exception as e:
        print(f"   ✗ Error testing download endpoints: {e}")
    
    print("\n=== DEBUG TEST COMPLETE ===")
    return True

def test_routes_logic():
    """Test the routes.py logic for zip file creation."""
    print("\n=== TESTING ROUTES.PY LOGIC ===\n")
    
    # Import the routes module
    try:
        from zyppts.routes import ensure_upload_dirs, cleanup_dirs
        print("✓ Successfully imported routes module")
    except Exception as e:
        print(f"✗ Error importing routes module: {e}")
        return False
    
    # Test directory creation
    try:
        test_upload_id = "test_routes_debug"
        temp_dir = tempfile.mkdtemp()
        
        # Mock current_app.config
        class MockConfig:
            UPLOAD_FOLDER = temp_dir
        
        class MockApp:
            config = MockConfig()
        
        # Test ensure_upload_dirs
        dirs = ensure_upload_dirs(test_upload_id)
        print(f"✓ Created directories: {list(dirs.keys())}")
        
        # Verify directories exist
        for name, path in dirs.items():
            if os.path.exists(path):
                print(f"  ✓ {name}: {path}")
            else:
                print(f"  ✗ {name}: {path} (missing)")
        
        # Test cleanup
        cleanup_dirs(dirs)
        print("✓ Cleanup function executed")
        
    except Exception as e:
        print(f"✗ Error testing routes logic: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
    
    return True

if __name__ == "__main__":
    print("Starting comprehensive zip export debug...")
    
    # Run all tests
    test_zip_creation_process()
    test_routes_logic()
    
    print("\nDebug complete. Check the output above for issues.") 