#!/usr/bin/env python3

import requests
import os
import re
import tempfile
from PIL import Image

# Configuration
BASE_URL = 'http://localhost:5003'
LOGIN_URL = f'{BASE_URL}/login'
TEST_EMAIL = 'testuser@example.com'
TEST_PASSWORD = 'password123'

# Test image path (MercyEvol logo in uploads)
SAMPLE_IMAGE = 'Backend/uploads/2d2280ee-52db-4d27-aff8-db8c8cdcedd9/uploads/mercyevol.png'

# Results directory
RESULTS_DIR = 'Backend/tests/test_results_mercyevol'
HTML_REPORT = f'{RESULTS_DIR}/mercyevol_logo_variation_comparison.html'

# Image resizing for faster processing
MAX_WIDTH = 800
MAX_HEIGHT = 600

# Test only distressed and vector (plus transparent as original)
VARIATIONS_TO_TEST = [
    'transparent',  # Use as "original" 
    'distressed',
    'vector'
]

def get_csrf_token(html):
    """Extract CSRF token from HTML"""
    match = re.search(r'name=["\']csrf_token["\'] value=["\']([^"\']+)["\']', html)
    return match.group(1) if match else None

def login(session):
    """Login to get authentication"""
    print('🔑 Logging in...')
    resp = session.get(LOGIN_URL)
    if resp.status_code != 200:
        raise Exception('Failed to load login page')
    
    csrf_token = get_csrf_token(resp.text)
    data = {
        'email': TEST_EMAIL,
        'password': TEST_PASSWORD
    }
    if csrf_token:
        data['csrf_token'] = csrf_token
    
    resp = session.post(LOGIN_URL, data=data, allow_redirects=True)
    if 'Invalid email or password' in resp.text or resp.url.endswith('/login'):
        raise Exception('Login failed: check test user credentials or create the user with a studio plan.')
    
    print('✅ Logged in as test user')

def resize_image_if_needed(image_path, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    """Resize image if it's larger than max dimensions, maintaining aspect ratio"""
    img = Image.open(image_path)
    original_size = img.size
    
    if img.width <= max_width and img.height <= max_height:
        print(f'Image size {original_size} is within limits, no resize needed')
        return image_path
    
    ratio = min(max_width / img.width, max_height / img.height)
    new_size = (int(img.width * ratio), int(img.height * ratio))
    
    print(f'Resizing image from {original_size} to {new_size} for faster processing')
    
    resized_img = img.resize(new_size, Image.Resampling.LANCZOS)
    resized_path = image_path.replace('.png', '_resized.png').replace('.jpg', '_resized.jpg').replace('.jpeg', '_resized.jpeg')
    resized_img.save(resized_path)
    
    return resized_path

def test_variation(session, variation_name, image_path):
    """Test a single logo variation"""
    url = f'{BASE_URL}/preview/{variation_name}'
    
    try:
        with open(image_path, 'rb') as f:
            files = {'file': f}
            response = session.post(url, files=files, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            if 'preview' in data:
                return True, data['preview']
            else:
                return False, f"No preview in response: {data}"
        else:
            return False, f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "Request timeout (60s)"
    except Exception as e:
        return False, str(e)

def save_image_from_base64(base64_data, save_path):
    """Save base64 image data to file"""
    try:
        # Extract base64 data (remove data:image/png;base64, prefix)
        if base64_data.startswith('data:'):
            base64_data = base64_data.split(',')[1]
        
        import base64
        img_data = base64.b64decode(base64_data)
        with open(save_path, 'wb') as f:
            f.write(img_data)
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False

def generate_html_comparison(results, sample_image_path):
    """Generate HTML comparison page"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Copy original image
    original_path = os.path.join(RESULTS_DIR, 'original.png')
    if os.path.exists(sample_image_path):
        img = Image.open(sample_image_path)
        img.save(original_path, 'PNG')
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Logo Variations Comparison - MercyEvol</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
            h1 {{ color: #333; text-align: center; margin-bottom: 30px; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .variation {{ 
                background: white; 
                border: 1px solid #ddd; 
                border-radius: 8px; 
                padding: 15px; 
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .variation h3 {{ 
                margin: 0 0 15px 0; 
                color: #555; 
                font-size: 18px;
                text-transform: capitalize;
            }}
            .variation img {{ 
                max-width: 100%; 
                height: 200px; 
                object-fit: contain; 
                border: 1px solid #eee;
                background: repeating-conic-gradient(#f0f0f0 0% 25%, transparent 0% 50%) 50% / 20px 20px;
            }}
            .status {{ margin-top: 10px; font-weight: bold; }}
            .success {{ color: #28a745; }}
            .error {{ color: #dc3545; }}
            .summary {{ 
                background: #e9ecef; 
                padding: 15px; 
                border-radius: 8px; 
                margin-bottom: 20px;
                text-align: center;
            }}
            .original {{ border: 3px solid #007bff; }}
            .distressed {{ border: 3px solid #fd7e14; }}
            .vector {{ border: 3px solid #6f42c1; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎨 Logo Variations Comparison</h1>
            <div class="summary">
                <strong>Test Results:</strong> {sum(1 for r in results.values() if r['success'])}/{len(results)} variations successful
                <br><strong>Image:</strong> MercyEvol Logo
                <br><strong>Focus:</strong> Distressed & Vector Quality Assessment
            </div>
            <div class="grid">
    """
    
    # Map variation names to display info
    variation_info = {
        'transparent': {
            'title': 'Original',
            'class': 'original',
            'description': 'Source logo with transparency'
        },
        'distressed': {
            'title': 'Distressed', 
            'class': 'distressed',
            'description': 'Grunge texture applied to logo only'
        },
        'vector': {
            'title': 'Vector',
            'class': 'vector', 
            'description': 'Vectorized with separated paths'
        }
    }
    
    # Show results in specific order
    for variation in ['transparent', 'distressed', 'vector']:
        if variation not in results:
            continue
            
        result = results[variation]
        info = variation_info[variation]
        
        if result['success']:
            image_filename = f"{variation}.png"
            status_class = "success"
            status_text = "✅ Success"
            
            if variation == 'transparent':
                # Use original image for transparent
                img_src = "original.png"
            else:
                img_src = image_filename
        else:
            img_src = ""
            status_class = "error"
            status_text = f"❌ {result.get('error', 'Failed')}"
        
        html_content += f"""
                <div class="variation {info['class']}">
                    <h3>{info['title']}</h3>
                    {f'<img src="{img_src}" alt="{info["title"]}">' if img_src else '<div style="height: 200px; background: #f8f9fa; border: 1px solid #dee2e6; display: flex; align-items: center; justify-content: center; color: #6c757d;">No Image</div>'}
                    <div class="status {status_class}">{status_text}</div>
                    <p style="font-size: 12px; color: #666; margin: 10px 0 0 0;">{info['description']}</p>
                </div>
        """
    
    html_content += """
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(HTML_REPORT, 'w') as f:
        f.write(html_content)

def main():
    print(f"🧪 Testing focused logo variations with MercyEvol logo...")
    
    if not os.path.exists(SAMPLE_IMAGE):
        print(f"❌ Sample image not found: {SAMPLE_IMAGE}")
        return
    
    # Create session
    session = requests.Session()
    
    try:
        # Login
        login(session)
        
        print(f"\n📏 Checking image size for optimization...")
        # Resize image if needed
        working_image = resize_image_if_needed(SAMPLE_IMAGE)
        
        print(f"\n🧪 Testing {len(VARIATIONS_TO_TEST)} focused logo variations...")
        
        # Test variations
        results = {}
        for variation in VARIATIONS_TO_TEST:
            print(f"\nTesting {variation}...", end=' ')
            success, result = test_variation(session, variation, working_image)
            
            if success:
                print("✅ Success")
                if variation == 'distressed':
                    print("   📄 Applied grunge texture to logo artwork only")
                elif variation == 'vector':
                    print("   🎯 Generated separated vector paths")
                
                # Save the result image
                save_path = os.path.join(RESULTS_DIR, f"{variation}.png")
                os.makedirs(RESULTS_DIR, exist_ok=True)
                if save_image_from_base64(result, save_path):
                    results[variation] = {'success': True, 'path': save_path}
                else:
                    results[variation] = {'success': False, 'error': 'Failed to save image'}
            else:
                print(f"❌ {result}")
                results[variation] = {'success': False, 'error': result}
        
        # Clean up resized file if it was created
        if working_image != SAMPLE_IMAGE and os.path.exists(working_image):
            os.remove(working_image)
            print(f"\n🧹 Cleaned up temporary resized file")
        
        # Generate HTML comparison
        generate_html_comparison(results, SAMPLE_IMAGE)
        
        # Summary
        successful = sum(1 for r in results.values() if r['success'])
        print(f"\n📊 Results: {successful}/{len(VARIATIONS_TO_TEST)} variations successful")
        print(f"📄 Focused comparison HTML generated: {HTML_REPORT}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == '__main__':
    main() 