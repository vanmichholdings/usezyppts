#!/usr/bin/env python3
"""
Focused Vector Trace Comparison Test
Compares original logo with new compound path vector trace method
"""

import os
import sys
import requests
import time
from pathlib import Path

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def test_vector_comparison():
    """Test original vs new compound path vector trace"""
    print("🎯 Testing Original vs New Compound Path Vector Trace...")
    
    # Setup
    base_url = "http://localhost:5003"
    test_dir = Path(__file__).parent / "test_results_vector_comparison"
    test_dir.mkdir(exist_ok=True)
    
    # Test image
    test_image_path = Path(__file__).parent / "test_results_mercyevol" / "original_mercyevol.png"
    
    if not test_image_path.exists():
        print(f"❌ Test image not found: {test_image_path}")
        return False
    
    # Login
    print("🔑 Logging in...")
    session = requests.Session()
    
    login_response = session.get(f"{base_url}/login")
    if login_response.status_code != 200:
        print("❌ Failed to access login page")
        return False
    
    login_data = {
        'username': 'testuser',
        'password': 'testpass123'
    }
    
    login_response = session.post(f"{base_url}/login", data=login_data, allow_redirects=False)
    if login_response.status_code != 302:
        print("❌ Login failed")
        return False
    
    print("✅ Logged in successfully")
    
    # Test original and vector trace
    results = {}
    
    # 1. Get original (transparent version)
    print("\n📸 Testing original logo...")
    with open(test_image_path, 'rb') as f:
        files = {'file': ('mercyevol.png', f, 'image/png')}
        response = session.post(f"{base_url}/preview/transparent", files=files)
        
        if response.status_code == 200:
            original_path = test_dir / "original.png"
            with open(original_path, 'wb') as f_out:
                f_out.write(response.content)
            results['original'] = str(original_path)
            print("✅ Original logo processed")
        else:
            print("❌ Original processing failed")
            return False
    
    # 2. Test new compound path vector trace
    print("\n🎨 Testing new compound path vector trace...")
    with open(test_image_path, 'rb') as f:
        files = {'file': ('mercyevol.png', f, 'image/png')}
        response = session.post(f"{base_url}/preview/vector", files=files)
        
        if response.status_code == 200:
            vector_path = test_dir / "vector_compound.png"
            with open(vector_path, 'wb') as f_out:
                f_out.write(response.content)
            results['vector'] = str(vector_path)
            print("✅ Compound path vector trace processed")
        else:
            print("❌ Vector processing failed")
            return False
    
    # Generate comparison HTML
    print("\n📄 Generating comparison...")
    generate_comparison_html(results, test_dir)
    
    print(f"\n📊 Results saved to: {test_dir}")
    print("✅ Vector comparison test completed successfully")
    
    return True

def generate_comparison_html(results, output_dir):
    """Generate focused comparison HTML"""
    html_content = """<!DOCTYPE html>
<html>
<head>
    <title>Vector Trace Comparison - Original vs Compound Path</title>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container { 
            max-width: 1200px; 
            margin: 0 auto; 
            background: white; 
            padding: 30px; 
            border-radius: 15px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        h1 { 
            color: #2c3e50; 
            text-align: center; 
            margin-bottom: 10px; 
            font-size: 2.5em;
            font-weight: 300;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 40px;
            font-size: 1.1em;
        }
        .comparison-grid { 
            display: grid; 
            grid-template-columns: 1fr 1fr; 
            gap: 40px; 
            margin-bottom: 30px;
        }
        .comparison-item { 
            background: #f8f9fa; 
            border-radius: 12px; 
            padding: 25px; 
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .comparison-item:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.15);
        }
        .comparison-item h3 { 
            margin: 0 0 20px 0; 
            color: #34495e; 
            font-size: 1.5em;
            font-weight: 600;
        }
        .comparison-item img { 
            max-width: 100%; 
            height: 300px; 
            object-fit: contain; 
            border: 2px solid #e9ecef;
            border-radius: 8px;
            background: repeating-conic-gradient(#f0f0f0 0% 25%, transparent 0% 50%) 50% / 20px 20px;
            margin-bottom: 15px;
        }
        .status { 
            margin: 15px 0; 
            font-weight: 600; 
            font-size: 1.1em;
        }
        .success { color: #27ae60; }
        .original { border-left: 5px solid #3498db; }
        .vector { border-left: 5px solid #9b59b6; }
        .details {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            text-align: left;
            font-size: 0.9em;
            color: #2c3e50;
        }
        .summary { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px; 
            border-radius: 12px; 
            margin-bottom: 30px;
            text-align: center;
        }
        .summary h2 {
            margin: 0 0 15px 0;
            font-size: 1.8em;
            font-weight: 300;
        }
        .improvements {
            background: #e8f5e8;
            border: 1px solid #27ae60;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
        }
        .improvements h4 {
            color: #27ae60;
            margin: 0 0 15px 0;
        }
        .improvements ul {
            margin: 0;
            padding-left: 20px;
        }
        .improvements li {
            margin: 5px 0;
            color: #2c3e50;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎨 Vector Trace Comparison</h1>
        <div class="subtitle">Original Logo vs New Compound Path Vector Trace</div>
        
        <div class="summary">
            <h2>🔍 Focused Analysis</h2>
            <p>Comparing the original logo with the new compound path vector tracing method to showcase improvements in detail preservation, nested contour detection, and compound path structure.</p>
        </div>
        
        <div class="comparison-grid">
            <div class="comparison-item original">
                <h3>📸 Original Logo</h3>
                <img src="original.png" alt="Original Logo">
                <div class="status success">✅ Source Image</div>
                <div class="details">
                    <strong>Format:</strong> PNG with transparency<br>
                    <strong>Details:</strong> Original raster logo with all fine features<br>
                    <strong>Features:</strong> Inner circles, star faces, letter details
                </div>
            </div>
            
            <div class="comparison-item vector">
                <h3>🎯 Compound Path Vector</h3>
                <img src="vector_compound.png" alt="Compound Path Vector">
                <div class="status success">✅ Advanced Vector Trace</div>
                <div class="details">
                    <strong>Algorithm:</strong> Hierarchy-based + Compound Path Preservation<br>
                    <strong>Quality:</strong> Ultra-conservative epsilon 0.00005<br>
                    <strong>Methods:</strong> Multi-strategy mask conversion<br>
                    <strong>Structure:</strong> External + nested contour preservation
                </div>
            </div>
        </div>
        
        <div class="improvements">
            <h4>🚀 Key Improvements in New Vector Trace Method:</h4>
            <ul>
                <li><strong>Compound Path Structure:</strong> Proper SVG compound paths with fill-rule="evenodd"</li>
                <li><strong>Nested Contour Detection:</strong> Full RETR_TREE hierarchy with parent-child relationships</li>
                <li><strong>Detail Preservation:</strong> Ultra-conservative simplification (epsilon 0.00005 for small contours)</li>
                <li><strong>Multi-Strategy Detection:</strong> Direct alpha, adaptive thresholding, edge-based, and distance transform</li>
                <li><strong>Hole Preservation:</strong> Inner details like circles in letters and star faces maintained</li>
                <li><strong>Recursive Processing:</strong> External contours processed with all nested children</li>
            </ul>
        </div>
        
        <div style="text-align: center; margin-top: 30px; color: #7f8c8d; font-size: 0.9em;">
            <p>Generated by Zyppts.V10 Logo Format Generator</p>
            <p>Compound Path Vector Tracing Technology</p>
        </div>
    </div>
</body>
</html>"""
    
    with open(output_dir / "vector_comparison.html", 'w') as f:
        f.write(html_content)
    
    print(f"📄 Comparison HTML generated: {output_dir}/vector_comparison.html")

if __name__ == "__main__":
    success = test_vector_comparison()
    sys.exit(0 if success else 1) 