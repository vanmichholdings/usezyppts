from app_config import create_app
import sys
import importlib.util
from rembg import remove

def check_library(library_name):
    return importlib.util.find_spec(library_name) is not None

def print_startup_info():
    print("=" * 50)
    # Check for ML libraries
    if check_library('rembg'):
        print("✓ ML-based background removal available (rembg library found)")
    
    # Check for PDF support
    if check_library('fitz') or check_library('PyMuPDF'):
        print("✓ PDF support enabled (PyMuPDF library found)")
    
    # Check for SVG tools
    if check_library('svgpathtools'):
        print("✓ SVG path optimization enabled (svgpathtools library found)")
    
    # Check for OpenCV
    if check_library('cv2'):
        print("✓ Vector tracing enabled (OpenCV library found)")
    
    print("=" * 50)
    print("Zyppts.V10 Logo Format Generator")
    print("=" * 50)
    
    print("✓ ML-based background removal")
    print("✓ PDF support")
    print("✓ Vector tracing")
    print("✓ SVG path optimization")
    print("=" * 50)
    print("Starting server on http://localhost:5003")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)

if __name__ == '__main__':
    print_startup_info()
    app = create_app()
    app.run(host='0.0.0.0', port=5003, debug=True)