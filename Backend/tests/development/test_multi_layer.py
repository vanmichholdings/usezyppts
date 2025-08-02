#!/usr/bin/env python3
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from utils.logo_processor import LogoProcessor

def test_multi_layer():
    """Test the multi-layer mask approach for free-standing objects."""
    
    # Initialize processor
    processor = LogoProcessor()
    
    # Test image path
    image_path = "original_mercyevol.png"
    output_dir = "multi_layer_output"
    
    print("🎯 Testing Multi-Layer Mask Approach")
    print("=" * 50)
    print("Focus: Create separate layers for each free-standing object")
    print("Using XOR operations to prevent merging")
    
    # Run vectorization
    result = processor.generate_vector_trace(image_path, output_dir)
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"✅ Success!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📄 Output files: {result.get('output_files', [])}")
    
    # Check file sizes
    if os.path.exists(os.path.join(output_dir, "final_mask.png")):
        size = os.path.getsize(os.path.join(output_dir, "final_mask.png"))
        print(f"📊 Mask file size: {size:,} bytes")
    
    print("\n🎨 Multi-Layer Summary:")
    print("• Main mask: Otsu thresholding (circle)")
    print("• Free-standing mask: Multiple thresholds (40-200)")
    print("• Connected component analysis")
    print("• Multi-layer approach with XOR operations")
    print("• Should preserve all 31 detected objects")

if __name__ == "__main__":
    test_multi_layer() 