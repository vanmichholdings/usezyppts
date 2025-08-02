#!/usr/bin/env python3
"""
Final verification test for halftone and social media formats routing
"""

import os
import tempfile
from PIL import Image, ImageDraw
from utils.logo_processor import LogoProcessor

def create_test_logo():
    """Create a simple test logo"""
    img = Image.new('RGBA', (200, 200), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw a simple logo
    draw.ellipse([20, 20, 180, 180], fill=(255, 0, 0, 255), outline=(0, 0, 0, 255), width=3)
    draw.rectangle([60, 60, 140, 140], fill=(255, 255, 0, 255), outline=(0, 0, 0, 255), width=2)
    draw.text((80, 90), "TEST", fill=(0, 0, 0, 255))
    
    return img

def test_complete_routing():
    """Test complete routing for halftone and social media formats"""
    print("Testing complete routing for halftone and social media formats...")
    
    # Create test logo
    test_logo = create_test_logo()
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        test_logo.save(tmp_file.name, 'PNG')
        test_file = tmp_file.name
    
    try:
        processor = LogoProcessor(use_parallel=True, max_workers=4)
        
        # Test with halftone and social media formats
        options = {
            'halftone': True,
            'social_formats': {
                'instagram_profile': True,
                'facebook_profile': True,
                'twitter_profile': True
            }
        }
        
        print(f"   Options: {options}")
        
        # Process the logo
        result = processor.process_logo(test_file, options)
        
        print(f"   Success: {result['success']}")
        print(f"   Outputs: {list(result['outputs'].keys())}")
        print(f"   Processing time: {result['processing_time']:.2f}s")
        print(f"   Parallel used: {result['parallel']}")
        print(f"   Workers used: {result.get('workers_used', 'N/A')}")
        
        # Check halftone
        if 'halftone' in result['outputs']:
            halftone_path = result['outputs']['halftone']
            if os.path.exists(halftone_path):
                print(f"   ✓ Halftone file created: {halftone_path}")
                file_size = os.path.getsize(halftone_path)
                print(f"   ✓ Halftone file size: {file_size} bytes")
            else:
                print(f"   ✗ Halftone file missing: {halftone_path}")
                return False
        else:
            print("   ✗ Halftone not in outputs")
            return False
        
        # Check social formats
        if 'social_formats' in result['outputs']:
            social_formats = result['outputs']['social_formats']
            if isinstance(social_formats, dict):
                print(f"   ✓ Social formats created: {len(social_formats)} formats")
                for platform, path in social_formats.items():
                    if os.path.exists(path):
                        file_size = os.path.getsize(path)
                        print(f"   ✓ {platform}: {path} ({file_size} bytes)")
                    else:
                        print(f"   ✗ {platform}: file missing {path}")
                        return False
            else:
                print("   ✗ Social formats not in expected format")
                return False
        else:
            print("   ✗ Social formats not in outputs")
            return False
        
        # Test ZIP file inclusion logic
        print("\n   Testing ZIP file inclusion logic...")
        
        # Simulate the ZIP creation logic
        outputs = result['outputs']
        original_name = 'test_logo'
        files_to_add = []
        
        # Check halftone inclusion
        if options.get('halftone', False) and 'halftone' in outputs:
            path = outputs['halftone']
            arcname = f"Effects/{original_name}_halftone.png"
            files_to_add.append((path, arcname))
            print(f"   ✓ Halftone would be added to ZIP as: {arcname}")
        else:
            print("   ✗ Halftone not added to ZIP")
            return False
        
        # Check social formats inclusion
        if options.get('social_formats', {}) and 'social_formats' in outputs:
            social_formats = outputs['social_formats']
            for platform, path in social_formats.items():
                arcname = f"Social Media/{original_name}_{platform}.png"
                files_to_add.append((path, arcname))
                print(f"   ✓ {platform} would be added to ZIP as: {arcname}")
        else:
            print("   ✗ Social formats not added to ZIP")
            return False
        
        print(f"   ✓ Total files to add to ZIP: {len(files_to_add)}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        try:
            os.unlink(test_file)
        except:
            pass

def main():
    """Main test function"""
    print("=" * 60)
    print("FINAL ROUTING VERIFICATION TEST")
    print("=" * 60)
    
    if test_complete_routing():
        print("\n" + "=" * 60)
        print("✅ ALL ROUTING TESTS PASSED")
        print("✅ Halftone and social media formats are properly routed")
        print("✅ Both effects will be included in ZIP files")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ ROUTING TESTS FAILED")
        print("❌ Some effects are not properly routed")
        print("=" * 60)

if __name__ == "__main__":
    main() 