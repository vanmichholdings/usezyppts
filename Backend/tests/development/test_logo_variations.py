import os
import sys
from PIL import Image
from zyppts.utils.logo_processor import LogoProcessor

def test_logo_variations():
    # Set up paths
    logo_path = os.path.join('zyppts', 'static', 'images', 'zyppts-logo-new.png')
    output_dir = os.path.join('outputs', 'test_variations')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the logo processor
    processor = LogoProcessor(
        upload_folder=os.path.join(output_dir, 'uploads'),
        output_folder=os.path.join(output_dir, 'outputs'),
        temp_folder=os.path.join(output_dir, 'temp')
    )
    
    # Define test variations
    test_cases = [
        {'name': 'transparent', 'options': {'transparent_png': True}},
        {'name': 'black_white', 'options': {'black_version': True}},
        {'name': 'distressed', 'options': {'distressed': True}},
        {'name': 'vector_trace', 'options': {'vector_trace': True}},
        {'name': 'color_separations', 'options': {'color_separations': True}},
        {
            'name': 'all_effects', 
            'options': {
                'transparent_png': True,
                'black_version': True,
                'distressed': True,
                'vector_trace': True,
                'color_separations': True
            }
        }
    ]
    
    print(f"Testing logo variations for: {logo_path}")
    print("=" * 50)
    
    # Test each variation
    for test_case in test_cases:
        print(f"\nTesting: {test_case['name']}")
        print("-" * 30)
        
        try:
            # Process the logo with current options
            result = processor.process_logo(logo_path, test_case['options'])
            
            if not result or not result.get('success'):
                print(f"❌ Failed: {result.get('message', 'Unknown error')}")
                continue
                
            # Print results
            print("✅ Successfully generated:")
            for output_type, output_path in result.get('outputs', {}).items():
                if isinstance(output_path, dict):
                    # Handle nested outputs like color separations
                    print(f"  - {output_type}:")
                    for sub_type, sub_path in output_path.items():
                        # For color separations, the path includes the separations directory
                        full_path = os.path.join('outputs', 'test_variations', 'outputs', sub_path)
                        print(f"    - {sub_type}: {sub_path}")
                        # Verify file exists
                        if os.path.exists(full_path):
                            print(f"      ✓ File exists at: {full_path}")
                        else:
                            print(f"      ❌ File NOT found at: {full_path}")
                else:
                    # For regular outputs, the path is relative to the output directory
                    filename = os.path.basename(output_path)
                    full_path = os.path.join('outputs', 'test_variations', 'outputs', filename)
                    print(f"  - {output_type}: {filename}")
                    # Verify file exists
                    if os.path.exists(full_path):
                        print(f"    ✓ File exists at: {full_path}")
                    else:
                        print(f"    ❌ File NOT found at: {full_path}")
                    
        except Exception as e:
            print(f"❌ Error processing {test_case['name']}: {str(e)}")
    
    print("\nTest complete! Check the output directory for results.")

if __name__ == "__main__":
    test_logo_variations()
