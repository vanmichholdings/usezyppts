import os
import sys
from PIL import Image
import importlib.util

# Dynamically load LogoProcessor to avoid initializing the full Flask app,
# which was causing library conflicts with Python 3.13.
module_path = os.path.join('zyppts', 'utils', 'logo_processor.py')
spec = importlib.util.spec_from_file_location('logo_processor', module_path)
logo_processor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(logo_processor_module)
LogoProcessor = logo_processor_module.LogoProcessor

def test_social_media_variations():
    # Set up paths
    # Using a standard square logo for testing
    logo_path = os.path.join('zyppts', 'static', 'images', 'zyppts-logo-new.png') 
    output_dir = os.path.join('outputs', 'test_social_variations')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the logo processor
    processor = LogoProcessor(
        upload_folder=os.path.join(output_dir, 'uploads'),
        output_folder=os.path.join(output_dir, 'outputs'),
        temp_folder=os.path.join(output_dir, 'temp')
    )
    
    # Define test cases for social media variations
    test_cases = [
        {
            'platform': 'instagram',
            'options': {'generate_preview': True}
        },
        {
            'platform': 'twitter', # Similar aspect ratio, should use content-aware crop
            'options': {'generate_preview': True}
        },
        {
            'platform': 'youtube', # Wider aspect ratio, should use generative fill
            'options': {'generate_preview': True}
        },
        {
            'platform': 'tiktok', # Taller aspect ratio, should use generative fill
            'options': {'generate_preview': True}
        },
        {
            'platform': 'facebook',
            'options': {
                'generate_preview': True,
                'focal_point': (100, 50)  # Manually set a focal point (top-left area)
            }
        }
    ]
    
    print(f"Testing Social Media Variations for: {logo_path}")
    print("=" * 60)

    # Define common options for all platforms
    options = {
        'generate_preview': True,
        'focal_point': (100, 50)  # Example focal point, can be adjusted
    }

    print(f"Processing all social media variations with options: {options}")
    print("=" * 60)

    # Call the parallel processing method once
    all_results = processor.process_all_social_variations(
        file_path=logo_path,
        options=options
    )

    all_successful = True
    for platform, result in all_results.items():
        print(f"\n--- Results for {platform.capitalize()} ---")
        if result and result.get('status') == 'success':
            output_path = result.get('output_path')
            preview_path = result.get('preview_path')
            print(f"✅  Successfully processed for {platform}:")
            print(f"   - Output: {output_path}")
            print(f"   - Preview: {preview_path}")
            print(f"   - Time: {result.get('processing_time')}s")
            
            if not os.path.exists(output_path):
                print(f"   - ❌ VERIFICATION FAILED: Output file not found!")
                all_successful = False
            if not os.path.exists(preview_path):
                print(f"   - ❌ VERIFICATION FAILED: Preview file not found!")
                all_successful = False

        else:
            all_successful = False
            print(f"❌  Processing failed for {platform}.")
            if result and result.get('message'):
                print(f"   - Error: {result.get('message')}")
    
    print("\n" + "=" * 60)
    if all_successful:
        print("✅ All social media variations were generated and verified successfully!")
    else:
        print("❌ Some social media variations failed to generate or verify.")
    print("Check the output directory for results.")

if __name__ == "__main__":
    test_social_media_variations()
