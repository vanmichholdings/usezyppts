# Backend Scripts

This directory contains utility scripts for the ZYPPTS application.

## compare_logo_variations.py

A utility script that generates a comprehensive comparison of all logo variations for a given logo file.

### Features
- Automatically finds available logo files in the uploads directory
- Generates all possible variations (transparent PNG, black version, PDF, WebP, favicon, email header, vector trace, color separations, distressed effect, contour cutline, social media formats)
- Creates an HTML comparison page with embedded previews and download links
- Outputs the comparison to `Frontend/templates/compare_logo_variations.html`

### Usage
```bash
cd Backend/scripts
python compare_logo_variations.py
```

### Requirements
- A logo file must exist in one of the upload directories
- All required dependencies must be installed (see main requirements.txt)
- The application must be properly configured

### Output
The script generates an HTML file at `Frontend/templates/compare_logo_variations.html` that contains:
- Visual previews of all logo variations
- Download links for each variation
- Organized layout with proper styling
- Base64-embedded images for immediate viewing

## Other Scripts

- `test_vector_simple.py` - Simple vector tracing test
- `test_paywall.py` - Paywall functionality testing
- `start_app.sh` - Application startup script
- `start_app.py` - Python application starter
- `activate.sh` - Environment activation script
- `provision_user.py` - User provisioning utility
- `optimize_homebrew.sh` - Homebrew optimization
- `update_test_user_credits.py` - Test user credit management
- `check_test_user.py` - Test user verification
- `update_test_user.py` - Test user updates
- `init_db.py` - Database initialization
- `install_dependencies.sh` - Dependency installation 