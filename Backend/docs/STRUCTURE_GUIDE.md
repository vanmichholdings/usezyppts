# Zyppts Frontend/Backend Structure Guide

## Overview

The application has been reorganized into a clean Frontend/Backend structure for better maintainability and separation of concerns.

## Structure

```
zyppts_v10/
├── Frontend/           # Frontend components
│   ├── templates/      # HTML templates (Flask Jinja2)
│   └── static/         # CSS, JS, images, assets
├── Backend/            # Backend components
│   ├── routes.py       # Flask routes and API endpoints
│   ├── models.py       # Database models
│   ├── config.py       # Configuration settings
│   ├── app_config.py   # Flask app factory (NEW)
│   ├── utils/          # Utility functions
│   │   └── logo_processor.py  # Main processing engine
│   ├── uploads/        # File upload directory
│   ├── outputs/        # Processed file outputs
│   ├── cache/          # Application cache
│   └── logs/           # Application logs
├── venv/               # Python virtual environment
├── requirements.txt    # Python dependencies
└── README.md          # Main documentation
```

## Key Changes

### 1. Flask App Configuration
- **New file**: `Backend/app_config.py` - Handles the new structure
- **Updated**: `Backend/run.py` - Uses the new app configuration
- **Template/Static paths**: Now point to Frontend directory

### 2. Path Configuration
- Templates: `Frontend/templates/`
- Static files: `Frontend/static/`
- Uploads: `Backend/uploads/`
- Outputs: `Backend/outputs/`
- Cache: `Backend/cache/`
- Logs: `Backend/logs/`

### 3. Symbolic Links
The app automatically creates symbolic links in the Backend directory:
- `Backend/templates` → `Frontend/templates`
- `Backend/static` → `Frontend/static`

## Running the Application

### Option 1: Using the startup script (Recommended)
```bash
python start_app.py
```

### Option 2: Manual startup
```bash
# Activate virtual environment
source venv/bin/activate

# Navigate to Backend
cd Backend

# Create symbolic links (if needed)
ln -sf ../Frontend/templates templates
ln -sf ../Frontend/static static

# Run the application
python run.py
```

### Option 3: Using the test script
```bash
python test_complete.py
```

## Development Workflow

### Frontend Development
- Work in the `Frontend/` directory
- Templates: `Frontend/templates/`
- Static assets: `Frontend/static/`
- Changes are immediately reflected

### Backend Development
- Work in the `Backend/` directory
- Routes: `Backend/routes.py`
- Models: `Backend/models.py`
- Configuration: `Backend/config.py`

### Adding New Routes
1. Add route in `Backend/routes.py`
2. Use `render_template()` with template names from `Frontend/templates/`
3. Reference static files from `Frontend/static/`

### Adding New Templates
1. Create HTML file in `Frontend/templates/`
2. Reference static files using `url_for('static', filename='...')`
3. Use Flask Jinja2 syntax for dynamic content

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the virtual environment
source venv/bin/activate

# Check if dependencies are installed
pip install -r requirements.txt
```

### Template/Static Not Found
```bash
# Check if symbolic links exist
ls -la Backend/templates Backend/static

# Recreate symbolic links
cd Backend
ln -sf ../Frontend/templates templates
ln -sf ../Frontend/static static
```

### Port Already in Use
```bash
# Change port in Backend/run.py
app.run(host='0.0.0.0', port=5004, debug=True)
```

## Benefits of New Structure

1. **Clear Separation**: Frontend and Backend are completely separated
2. **Easy Maintenance**: Changes to one don't affect the other
3. **Scalability**: Easy to add new frontend frameworks or backend services
4. **Team Collaboration**: Frontend and backend developers can work independently
5. **Deployment**: Can deploy frontend and backend separately if needed

## Migration Notes

- All existing functionality is preserved
- Routes and templates work exactly as before
- Database and user data remain unchanged
- Configuration is automatically updated for new paths 