# Zyppts - Vector Logo Processing Application

A comprehensive web application for processing and vectorizing logos with advanced smart code enhancements.

## Project Structure

```
zyppts_v10/
├── Frontend/           # Frontend components
│   ├── templates/      # HTML templates
│   └── static/         # CSS, JS, images, and static assets
├── Backend/            # Backend components
│   ├── routes.py       # Flask routes and API endpoints
│   ├── models.py       # Database models
│   ├── config.py       # Configuration settings
│   ├── utils/          # Utility functions
│   │   └── logo_processor.py  # Main logo processing engine
│   ├── tests/          # Test files
│   ├── uploads/        # File upload directory
│   ├── outputs/        # Processed file outputs
│   ├── cache/          # Application cache
│   └── logs/           # Application logs
├── venv/               # Python virtual environment
├── requirements.txt    # Python dependencies
├── activate.sh         # Virtual environment activation script
└── README.md          # This file
```

## Features

- **Logo Vectorization**: Convert raster images to scalable vector graphics
- **AI Enhancement**: Advanced image processing with AI models
- **Quality Optimization**: Multiple quality levels and processing options
- **User Management**: Authentication and user account management
- **Subscription Plans**: Tiered pricing with different feature sets
- **Real-time Processing**: Background task processing with Celery

## Technology Stack

### Backend
- **Flask**: Web framework
- **SQLAlchemy**: Database ORM
- **Celery**: Background task processing
- **OpenCV**: Image processing
- **Pillow**: Image manipulation
- **NumPy**: Numerical computing
- **PyTorch**: Deep learning framework
- **scikit-image**: Advanced image processing
- **scikit-learn**: Machine learning
- **Rembg**: Background removal
- **Diffusers**: AI image generation

### Frontend
- **HTML5/CSS3**: Modern responsive design
- **JavaScript**: Interactive user interface
- **Bootstrap**: UI framework
- **jQuery**: DOM manipulation

### AI/ML
- **Waifu2x**: Image upscaling
- **Real-ESRGAN**: Image enhancement
- **VTracer**: Vector tracing

## Quick Start

### Option 1: Using the Activation Script (Recommended)
```bash
# Make the script executable (first time only)
chmod +x activate.sh

# Activate the environment
./activate.sh
```

### Option 2: Manual Setup
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Installation

1. Clone the repository
2. Run the activation script: `./activate.sh`
3. Set up the database:
   ```bash
   cd Backend
   python init_db.py
   ```
4. Run the application:
   ```bash
   cd Backend
   python run.py
   ```

## Development

- **Frontend Development**: Work in the `Frontend/` directory
- **Backend Development**: Work in the `Backend/` directory
- **Testing**: Run tests from the `Backend/` directory

## Dependencies

All required dependencies are listed in `requirements.txt` and include:

### Core Dependencies
- `opencv-contrib-python-headless>=4.8.0`
- `numpy>=1.24.0`
- `Pillow>=10.0.0`
- `torch>=2.0.0`
- `scikit-image>=0.21.0`
- `scipy>=1.11.0`

### Advanced Processing
- `rembg>=2.0.0` - Background removal
- `diffusers>=0.21.0` - AI image generation
- `psd-tools>=1.9.0` - Photoshop file support
- `colormath>=3.0.0` - Color management
- `cairosvg>=2.7.0` - SVG processing

### Web Framework
- `flask>=2.3.0`
- `celery>=5.3.0`
- `sqlalchemy>=2.0.0`

## Troubleshooting

### Import Errors
If you encounter import errors, ensure you're using the virtual environment:
```bash
source venv/bin/activate
```

### Missing Dependencies
If dependencies are missing, reinstall them:
```bash
pip install -r requirements.txt
```

### Virtual Environment Issues
If the virtual environment is corrupted, recreate it:
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## License

This project is proprietary software. All rights reserved. # Zyppts
# ZypptsV10
