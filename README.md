# Zyppts V10 - Logo Format Generator

A powerful Flask-based web application for processing and generating various logo formats with advanced AI-powered features.

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Virtual environment
- Required dependencies (see requirements.txt)

### Installation & Setup

1. **Clone and navigate to the project:**
   ```bash
   cd zyppts_v10
   ```

2. **Create and activate virtual environment:**
   ```bash
   cd Backend
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   cd ..
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application:**
   ```bash
   # Option 1: Using the main startup script
   python Backend/scripts/start_app.py
   
   # Option 2: Manual startup
   cd Backend
   source venv/bin/activate
   python run.py
   ```

5. **Access the application:**
   ```
   http://localhost:5003
   ```

## 📁 Project Structure

```
zyppts_v10/
├── Frontend/                 # Frontend components
│   ├── templates/           # HTML templates (Flask Jinja2)
│   └── static/              # CSS, JS, images, assets
├── Backend/                 # Backend components
│   ├── routes.py            # Flask routes and API endpoints
│   ├── models.py            # Database models
│   ├── config.py            # Configuration settings
│   ├── app_config.py        # Flask app factory
│   ├── run.py               # Application entry point
│   ├── utils/               # Utility functions
│   │   └── logo_processor.py # Main processing engine
│   ├── scripts/             # Startup and utility scripts
│   │   ├── start_app.py     # Main startup script
│   │   ├── start_app_fixed.py # Alternative startup
│   │   ├── start_app.sh     # Shell startup script
│   │   └── activate.sh      # Environment activation
│   ├── tests/               # Test files
│   │   ├── test_complete.py # Complete structure test
│   │   ├── test_structure.py # Structure validation
│   │   └── test_app.py      # App functionality test
│   ├── docs/                # Documentation
│   │   └── STRUCTURE_GUIDE.md # Structure documentation
│   ├── venv/                # Python virtual environment
│   ├── uploads/             # File upload directory
│   ├── outputs/             # Processed file outputs
│   ├── cache/               # Application cache
│   └── logs/                # Application logs
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🛠️ Development

### Running Tests
```bash
# Test the complete structure
python Backend/tests/test_complete.py

# Test basic structure
python Backend/tests/test_structure.py

# Test app functionality
python Backend/tests/test_app.py
```

### Development Workflow
- **Frontend Development**: Work in `Frontend/` directory
- **Backend Development**: Work in `Backend/` directory
- **Scripts**: Use files in `Backend/scripts/` for automation
- **Testing**: Use files in `Backend/tests/` for validation

## 🔧 Configuration

The application uses a clean Frontend/Backend structure:
- **Templates**: `Frontend/templates/` → `Backend/templates` (symlink)
- **Static Files**: `Frontend/static/` → `Backend/static` (symlink)
- **Uploads**: `Backend/uploads/`
- **Outputs**: `Backend/outputs/`
- **Cache**: `Backend/cache/`
- **Logs**: `Backend/logs/`

## 🚀 Features

- **ML Background Removal** (rembg library)
- **SVG Path Optimization** (svgpathtools)
- **Vector Tracing** (OpenCV)
- **PDF Support**
- **User Authentication**
- **File Upload/Processing**
- **Multiple Output Formats**
- **Social Media Optimization**
- **Color Separations**
- **Advanced Effects**

## 📚 Documentation

- **Structure Guide**: `Backend/docs/STRUCTURE_GUIDE.md`
- **API Documentation**: Available in the application
- **Configuration**: See `Backend/config.py`

## 🐛 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure virtual environment is activated
   ```bash
   cd Backend
   source venv/bin/activate
   cd ..
   ```

2. **Missing Dependencies**: Install requirements
   ```bash
   pip install -r requirements.txt
   ```

3. **Port Already in Use**: Change port in `Backend/run.py`
   ```python
   app.run(host='0.0.0.0', port=5004, debug=True)
   ```

4. **Template/Static Not Found**: Recreate symbolic links
   ```bash
   cd Backend
   ln -sf ../Frontend/templates templates
   ln -sf ../Frontend/static static
   ```

### Getting Help

- Check the logs in `Backend/logs/`
- Run tests to validate structure
- Review the structure guide in `Backend/docs/`

## 📄 License

This project is proprietary software. All rights reserved.

## 🤝 Contributing

For development and contribution guidelines, please refer to the internal documentation in `Backend/docs/`.
