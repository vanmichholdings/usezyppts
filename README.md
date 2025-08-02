# ZYPPTS Design Automation Software

A high-performance Flask web application for professional logo processing with parallel execution, smart background removal, and comprehensive format generation.

## 🚀 Features

### **Core Processing**
- ✅ **Smart Transparent PNG** - Intelligent background removal with edge detection
- ✅ **Enhanced Contour Cutline** - Full-color logo preservation + professional magenta cutlines
- ✅ **Vector Tracing** - High-quality SVG generation with vtracer
- ✅ **Color Separations** - PMS/CMYK separation for print production
- ✅ **Effects** - Distressed, halftone, black & white versions
- ✅ **Social Media Formats** - All major platform sizes
- ✅ **Print Formats** - PDF, WebP, favicon, email headers

### **Performance & Scalability**
- ⚡ **Parallel Processing** - Sub-10 second processing with 8 workers
- 🔄 **Real-time Progress Bar** - Server-sent events for live updates
- 💾 **Redis Integration** - Session management, rate limiting, and caching
- 🎯 **Smart Background Detection** - Perfect for complex logos with nested details
- 📊 **Production Ready** - Logging, error handling, and monitoring

### **User Management**
- 👤 **Authentication System** - User registration, login, and sessions
- 💳 **Subscription Management** - Stripe integration for payments
- 📧 **Email Notifications** - Contact forms and user communications
- 🔒 **Rate Limiting** - API protection and abuse prevention

## 📁 Project Structure

```
zyppts_v10/
├── Backend/                    # Flask application
│   ├── utils/                  # Core processing logic
│   │   └── logo_processor.py   # Main logo processing engine
│   ├── routes.py              # API endpoints and views
│   ├── models.py              # Database models
│   ├── config.py              # Application configuration
│   ├── app_config.py          # Flask app factory
│   ├── run.py                 # Application entry point
│   ├── uploads/               # User uploaded files
│   ├── outputs/               # Generated files
│   ├── cache/                 # Processing cache
│   ├── temp/                  # Temporary files
│   └── logs/                  # Application logs
├── Frontend/                   # Templates and static files
│   ├── templates/             # Jinja2 templates
│   └── static/                # CSS, JS, images
├── venv/                      # Virtual environment
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.11+
- Redis Server
- macOS/Linux (recommended)

### 1. Clone and Setup

```bash
git clone <repository-url>
cd zyppts_v10
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Install Redis

**macOS (Homebrew):**
```bash
brew install redis
brew services start redis
```

**Ubuntu/Debian:**
```bash
sudo apt-get install redis-server
sudo systemctl start redis-server
```

### 3. Environment Configuration

Create a `.env` file in the Backend directory:

```env
SECRET_KEY=your-secret-key-here
FLASK_DEBUG=True
DATABASE_URL=sqlite:///instance/app.db
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
STRIPE_SECRET_KEY=sk_test_your-stripe-key
STRIPE_WEBHOOK_SECRET=whsec_your-webhook-secret
```

### 4. Initialize Database

```bash
cd Backend
python -c "from app_config import create_app, db; app = create_app(); app.app_context().push(); db.create_all()"
```

## 🚀 Running the Application

### Development Mode
```bash
cd Backend
python run.py
```

The application will be available at: **http://localhost:5003**

### Production Mode
```bash
cd Backend
gunicorn --bind 0.0.0.0:5003 --workers 4 app_config:create_app()
```

## 📋 System Requirements

### Core Dependencies
- **Flask 3.1+** - Web framework
- **Redis 6.0+** - Session storage and rate limiting
- **Pillow (PIL)** - Image processing
- **OpenCV** - Computer vision for smart background removal
- **NumPy** - Array operations
- **vtracer** - Vector tracing

### Optional Dependencies
- **scikit-learn** - K-means clustering for color analysis
- **cairosvg** - SVG to PDF conversion
- **stripe** - Payment processing

## 🎯 Key Features Deep Dive

### Smart Background Removal
Uses intelligent contour analysis to distinguish between:
- Outer logo boundaries
- Interior details (letters, stars, nested elements)
- Background areas to be removed

Perfect for complex logos like crests with internal white details.

### Enhanced Contour Cutline
- Preserves full-color original logo
- Adds professional magenta (#FF00FF) cutlines
- Supports hierarchical contour detection
- Generates PNG, SVG, and PDF outputs for print production

### Parallel Processing
- Utilizes ThreadPoolExecutor for concurrent task execution
- Processes multiple variations simultaneously
- Optimized for sub-10 second completion times
- Real-time progress tracking via Server-Sent Events

### Redis Integration
- Session management with persistence across restarts
- Rate limiting with Redis storage
- Caching for improved performance
- Horizontal scaling support

## 🔧 Configuration

### Performance Tuning
Adjust parallel processing in `Backend/utils/logo_processor.py`:

```python
# Increase workers for faster processing (requires more CPU/memory)
processor = LogoProcessor(max_workers=12)

# Optimize for memory usage
processor = LogoProcessor(max_workers=4)
```

### Redis Configuration
Modify Redis settings in `Backend/config.py`:

```python
REDIS_HOST = 'localhost'
REDIS_PORT = 6379
REDIS_DB = 0
REDIS_MAX_CONNECTIONS = 1000
```

## 📊 API Endpoints

### Logo Processing
- `POST /logo_processor` - Main logo processing endpoint
- `GET /progress/<session_id>` - Real-time progress updates (SSE)

### Preview Endpoints
- `POST /preview/transparent` - Transparent PNG preview
- `POST /preview/vector` - Vector trace preview
- `POST /preview/outline` - Contour cutline preview
- `POST /preview/black` - Black & white preview

### User Management
- `POST /login` - User authentication
- `POST /register` - User registration
- `GET /account` - Account dashboard

## 🧪 Testing

### Basic Functionality Test
```bash
cd Backend
python -c "
from utils.logo_processor import LogoProcessor
processor = LogoProcessor()
print('✅ Logo processor initialized successfully')
"
```

### Redis Connection Test
```bash
redis-cli ping
# Should return: PONG
```

### Full System Test
Upload a logo through the web interface and verify:
- Progress bar displays correctly
- All selected variations are generated
- Download package contains expected files

## 🔍 Troubleshooting

### Common Issues

**Redis Connection Errors:**
```bash
# Check if Redis is running
brew services list | grep redis
redis-cli ping
```

**Import Errors:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

**Memory Issues with Large Files:**
- Reduce `max_workers` in LogoProcessor
- Increase system memory
- Use smaller input files for testing

**Permission Errors:**
```bash
# Ensure proper directory permissions
chmod 755 Backend/uploads Backend/outputs Backend/cache Backend/temp
```

## 📝 Logging

Application logs are stored in `Backend/logs/zyppts.log` and include:
- Processing times and performance metrics
- Error details with stack traces
- User activity and system events
- Redis connection status

## 🔒 Security

- Rate limiting: 200 requests/day, 50 requests/hour per IP
- Secure session cookies with Redis storage
- File upload validation and size limits
- SQL injection protection with SQLAlchemy ORM

## 🚀 Deployment

### Production Checklist
- [ ] Set `FLASK_DEBUG=False` in environment
- [ ] Configure proper SSL certificates
- [ ] Set up database backups
- [ ] Configure Redis persistence
- [ ] Set up monitoring and logging
- [ ] Configure firewall rules
- [ ] Set up automated backups

### Docker Deployment (Optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5003
CMD ["python", "Backend/run.py"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -m 'Add feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## 📄 License

This project is proprietary software. All rights reserved.

## 📞 Support

For technical support or questions:
- Email: support@zyppts.com
- Documentation: [Internal Wiki]
- Issues: [Internal Issue Tracker]

---

**ZYPPTS Logo Processor v10** - Professional logo processing with intelligent background removal and parallel execution. 