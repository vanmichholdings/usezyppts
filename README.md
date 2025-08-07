# 🎨 Zyppts v10 - Professional Logo Processing Platform

A powerful, AI-driven logo processing and design automation platform that transforms logos into multiple formats and variations for professional use.

## ✨ Features

### 🎯 Core Logo Processing
- **Transparent PNG Generation** - Smart background removal with AI-powered detection
- **Multiple Format Support** - PNG, JPG, WebP, PDF, SVG processing
- **Vector Tracing** - Convert raster images to scalable vector formats
- **Color Variations** - Generate black, white, and custom color versions
- **Favicon Creation** - Multiple sizes for web applications
- **Email Header Generation** - Optimized headers for email marketing

### 🌟 Advanced Effects
- **Distressed Effects** - Vintage and weathered logo styles
- **Halftone Effects** - Classic printing technique simulation
- **Contour Cutlines** - Perfect for vinyl cutting and CNC machines
- **Color Separations** - Professional printing preparation

### 📱 Social Media Optimization
- **Instagram** - Profile, posts, and stories
- **Facebook** - Profile, posts, and cover images
- **Twitter/X** - Profile, posts, and headers
- **LinkedIn** - Profile, posts, and banners
- **YouTube** - Profile, thumbnails, and banners
- **TikTok** - Profile and video formats
- **Slack & Discord** - Profile images

### 🚀 Performance Features
- **Batch Processing** - Process multiple logos simultaneously (Studio/Enterprise plans)
- **Real-time Progress Tracking** - Live progress updates with detailed status
- **Intelligent Caching** - Fast processing with smart result caching
- **Background Processing** - Celery workers for scalable performance
- **Memory Optimization** - Efficient resource management

### 💳 Subscription Plans
- **Free Plan** - 3 logo credits per month
- **Pro Plan** - 100 logo credits per month ($9.99)
- **Studio Plan** - 500 logo credits per month with batch processing ($29.99)
- **Enterprise Plan** - Unlimited processing with custom integrations ($199)

## 🏗️ Project Structure

```
zyppts_v10/
├── Backend/                    # Flask backend application
│   ├── utils/                  # Core processing utilities
│   │   ├── logo_processor.py   # Main logo processing engine
│   │   ├── celery_worker.py    # Background task processing
│   │   ├── analytics_tracker.py # User analytics tracking
│   │   ├── analytics_collector.py # Analytics data collection
│   │   └── email_sender.py     # Email notification system
│   ├── routes.py               # Main application routes
│   ├── models.py               # Database models
│   ├── config.py               # Application configuration
│   ├── app_config.py           # Flask app configuration
│   ├── run.py                  # Application entry point
│   ├── docker-compose.yml      # Production Docker configuration
│   ├── docker-compose.dev.yml  # Development Docker configuration
│   ├── deploy.sh               # Docker deployment script
│   ├── deploy_fly.sh           # Fly.io deployment script
│   └── requirements.txt        # Backend dependencies
├── Frontend/                   # Frontend templates and assets
│   ├── templates/              # HTML templates
│   │   ├── base.html           # Base template
│   │   └── logo_processor.html # Main logo processor interface
│   └── static/                 # Static assets (CSS, JS, images)
├── Dockerfile                  # Main Docker configuration
├── fly.toml                    # Fly.io deployment configuration
├── requirements.txt            # Root Python dependencies
├── README.md                   # This file
├── FLY_DEPLOYMENT_GUIDE.md     # Fly.io deployment guide
├── .env.example               # Environment variables template
└── venv/                      # Virtual environment
```

## 🛠️ Technology Stack

### Backend
- **Flask 3.1.1** - Web framework
- **SQLAlchemy 2.0.42** - Database ORM
- **Celery 5.5.3** - Background task processing
- **Redis 6.3.0** - Task queue and caching
- **Pillow 11.3.0** - Image processing
- **OpenCV 4.12.0** - Computer vision and image analysis
- **NumPy 2.2.6** - Numerical computing
- **CairoSVG 2.8.2** - SVG processing
- **pdf2image 1.17.0** - PDF to image conversion
- **Stripe 12.4.0** - Payment processing

### Frontend
- **Bootstrap 5.3** - Responsive UI framework
- **Font Awesome 6.4** - Icon library
- **Modern JavaScript** - ES6+ features
- **HTML5/CSS3** - Modern web standards

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Redis server
- PostgreSQL (optional, SQLite for development)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd zyppts_v10
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Start Redis server:**
```bash
redis-server
```

6. **Initialize the database:**
```bash
cd Backend
flask db upgrade
```

7. **Start Celery workers:**
```bash
cd Backend
celery -A utils.celery_worker worker --loglevel=INFO --concurrency=2 --queues=logo_processing
```

8. **Run the application:**
```bash
# Development
flask run

# Production
gunicorn -c Backend/gunicorn.conf.py "Backend:create_app()"
```

## 🐳 Docker Deployment

### Quick Docker Setup
```bash
# Navigate to Backend directory
cd Backend

# Build and start the application
./deploy.sh build
./deploy.sh start

# For development
./deploy.sh dev-start
```

### Docker Commands
```bash
# Production
./deploy.sh build      # Build the application
./deploy.sh start      # Start production environment
./deploy.sh stop       # Stop the application
./deploy.sh restart    # Restart the application
./deploy.sh logs       # View logs
./deploy.sh status     # Check status

# Development
./deploy.sh dev-start  # Start development environment
./deploy.sh dev-stop   # Stop development environment
./deploy.sh dev-logs   # View development logs
```

## ☁️ Fly.io Deployment

### Deploy to Fly.io
```bash
# Navigate to Backend directory
cd Backend

# Deploy to Fly.io
./deploy_fly.sh
```

### Manual Fly.io Setup
```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Login to Fly.io
fly auth login

# Deploy from root directory
fly deploy
```

For detailed Fly.io deployment instructions, see [FLY_DEPLOYMENT_GUIDE.md](FLY_DEPLOYMENT_GUIDE.md).

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:

```env
# Flask Configuration
SECRET_KEY=your-secret-key-here
FLASK_DEBUG=False

# Database
DATABASE_URL=postgresql://user:password@localhost/zyppts

# Email Configuration
MAIL_USERNAME=your-email@gmail.com
MAIL_PASSWORD=your-app-password
MAIL_DEFAULT_SENDER=Zyppts HQ <zyppts@gmail.com>

# Stripe Configuration
STRIPE_SECRET_KEY=sk_test_...
STRIPE_PUBLISHABLE_KEY=pk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...

# Redis Configuration
REDIS_URL=redis://localhost:6379

# Admin Configuration
ADMIN_ALERT_EMAIL=admin@yourdomain.com
SITE_URL=https://yourdomain.com
```

## 📊 Recent Updates

### ✅ Fixed Issues (Latest)
- **Directory Reorganization** - Moved deployment files to Backend directory for better organization
- **Celery Workers** - Fixed import errors and worker startup issues
- **Transparent PNG Generation** - Improved background removal with smart color detection
- **Cache Management** - Fixed caching issues that prevented proper processing
- **UI Improvements** - Removed CSS text appearing at bottom of pages

### 🔧 Technical Improvements
- **Background Processing** - Optimized Celery worker configuration for 20-second processing target
- **Memory Management** - Enhanced resource cleanup and optimization
- **Error Handling** - Improved error recovery and user feedback
- **Performance** - Reduced processing time with intelligent caching

## 🧪 Testing

Run the test suite to verify everything is working:

```bash
cd Backend
python -m pytest tests/
```

Or run individual tests:

```bash
# Test transparent PNG generation
python tests/test_transparent_png.py

# Test Celery workers
python tests/test_app.py

# Test analytics system
python tests/test_analytics_system.py
```

## 📈 Performance

### Processing Times
- **Transparent PNG**: ~3-5 seconds
- **Vector Tracing**: ~8-12 seconds
- **Social Media Formats**: ~2-3 seconds per platform
- **Batch Processing**: ~20 seconds for 10 logos

### Scalability
- **Concurrent Users**: 100+ simultaneous users
- **File Size Limit**: 16MB per file
- **Supported Formats**: PNG, JPG, GIF, SVG, PDF, WebP
- **Output Quality**: Up to 8K resolution

## 🔒 Security

- **Authentication** - Flask-Login with secure session management
- **File Upload** - Secure file handling with validation
- **Rate Limiting** - Protection against abuse
- **Input Validation** - Comprehensive input sanitization
- **HTTPS** - Secure communication in production

## 📞 Support

For support and documentation:
- **Documentation**: Check `Backend/docs/` for detailed guides
- **Issues**: Report bugs through the issue tracker
- **Features**: Request new features through pull requests

## 📄 License

Proprietary - All rights reserved

---

**Built with ❤️ by the Zyppts Team**
