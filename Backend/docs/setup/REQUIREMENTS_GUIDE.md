# Requirements Management Guide

## 📁 Requirements Files Overview

This project uses different requirements files for different purposes:

### 1. `requirements.txt` (Main - Full Development)
- **Purpose**: Full development environment with all dependencies
- **Use**: Local development on macOS
- **Size**: ~230 lines, includes all packages
- **Features**: Testing, development tools, full functionality

### 2. `requirements-fly.txt` (Optimized - Production)
- **Purpose**: Optimized for Fly.io deployment
- **Use**: Production deployment
- **Size**: ~132 lines, essential packages only
- **Features**: Core functionality, no dev tools

### 3. `requirements-dev.txt` (Development - Full)
- **Purpose**: Complete development environment
- **Use**: Local development setup
- **Size**: ~230 lines, includes all packages
- **Features**: Testing, development tools, full functionality

## 🚀 Deployment vs Development

### For Fly.io Deployment:
```bash
# Dockerfile automatically uses requirements-fly.txt
flyctl deploy
```

### For Local Development:
```bash
# Use the setup script
./setup-dev.sh

# Or manually
source venv/bin/activate
pip install -r requirements-dev.txt
```

## 📊 Package Comparison

| Package Category | Development | Production | Notes |
|------------------|-------------|------------|-------|
| Core Flask | ✅ | ✅ | Essential |
| Image Processing | ✅ | ✅ | Essential |
| Vector Graphics | ✅ | ✅ | Essential (vtracer) |
| Machine Learning | ✅ | ✅ | Essential (torch) |
| Background Tasks | ✅ | ✅ | Essential |
| Testing Tools | ✅ | ❌ | Development only |
| Development Tools | ✅ | ❌ | Development only |
| Monitoring | ✅ | ❌ | Development only |
| WebSocket | ✅ | ❌ | Development only |

## 🔧 File Structure

```
zyppts_v10/
├── requirements.txt          # Main requirements (full)
├── requirements-fly.txt      # Optimized for Fly.io
├── requirements-dev.txt      # Development requirements
├── setup-dev.sh             # Development setup script
├── Dockerfile               # Uses requirements-fly.txt
└── .dockerignore            # Excludes requirements.txt
```

## 🎯 Key Differences

### Production (requirements-fly.txt):
- ✅ All essential functionality
- ✅ vtracer for vector tracing
- ✅ OpenCV for image processing
- ✅ PyTorch for ML features
- ❌ No testing frameworks
- ❌ No development tools
- ❌ No monitoring tools
- ❌ No WebSocket libraries

### Development (requirements.txt/dev.txt):
- ✅ All production features
- ✅ pytest for testing
- ✅ black/flake8 for formatting
- ✅ sentry-sdk for monitoring
- ✅ websockets for real-time
- ✅ All development tools

## 🚀 Quick Commands

### Setup Development:
```bash
cd Backend
./setup-dev.sh
```

### Deploy to Fly.io:
```bash
cd Backend
./deploy_fly.sh
```

### Check Current Environment:
```bash
# Check which requirements are installed
pip list | grep -E "(pytest|black|sentry|websocket)"
```

## 🔍 Verification

### Verify Production Build:
```bash
# Check Docker build uses correct requirements
docker build -t test-build .
docker run --rm test-build pip list | wc -l
# Should show ~80-90 packages (optimized)
```

### Verify Development Environment:
```bash
# Check development environment
source venv/bin/activate
pip list | wc -l
# Should show ~120-130 packages (full)
```

## 🎯 Best Practices

1. **Always use `requirements-fly.txt` for deployment**
2. **Always use `requirements-dev.txt` for development**
3. **Keep `requirements.txt` as the main reference**
4. **Test both environments regularly**
5. **Update all files when adding new dependencies**

## 🔧 Troubleshooting

### If deployment fails:
```bash
# Check if using correct requirements
cat Dockerfile | grep requirements
# Should show: COPY requirements-fly.txt requirements.txt
```

### If development setup fails:
```bash
# Reinstall development requirements
pip install -r requirements-dev.txt --force-reinstall
```

### If packages are missing:
```bash
# Check which requirements file has the package
grep -r "package_name" requirements*.txt
```
