#!/bin/bash

echo "🚀 Starting Zyppts Application..."

# Navigate to project directory
cd /Users/chicoperfecto/zyppts_v10

# Activate virtual environment - now in Backend directory
source Backend/venv/bin/activate

# Create symbolic links for templates and static if they don't exist
cd Backend
if [ ! -L templates ]; then
    ln -sf ../Frontend/templates templates
    echo "✅ Created templates symlink"
fi

if [ ! -L static ]; then
    ln -sf ../Frontend/static static
    echo "✅ Created static symlink"
fi

# Install any missing dependencies
echo "📦 Checking dependencies..."
pip install -r ../requirements.txt

# Start the application
echo "🌐 Starting Flask application on http://localhost:5003"
echo "Press Ctrl+C to stop the server"
echo "=================================="

python run.py 