#!/bin/bash

echo "🔧 Activating Zyppts Environment..."

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$PROJECT_ROOT/Backend"

# Check if virtual environment exists
if [ ! -d "$BACKEND_DIR/venv" ]; then
    echo "❌ Virtual environment not found in $BACKEND_DIR/venv"
    echo "💡 Creating new virtual environment..."
    cd "$BACKEND_DIR"
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source "$BACKEND_DIR/venv/bin/activate"

# Install dependencies if requirements.txt exists
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    echo "📦 Installing dependencies..."
    pip install -r "$PROJECT_ROOT/requirements.txt"
    echo "✅ Dependencies installed"
fi

echo "🎉 Environment activated successfully!"
echo "💡 You can now run: python Backend/run.py" 