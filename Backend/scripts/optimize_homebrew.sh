#!/bin/bash
# Homebrew Optimization Script for Zyppts Image Processing
# Created: $(date)

set -e  # Exit on any error

echo "====================================================="
echo "  Optimizing environment with Homebrew for Zyppts"
echo "====================================================="

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew is not installed. Installing now..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH if needed
    if [[ $(uname -m) == 'arm64' ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    else
        echo 'eval "$(/usr/local/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/usr/local/bin/brew shellenv)"
    fi
else
    echo "✓ Homebrew is already installed"
fi

# Update Homebrew
echo "Updating Homebrew..."
brew update

# Install core image processing libraries
echo "Installing optimized image processing libraries..."
brew install cairo pixman libpng jpeg libtiff webp || true

# Install optimized math libraries
echo "Installing optimized math libraries..."
brew install openblas || true

# Install libraries for vector graphics
echo "Installing vector graphics libraries..."
brew install librsvg potrace || true

# Install OpenCV with optimization flags
echo "Installing OpenCV with optimizations..."
brew install opencv || true

# Install libraries for GPU acceleration
echo "Installing GPU acceleration libraries..."
brew install libomp || true

# Check if Python 3.13 is installed through Homebrew
if ! brew list python@3.13 &>/dev/null; then
    echo "Installing Python 3.13 through Homebrew..."
    brew install python@3.13
else
    echo "✓ Python 3.13 is already installed through Homebrew"
fi

# Get Homebrew Python path
BREW_PYTHON=$(brew --prefix python@3.13)/bin/python3.13

# Setup virtual environment using Homebrew Python
echo "Setting up virtual environment with Homebrew Python..."
PROJECT_DIR="$(pwd)"
cd "$PROJECT_DIR"

# Remove existing virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf .venv
fi

# Create new virtual environment
echo "Creating new virtual environment with Homebrew Python..."
$BREW_PYTHON -m venv .venv

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Configure pip to use Homebrew libraries
echo "Configuring pip to use Homebrew libraries..."
mkdir -p ~/.config/pip
echo "[global]
find-links = $(brew --prefix)/lib" > ~/.config/pip/pip.conf

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install packages with optimized build flags
echo "Installing Python packages with optimized build flags..."
export CFLAGS="-I$(brew --prefix)/include"
export LDFLAGS="-L$(brew --prefix)/lib"
export OPENBLAS="$(brew --prefix openblas)"
export PKG_CONFIG_PATH="$(brew --prefix)/lib/pkgconfig"

# Install requirements
echo "Installing requirements with optimized flags..."
pip install -r requirements.txt

# Create runtime optimization script
echo "Creating runtime optimization script..."
cat > "$PROJECT_DIR/optimize_runtime.sh" << 'EOF'
#!/bin/bash
# Runtime optimization for Zyppts

# Set environment variables for better performance
export OMP_NUM_THREADS=$(sysctl -n hw.ncpu)
export MKL_NUM_THREADS=$(sysctl -n hw.ncpu)
export OPENBLAS_NUM_THREADS=$(sysctl -n hw.ncpu)
export VECLIB_MAXIMUM_THREADS=$(sysctl -n hw.ncpu)

# Activate the virtual environment
source "$(dirname "$0")/.venv/bin/activate"

# Run the application with optimized settings
python "$(dirname "$0")/run.py" "$@"
EOF

chmod +x "$PROJECT_DIR/optimize_runtime.sh"

# Install monitoring tools
echo "Installing monitoring tools..."
brew install htop glances || true

echo "====================================================="
echo "  Optimization complete!"
echo "====================================================="
echo "To run your application with optimized settings, use:"
echo "  ./optimize_runtime.sh"
echo "====================================================="

# Deactivate virtual environment
deactivate
