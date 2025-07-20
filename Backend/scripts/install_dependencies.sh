#!/bin/bash

# Create and activate virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Verify installation
echo "Verifying installation..."
python3 -c "
import numpy as np
import cv2
from PIL import Image
import svgwrite
import scipy
import skimage
import torch
import tensorflow as tf
import psutil
import magic
import cairosvg
import svgpathtools

print('All dependencies installed successfully!')
"

# Check for any missing dependencies
echo "Checking for missing dependencies..."
pip check

# Deactivate virtual environment
deactivate

echo "Installation complete!" 