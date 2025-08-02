#!/usr/bin/env python3
"""
Setup Virtual Environment and Install Dependencies
"""

import os
import sys
import subprocess
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please use Python 3.8 or higher")
        return False

def create_virtual_environment():
    """Create a virtual environment"""
    venv_path = Path("venv")
    
    if venv_path.exists():
        print("⚠️  Virtual environment already exists")
        response = input("Do you want to recreate it? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            print("🗑️  Removing existing virtual environment...")
            run_command("rm -rf venv", "Remove existing virtual environment")
        else:
            print("✅ Using existing virtual environment")
            return True
    
    return run_command("python3 -m venv venv", "Create virtual environment")

def activate_virtual_environment():
    """Activate the virtual environment"""
    activate_script = Path("venv/bin/activate")
    if not activate_script.exists():
        print("❌ Virtual environment not found")
        return False
    
    print("✅ Virtual environment created successfully")
    print("\n📋 To activate the virtual environment, run:")
    print("   source venv/bin/activate")
    print("\n📋 Then install dependencies with:")
    print("   pip install -r requirements.txt")
    
    return True

def install_minimal_dependencies():
    """Install minimal dependencies for email system"""
    print("\n📦 Installing minimal dependencies for email system...")
    
    minimal_deps = [
        "Flask==3.1.1",
        "Flask-Mail==0.10.0",
        "Flask-Login==0.6.3",
        "Flask-SQLAlchemy==3.1.1",
        "python-dotenv==1.1.1",
        "APScheduler==3.10.4",
        "email-validator>=2.1.0,<3.0.0"
    ]
    
    for dep in minimal_deps:
        if not run_command(f"pip install {dep}", f"Install {dep}"):
            return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 ZYPPTS Environment Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create virtual environment
    if not create_virtual_environment():
        return False
    
    # Activate virtual environment
    if not activate_virtual_environment():
        return False
    
    print("\n🎉 Environment setup completed!")
    print("\n📋 Next steps:")
    print("1. Activate the virtual environment:")
    print("   source venv/bin/activate")
    print("2. Install all dependencies:")
    print("   pip install -r requirements.txt")
    print("3. Or install minimal dependencies for email testing:")
    print("   pip install Flask Flask-Mail Flask-Login Flask-SQLAlchemy python-dotenv APScheduler email-validator")
    print("4. Test the email system:")
    print("   python3 scripts/test_all_emails.py")
    
    return True

if __name__ == "__main__":
    main() 