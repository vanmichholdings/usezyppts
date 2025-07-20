#!/usr/bin/env python3

import os
import sys
import subprocess

def main():
    print("🚀 Starting Zyppts Application...")
    
    # Get project paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    backend_dir = os.path.join(project_root, 'Backend')
    frontend_dir = os.path.join(project_root, 'Frontend')
    
    # Check structure
    if not os.path.exists(backend_dir):
        print("❌ Backend directory not found")
        return
    
    if not os.path.exists(frontend_dir):
        print("❌ Frontend directory not found")
        return
    
    # Activate virtual environment - now in Backend directory
    venv_activate = os.path.join(backend_dir, 'venv', 'bin', 'activate')
    if not os.path.exists(venv_activate):
        print("❌ Virtual environment not found. Please run: python3 -m venv Backend/venv")
        return
    
    # Change to backend directory
    os.chdir(backend_dir)
    
    # Create symbolic links if they don't exist
    templates_link = os.path.join(backend_dir, 'templates')
    static_link = os.path.join(backend_dir, 'static')
    
    if not os.path.exists(templates_link):
        os.symlink(os.path.join(frontend_dir, 'templates'), templates_link)
        print("✅ Created templates symlink")
    
    if not os.path.exists(static_link):
        os.symlink(os.path.join(frontend_dir, 'static'), static_link)
        print("✅ Created static symlink")
    
    # Start the application
    print("🌐 Starting Flask application on http://localhost:5003")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Run the app
    try:
        subprocess.run([sys.executable, 'run.py'], cwd=backend_dir)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 