#!/usr/bin/env python3

import os
import sys
import subprocess

def main():
    print("🚀 Starting Zyppts with Fixed Structure...")
    
    # Get the project root directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    backend_dir = os.path.join(project_root, 'Backend')
    frontend_dir = os.path.join(project_root, 'Frontend')
    
    # Check if directories exist
    if not os.path.exists(backend_dir):
        print(f"❌ Backend directory not found: {backend_dir}")
        return
    
    if not os.path.exists(frontend_dir):
        print(f"❌ Frontend directory not found: {frontend_dir}")
        return
    
    # Create symbolic links in Backend directory
    os.chdir(backend_dir)
    
    # Create templates symlink
    templates_link = os.path.join(backend_dir, 'templates')
    templates_target = os.path.join(frontend_dir, 'templates')
    
    if os.path.exists(templates_link):
        os.remove(templates_link)
    os.symlink(templates_target, templates_link)
    print(f"✅ Created templates symlink: {templates_link} -> {templates_target}")
    
    # Create static symlink
    static_link = os.path.join(backend_dir, 'static')
    static_target = os.path.join(frontend_dir, 'static')
    
    if os.path.exists(static_link):
        os.remove(static_link)
    os.symlink(static_target, static_link)
    print(f"✅ Created static symlink: {static_link} -> {static_target}")
    
    # Activate virtual environment and run the app
    venv_python = os.path.join(backend_dir, 'venv', 'bin', 'python')
    
    if not os.path.exists(venv_python):
        print("❌ Virtual environment not found. Please run: python3 -m venv Backend/venv")
        return
    
    print("🌐 Starting Flask application...")
    print("💡 The app will be available at: http://localhost:5003")
    print("💡 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Run the Flask app
    try:
        subprocess.run([venv_python, 'run.py'], cwd=backend_dir)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

if __name__ == "__main__":
    main() 