#!/usr/bin/env python3
"""
Simple test Flask application for Zyppts v10
"""

from flask import Flask, jsonify
import os

# Create a simple Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        'message': 'Zyppts v10 is running!',
        'status': 'success',
        'version': '1.0.0'
    })

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': '2025-08-06T22:46:00Z'
    })

@app.route('/test')
def test():
    return jsonify({
        'message': 'Test endpoint working!',
        'features': [
            'Flask application',
            'JSON responses',
            'Health check',
            'Basic routing'
        ]
    })

if __name__ == '__main__':
    print("=" * 50)
    print("🧪 Zyppts v10 Test Application")
    print("=" * 50)
    print("✅ Starting test server on http://localhost:5003")
    print("📋 Available endpoints:")
    print("   - / (home)")
    print("   - /health (health check)")
    print("   - /test (test endpoint)")
    print("=" * 50)
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    app.run(host='0.0.0.0', port=5003, debug=True) 