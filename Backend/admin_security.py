"""
Admin Security Module - Comprehensive protection for admin routes
"""

import os
import hashlib
import hmac
import time
import logging
from functools import wraps
from flask import request, jsonify, current_app, abort
from datetime import datetime, timedelta

# Security configuration
ADMIN_SECRET_KEY = os.environ.get('ADMIN_SECRET_KEY', 'your-super-secret-admin-key-change-this')
ADMIN_RATE_LIMIT = int(os.environ.get('ADMIN_RATE_LIMIT', 100))  # requests per hour
ADMIN_SESSION_TIMEOUT = int(os.environ.get('ADMIN_SESSION_TIMEOUT', 3600))  # 1 hour
ADMIN_MAX_LOGIN_ATTEMPTS = int(os.environ.get('ADMIN_MAX_LOGIN_ATTEMPTS', 5))
ADMIN_LOCKOUT_DURATION = int(os.environ.get('ADMIN_LOCKOUT_DURATION', 900))  # 15 minutes

# Store failed login attempts
failed_login_attempts = {}

def generate_admin_token(user_id, timestamp):
    """Generate secure admin token"""
    message = f"{user_id}:{timestamp}"
    signature = hmac.new(
        ADMIN_SECRET_KEY.encode(),
        message.encode(),
        hashlib.sha256
    ).hexdigest()
    return f"{user_id}:{timestamp}:{signature}"

def verify_admin_token(token):
    """Verify admin token"""
    try:
        parts = token.split(':')
        if len(parts) != 3:
            return False
        
        user_id, timestamp, signature = parts
        timestamp = int(timestamp)
        
        # Check if token is expired (1 hour)
        if time.time() - timestamp > 3600:
            return False
        
        # Verify signature
        expected_signature = hmac.new(
            ADMIN_SECRET_KEY.encode(),
            f"{user_id}:{timestamp}".encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)
    except:
        return False

def check_rate_limit(identifier, limit=ADMIN_RATE_LIMIT, window=3600):
    """Check rate limiting for admin actions"""
    current_time = time.time()
    window_start = current_time - window
    
    # Get or create rate limit data
    if not hasattr(current_app, 'admin_rate_limits'):
        current_app.admin_rate_limits = {}
    
    if identifier not in current_app.admin_rate_limits:
        current_app.admin_rate_limits[identifier] = []
    
    # Clean old entries
    current_app.admin_rate_limits[identifier] = [
        timestamp for timestamp in current_app.admin_rate_limits[identifier]
        if timestamp > window_start
    ]
    
    # Check if limit exceeded
    if len(current_app.admin_rate_limits[identifier]) >= limit:
        return False
    
    # Add current request
    current_app.admin_rate_limits[identifier].append(current_time)
    return True

def log_security_event(event_type, details=None):
    """Log security events"""
    security_logger = logging.getLogger('admin_security')
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'event_type': event_type,
        'ip_address': request.remote_addr,
        'user_agent': request.headers.get('User-Agent', ''),
        'details': details or {}
    }
    security_logger.warning(f"SECURITY_EVENT: {log_entry}")

def validate_admin_request():
    """Validate admin request headers and parameters"""
    # Check for suspicious headers
    suspicious_headers = [
        'X-Forwarded-For',
        'X-Real-IP',
        'X-Client-IP',
        'CF-Connecting-IP'
    ]
    
    for header in suspicious_headers:
        if header in request.headers:
            log_security_event('SUSPICIOUS_HEADER', {'header': header, 'value': request.headers[header]})
    
    # Check for common attack patterns
    user_agent = request.headers.get('User-Agent', '').lower()
    attack_patterns = [
        'sqlmap', 'nikto', 'nmap', 'w3af', 'burp', 'zap',
        'scanner', 'crawler', 'bot', 'spider'
    ]
    
    for pattern in attack_patterns:
        if pattern in user_agent:
            log_security_event('SUSPICIOUS_USER_AGENT', {'user_agent': user_agent})
            return False
    
    return True

def admin_security_required(f):
    """Enhanced security decorator for admin routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Validate request
        if not validate_admin_request():
            log_security_event('REQUEST_VALIDATION_FAILED')
            abort(403)
        
        # Check rate limiting
        identifier = f"{request.remote_addr}:admin"
        if not check_rate_limit(identifier):
            log_security_event('RATE_LIMIT_EXCEEDED', {'identifier': identifier})
            return jsonify({'error': 'Rate limit exceeded'}), 429
        
        # Add security headers
        response = f(*args, **kwargs)
        if hasattr(response, 'headers'):
            response.headers['X-Content-Type-Options'] = 'nosniff'
            response.headers['X-Frame-Options'] = 'DENY'
            response.headers['X-XSS-Protection'] = '1; mode=block'
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
            response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' cdn.jsdelivr.net cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline' cdn.jsdelivr.net cdnjs.cloudflare.com fonts.googleapis.com; font-src 'self' fonts.gstatic.com; img-src 'self' data:; connect-src 'self'"
            response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
            response.headers['Permissions-Policy'] = 'geolocation=(), microphone=(), camera=()'
        
        return response
    return decorated_function

def check_failed_login_attempts(identifier):
    """Check if IP is locked out due to failed login attempts"""
    if identifier in failed_login_attempts:
        attempts, first_attempt = failed_login_attempts[identifier]
        if attempts >= ADMIN_MAX_LOGIN_ATTEMPTS:
            if time.time() - first_attempt < ADMIN_LOCKOUT_DURATION:
                return False
            else:
                # Reset after lockout duration
                del failed_login_attempts[identifier]
    return True

def record_failed_login_attempt(identifier):
    """Record a failed login attempt"""
    current_time = time.time()
    if identifier in failed_login_attempts:
        attempts, first_attempt = failed_login_attempts[identifier]
        failed_login_attempts[identifier] = (attempts + 1, first_attempt)
    else:
        failed_login_attempts[identifier] = (1, current_time)
    
    log_security_event('FAILED_LOGIN_ATTEMPT', {
        'identifier': identifier,
        'attempts': failed_login_attempts[identifier][0]
    })

def clear_failed_login_attempts(identifier):
    """Clear failed login attempts for successful login"""
    if identifier in failed_login_attempts:
        del failed_login_attempts[identifier]

def sanitize_admin_data(data):
    """Sanitize admin data before logging or storing"""
    if isinstance(data, dict):
        sanitized = {}
        for key, value in data.items():
            if key.lower() in ['password', 'token', 'secret', 'key']:
                sanitized[key] = '***REDACTED***'
            else:
                sanitized[key] = sanitize_admin_data(value)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_admin_data(item) for item in data]
    else:
        return data

def get_admin_audit_log():
    """Get admin audit log entries"""
    try:
        log_file = os.path.join(current_app.root_path, 'logs', 'admin_actions.log')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = f.readlines()[-100:]  # Last 100 lines
                return [line.strip() for line in lines if line.strip()]
        return []
    except Exception as e:
        current_app.logger.error(f"Failed to read admin audit log: {e}")
        return [] 