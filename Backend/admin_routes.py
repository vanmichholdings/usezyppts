"""
Administrator routes and dashboard functionality
"""

from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, current_app, session
from flask_login import login_required, current_user
from functools import wraps
from datetime import datetime, date, timedelta
from sqlalchemy import func, desc, and_
import json
import csv
from io import StringIO
import os
import logging

from app_config import db
from models import User, Subscription, LogoUpload, LogoVariation

# Create admin blueprint
admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# Security configuration
ADMIN_IP_WHITELIST = os.environ.get('ADMIN_IP_WHITELIST', '').split(',')  # Comma-separated IPs
ADMIN_ALLOWED_EMAILS = os.environ.get('ADMIN_ALLOWED_EMAILS', 'mike@usezyppts.com,test@zyppts.com').split(',')  # Comma-separated emails
ADMIN_SESSION_TIMEOUT = int(os.environ.get('ADMIN_SESSION_TIMEOUT', 86400))  # 24 hours default

# Set up admin-specific logging
admin_logger = logging.getLogger('admin_actions')
admin_logger.setLevel(logging.INFO)

def log_admin_action(action, details=None, user_id=None):
    """Log admin actions for audit trail"""
    try:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'user_id': user_id or (current_user.id if current_user.is_authenticated else None),
            'username': current_user.username if current_user.is_authenticated else 'unknown',
            'ip_address': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', ''),
            'details': details or {}
        }
        admin_logger.info(f"ADMIN_ACTION: {json.dumps(log_entry)}")
    except Exception as e:
        current_app.logger.error(f"Failed to log admin action: {e}")

def check_ip_whitelist():
    """Check if client IP is in whitelist"""
    if not ADMIN_IP_WHITELIST or not ADMIN_IP_WHITELIST[0]:
        return True  # No whitelist configured, allow all IPs
    
    client_ip = request.remote_addr
    return client_ip in ADMIN_IP_WHITELIST

def check_admin_email():
    """Check if user email is in allowed admin emails"""
    if not current_user.is_authenticated:
        return False
    
    # Temporarily bypass email check for owner account
    if current_user.email == 'mike@usezyppts.com':
        return True
    
    # Get the current admin emails configuration
    admin_emails_env = os.environ.get('ADMIN_ALLOWED_EMAILS')
    if not admin_emails_env:
        # Use default values if no environment variable is set
        admin_emails = 'mike@usezyppts.com,test@zyppts.com'
    else:
        admin_emails = admin_emails_env
    
    allowed_emails = admin_emails.split(',')
    
    # If no email restriction is configured, allow all admin users
    if not allowed_emails or len(allowed_emails) == 0 or not allowed_emails[0]:
        return True
    
    return current_user.email in allowed_emails

def admin_required(f):
    """Enhanced decorator to require admin access with multiple security layers"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if user is authenticated
        if not current_user.is_authenticated:
            log_admin_action('UNAUTHORIZED_ACCESS_ATTEMPT', {'reason': 'not_authenticated'})
            flash('Access denied. Please log in.', 'error')
            return redirect(url_for('main.login'))
        
        # Check if user is admin
        if not current_user.is_administrator():
            log_admin_action('UNAUTHORIZED_ACCESS_ATTEMPT', {
                'reason': 'not_admin',
                'user_id': current_user.id,
                'email': current_user.email
            })
            flash('Access denied. Administrator privileges required.', 'error')
            return redirect(url_for('main.home'))
        
        # Check IP whitelist
        if not check_ip_whitelist():
            log_admin_action('UNAUTHORIZED_ACCESS_ATTEMPT', {
                'reason': 'ip_not_whitelisted',
                'ip': request.remote_addr,
                'user_id': current_user.id
            })
            flash('Access denied. IP address not authorized.', 'error')
            return redirect(url_for('main.home'))
        
        # Check email whitelist
        if not check_admin_email():
            log_admin_action('UNAUTHORIZED_ACCESS_ATTEMPT', {
                'reason': 'email_not_authorized',
                'email': current_user.email,
                'user_id': current_user.id
            })
            flash('Access denied. Email not authorized for admin access.', 'error')
            return redirect(url_for('main.home'))
        
        # Check session timeout
        if 'admin_session_start' in session:
            session_age = datetime.utcnow().timestamp() - session['admin_session_start']
            if session_age > ADMIN_SESSION_TIMEOUT:
                log_admin_action('SESSION_EXPIRED', {'user_id': current_user.id})
                flash('Admin session expired. Please log in again.', 'error')
                return redirect(url_for('main.logout'))
        else:
            # Initialize admin session if not set
            session['admin_session_start'] = datetime.utcnow().timestamp()
            session.permanent = True
        
        # Log successful admin access
        log_admin_action('ADMIN_ACCESS', {'endpoint': request.endpoint})
        
        return f(*args, **kwargs)
    return decorated_function

@admin_bp.before_request
def before_admin_request():
    """Additional security checks before each admin request"""
    # Set admin session start time if not set
    if current_user.is_authenticated and current_user.is_administrator():
        if 'admin_session_start' not in session:
            session['admin_session_start'] = datetime.utcnow().timestamp()
            session.permanent = True
        else:
            # Extend session on each request (sliding window)
            session['admin_session_start'] = datetime.utcnow().timestamp()

@admin_bp.route('/')
@login_required
@admin_required
def dashboard():
    """Main admin dashboard"""
    # Get key metrics
    total_users = User.query.count()
    active_users = User.query.filter_by(is_active=True).count()
    total_subscriptions = Subscription.query.count()
    active_subscriptions = Subscription.query.filter(Subscription.status == 'active').count()
    
    # Recent activity
    recent_users = User.query.order_by(desc(User.created_at)).limit(10).all()
    recent_uploads = LogoUpload.query.order_by(desc(LogoUpload.upload_date)).limit(10).all()
    
    # Revenue metrics (if you have payment data)
    total_revenue = 0  # Calculate from your payment system
    
    # System health
    system_stats = {
        'total_users': total_users,
        'active_users': active_users,
        'total_subscriptions': total_subscriptions,
        'active_subscriptions': active_subscriptions,
        'total_revenue': total_revenue,
        'recent_users': recent_users,
        'recent_uploads': recent_uploads
    }
    
    log_admin_action('DASHBOARD_ACCESSED')
    return render_template('admin/dashboard.html', stats=system_stats)

@admin_bp.route('/users')
@login_required
@admin_required
def users():
    """User management page"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    # Filtering options
    search = request.args.get('search', '')
    status = request.args.get('status', '')
    plan = request.args.get('plan', '')
    
    query = User.query
    
    if search:
        query = query.filter(
            (User.username.contains(search)) |
            (User.email.contains(search))
        )
    
    if status == 'active':
        query = query.filter_by(is_active=True)
    elif status == 'inactive':
        query = query.filter_by(is_active=False)
    
    if plan:
        query = query.join(Subscription).filter(Subscription.plan == plan)
    
    users = query.paginate(page=page, per_page=per_page, error_out=False)
    
    return render_template('admin/users.html', users=users, search=search, status=status, plan=plan)

@admin_bp.route('/user/<int:user_id>')
@login_required
@admin_required
def user_detail(user_id):
    """Detailed user information"""
    user = User.query.get_or_404(user_id)
    
    # Get user's uploads
    uploads = LogoUpload.query.filter_by(user_id=user_id).order_by(desc(LogoUpload.upload_date)).all()
    
    # Get user's subscription history
    subscription = user.subscription
    
    # Get user activity (if analytics tables exist)
    try:
        from models import UserAnalytics, UserSession, UserMetrics
        recent_activity = UserAnalytics.query.filter_by(user_id=user_id).order_by(desc(UserAnalytics.timestamp)).limit(20).all()
        sessions = UserSession.query.filter_by(user_id=user_id).order_by(desc(UserSession.start_time)).limit(10).all()
        metrics = UserMetrics.query.filter_by(user_id=user_id).order_by(desc(UserMetrics.date)).limit(30).all()
    except:
        recent_activity = []
        sessions = []
        metrics = []
    
    return render_template('admin/user_detail.html', 
                         user=user, 
                         uploads=uploads, 
                         subscription=subscription,
                         recent_activity=recent_activity,
                         sessions=sessions,
                         metrics=metrics)

@admin_bp.route('/subscriptions')
@login_required
@admin_required
def subscriptions():
    """Subscription management"""
    page = request.args.get('page', 1, type=int)
    per_page = 20
    
    # Filtering
    status = request.args.get('status', '')
    plan = request.args.get('plan', '')
    
    query = Subscription.query.join(User)
    
    if status:
        query = query.filter(Subscription.status == status)
    if plan:
        query = query.filter(Subscription.plan == plan)
    
    subscriptions = query.paginate(page=page, per_page=per_page, error_out=False)
    
    return render_template('admin/subscriptions.html', subscriptions=subscriptions, status=status, plan=plan)

@admin_bp.route('/analytics')
@login_required
@admin_required
def analytics():
    """Analytics dashboard"""
    # Date range
    days = request.args.get('days', 30, type=int)
    end_date = date.today()
    start_date = end_date - timedelta(days=days)
    
    # User growth
    new_users = User.query.filter(
        User.created_at >= datetime.combine(start_date, datetime.min.time()),
        User.created_at <= datetime.combine(end_date, datetime.max.time())
    ).count()
    
    # Subscription metrics
    active_subscriptions = Subscription.query.filter(Subscription.status == 'active').count()
    plan_distribution = db.session.query(
        Subscription.plan, 
        func.count(Subscription.id)
    ).group_by(Subscription.plan).all()
    
    # Convert plan_distribution to list of dictionaries for JSON serialization
    plan_distribution_data = [{'plan': plan, 'count': count} for plan, count in plan_distribution]
    
    # Upload metrics
    total_uploads = LogoUpload.query.count()
    recent_uploads = LogoUpload.query.filter(
        LogoUpload.upload_date >= datetime.combine(start_date, datetime.min.time())
    ).count()
    
    # Try to get analytics data if tables exist
    try:
        from models import UserAnalytics, UserMetrics, SubscriptionAnalytics
        
        # Daily user activity
        daily_activity = db.session.query(
            func.date(UserAnalytics.timestamp).label('date'),
            func.count(UserAnalytics.id).label('actions')
        ).filter(
            UserAnalytics.timestamp >= datetime.combine(start_date, datetime.min.time())
        ).group_by(func.date(UserAnalytics.timestamp)).all()
        
        # Convert daily_activity to list of dictionaries
        daily_activity_data = [{'date': str(activity.date), 'actions': activity.actions} for activity in daily_activity]
        
        # Top active users
        top_users = db.session.query(
            User.username,
            func.count(UserAnalytics.id).label('activity_count')
        ).join(UserAnalytics).filter(
            UserAnalytics.timestamp >= datetime.combine(start_date, datetime.min.time())
        ).group_by(User.id, User.username).order_by(desc('activity_count')).limit(10).all()
        
        # Convert top_users to list of dictionaries
        top_users_data = [{'username': user.username, 'activity_count': user.activity_count} for user in top_users]
        
    except:
        daily_activity_data = []
        top_users_data = []
    
    analytics_data = {
        'period_days': days,
        'new_users': new_users,
        'active_subscriptions': active_subscriptions,
        'plan_distribution': plan_distribution_data,
        'total_uploads': total_uploads,
        'recent_uploads': recent_uploads,
        'daily_activity': daily_activity_data,
        'top_users': top_users_data
    }
    
    return render_template('admin/analytics.html', data=analytics_data)

@admin_bp.route('/system')
@login_required
@admin_required
def system():
    """System health and monitoring"""
    # Database stats
    db_stats = {
        'total_users': User.query.count(),
        'total_subscriptions': Subscription.query.count(),
        'total_uploads': LogoUpload.query.count(),
        'total_variations': LogoVariation.query.count()
    }
    
    # Recent errors (from logs)
    try:
        with open('Backend/logs/zyppts.log', 'r') as f:
            error_lines = [line for line in f.readlines()[-100:] if 'ERROR' in line]
            recent_errors = error_lines[-10:] if error_lines else []
    except:
        recent_errors = []
    
    # System health
    system_health = {
        'database': 'Healthy',
        'redis': 'Healthy',  # You can add Redis health check
        'disk_space': 'OK',  # Add disk space check
        'memory_usage': 'Normal'  # Add memory check
    }
    
    return render_template('admin/system.html', 
                         db_stats=db_stats, 
                         recent_errors=recent_errors,
                         system_health=system_health)

@admin_bp.route('/export/users')
@login_required
@admin_required
def export_users():
    """Export user data to CSV"""
    users = User.query.all()
    
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['ID', 'Username', 'Email', 'Created', 'Last Login', 'Active', 'Admin', 'Beta', 'Subscription Plan', 'Subscription Status'])
    
    # Write data
    for user in users:
        subscription_info = user.subscription
        plan = subscription_info.plan if subscription_info else 'None'
        status = subscription_info.status if subscription_info else 'None'
        
        writer.writerow([
            user.id,
            user.username,
            user.email,
            user.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            user.last_login.strftime('%Y-%m-%d %H:%M:%S') if user.last_login else 'Never',
            'Yes' if user.is_active else 'No',
            'Yes' if user.is_admin else 'No',
            'Yes' if user.is_beta else 'No',
            plan,
            status
        ])
    
    output.seek(0)
    
    from flask import Response
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=users_{date.today()}.csv'}
    )

@admin_bp.route('/export/subscriptions')
@login_required
@admin_required
def export_subscriptions():
    """Export subscription data to CSV"""
    subscriptions = Subscription.query.join(User).all()
    
    output = StringIO()
    writer = csv.writer(output)
    
    # Write header
    writer.writerow(['User ID', 'Username', 'Email', 'Plan', 'Status', 'Credits', 'Used Credits', 'Start Date', 'End Date', 'Auto Renew'])
    
    # Write data
    for sub in subscriptions:
        writer.writerow([
            sub.user.id,
            sub.user.username,
            sub.user.email,
            sub.plan,
            sub.status,
            sub.monthly_credits,
            sub.used_credits,
            sub.start_date.strftime('%Y-%m-%d %H:%M:%S'),
            sub.end_date.strftime('%Y-%m-%d %H:%M:%S') if sub.end_date else 'None',
            'Yes' if sub.auto_renew else 'No'
        ])
    
    output.seek(0)
    
    from flask import Response
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename=subscriptions_{date.today()}.csv'}
    )

@admin_bp.route('/api/user/<int:user_id>/toggle_admin', methods=['POST'])
@login_required
@admin_required
def toggle_admin(user_id):
    """Toggle admin status for a user"""
    # Add security headers
    response = jsonify({})
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    user = User.query.get_or_404(user_id)
    
    if user.id == current_user.id:
        log_admin_action('ADMIN_TOGGLE_ATTEMPT', {
            'target_user_id': user_id,
            'error': 'self_modification_attempt'
        })
        return jsonify({'error': 'Cannot modify your own admin status'}), 400
    
    old_status = user.is_admin
    user.is_admin = not user.is_admin
    db.session.commit()
    
    log_admin_action('ADMIN_TOGGLE_SUCCESS', {
        'target_user_id': user_id,
        'target_username': user.username,
        'old_status': old_status,
        'new_status': user.is_admin
    })
    
    return jsonify({
        'success': True,
        'is_admin': user.is_admin,
        'message': f'Admin status {"enabled" if user.is_admin else "disabled"} for {user.username}'
    })

@admin_bp.route('/api/user/<int:user_id>/toggle_active', methods=['POST'])
@login_required
@admin_required
def toggle_active(user_id):
    """Toggle active status for a user"""
    # Add security headers
    response = jsonify({})
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    user = User.query.get_or_404(user_id)
    
    if user.id == current_user.id:
        log_admin_action('ACTIVE_TOGGLE_ATTEMPT', {
            'target_user_id': user_id,
            'error': 'self_modification_attempt'
        })
        return jsonify({'error': 'Cannot modify your own active status'}), 400
    
    old_status = user.is_active
    user.is_active = not user.is_active
    db.session.commit()
    
    log_admin_action('ACTIVE_TOGGLE_SUCCESS', {
        'target_user_id': user_id,
        'target_username': user.username,
        'old_status': old_status,
        'new_status': user.is_active
    })
    
    return jsonify({
        'success': True,
        'is_active': user.is_active,
        'message': f'Account {"activated" if user.is_active else "deactivated"} for {user.username}'
    })

@admin_bp.route('/api/subscription/<int:sub_id>/update_credits', methods=['POST'])
@login_required
@admin_required
def update_credits(sub_id):
    """Update subscription credits"""
    # Add security headers
    response = jsonify({})
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    subscription = Subscription.query.get_or_404(sub_id)
    
    data = request.get_json()
    new_credits = data.get('credits', 0)
    
    if new_credits < 0:
        log_admin_action('CREDIT_UPDATE_ATTEMPT', {
            'subscription_id': sub_id,
            'error': 'negative_credits',
            'requested_credits': new_credits
        })
        return jsonify({'error': 'Credits cannot be negative'}), 400
    
    old_credits = subscription.monthly_credits
    subscription.monthly_credits = new_credits
    db.session.commit()
    
    log_admin_action('CREDIT_UPDATE_SUCCESS', {
        'subscription_id': sub_id,
        'user_id': subscription.user.id,
        'username': subscription.user.username,
        'old_credits': old_credits,
        'new_credits': new_credits
    })
    
    return jsonify({
        'success': True,
        'credits': subscription.monthly_credits,
        'message': f'Credits updated to {new_credits} for {subscription.user.username}'
    }) 

@admin_bp.route('/notifications')
@login_required
@admin_required
def notifications():
    """Email notification management"""
    # Get notification settings
    notification_settings = {
        'admin_alert_email': current_app.config.get('ADMIN_ALERT_EMAIL', 'Not configured'),
        'email_enabled': bool(current_app.config.get('ADMIN_ALERT_EMAIL')),
        'daily_summary_enabled': True,
        'weekly_report_enabled': True,
        'new_user_notifications': True,
        'security_alerts': True
    }
    
    # Get scheduled jobs
    try:
        from utils.scheduled_tasks import get_scheduled_jobs
        scheduled_jobs = get_scheduled_jobs()
    except:
        scheduled_jobs = []
    
    return render_template('admin/notifications.html', 
                         settings=notification_settings,
                         scheduled_jobs=scheduled_jobs)

@admin_bp.route('/notifications/test')
@login_required
@admin_required
def test_notification():
    """Test email notification"""
    try:
        from utils.email_notifications import send_email
        
        # Send test email
        send_email(
            subject="🧪 Test Email - Zyppts Admin",
            recipients=[current_app.config.get('ADMIN_ALERT_EMAIL')],
            template='test_notification',
            admin_name=current_user.username,
            timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        )
        
        flash('Test email sent successfully!', 'success')
        log_admin_action('TEST_NOTIFICATION_SENT')
        
    except Exception as e:
        flash(f'Failed to send test email: {str(e)}', 'error')
        log_admin_action('TEST_NOTIFICATION_FAILED', {'error': str(e)})
    
    return redirect(url_for('admin.notifications'))

@admin_bp.route('/notifications/template-preview/<template_name>')
@login_required
@admin_required
def template_preview(template_name):
    """Preview email template"""
    try:
        # Define template data for preview
        preview_data = {
            'test_notification': {
                'admin_name': 'Admin User',
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            },
            'welcome_email': {
                'username': 'TestUser',
                'login_url': 'https://zyppts.com/login'
            },
            'new_account_notification': {
                'user': {
                    'username': 'newuser123',
                    'email': 'newuser@example.com',
                    'created_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                    'ip_address': '192.168.1.100',
                    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                },
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            }
        }
        
        # Get template data
        template_data = preview_data.get(template_name, {})
        
        # Render template preview
        html_content = render_template(f'emails/{template_name}.html', **template_data)
        
        return render_template('admin/template_preview.html', 
                             template_name=template_name,
                             html_content=html_content)
                             
    except Exception as e:
        flash(f'Failed to preview template: {str(e)}', 'error')
        return redirect(url_for('admin.notifications'))

@admin_bp.route('/notifications/send-template-test', methods=['POST'])
@login_required
@admin_required
def send_template_test():
    """Send test email with selected template"""
    try:
        from utils.email_notifications import send_email
        
        template_name = request.form.get('template_name')
        test_email = request.form.get('test_email', current_app.config.get('ADMIN_ALERT_EMAIL'))
        
        if not template_name:
            flash('Template name is required', 'error')
            return redirect(url_for('admin.notifications'))
        
        # Define template data for testing
        test_data = {
            'test_notification': {
                'admin_name': current_user.username,
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            },
            'welcome_email': {
                'username': 'TestUser',
                'login_url': current_app.config.get('SITE_URL', 'https://zyppts.com') + '/login'
            },
            'new_account_notification': {
                'user': {
                    'username': 'testuser123',
                    'email': 'testuser@example.com',
                    'created_at': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                    'ip_address': '127.0.0.1',
                    'user_agent': 'Test Browser'
                },
                'timestamp': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
            }
        }
        
        # Get template data
        template_data = test_data.get(template_name, {})
        
        # Define subjects for each template
        subjects = {
            'test_notification': '🧪 Test Email - Zyppts Admin',
            'welcome_email': '🎉 Welcome to Zyppts!',
            'new_account_notification': '🆕 New User Registration - Test'
        }
        
        subject = subjects.get(template_name, f'Test: {template_name}')
        
        # Send test email
        send_email(
            subject=subject,
            recipients=[test_email],
            template=template_name,
            **template_data
        )
        
        flash(f'Test email sent successfully using {template_name} template!', 'success')
        log_admin_action('TEMPLATE_TEST_SENT', {'template': template_name, 'recipient': test_email})
        
    except Exception as e:
        flash(f'Failed to send test email: {str(e)}', 'error')
        log_admin_action('TEMPLATE_TEST_FAILED', {'error': str(e), 'template': template_name})
    
    return redirect(url_for('admin.notifications'))

@admin_bp.route('/notifications/daily-summary')
@login_required
@admin_required
def send_daily_summary_manual():
    """Manually send daily summary"""
    try:
        from utils.scheduled_tasks import run_manual_daily_summary
        
        if run_manual_daily_summary():
            flash('Daily summary sent successfully!', 'success')
            log_admin_action('MANUAL_DAILY_SUMMARY_SENT')
        else:
            flash('Failed to send daily summary', 'error')
            log_admin_action('MANUAL_DAILY_SUMMARY_FAILED')
            
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        log_admin_action('MANUAL_DAILY_SUMMARY_ERROR', {'error': str(e)})
    
    return redirect(url_for('admin.notifications'))

@admin_bp.route('/notifications/weekly-report')
@login_required
@admin_required
def send_weekly_report_manual():
    """Manually send weekly report"""
    try:
        from utils.scheduled_tasks import run_manual_weekly_report
        
        if run_manual_weekly_report():
            flash('Weekly report sent successfully!', 'success')
            log_admin_action('MANUAL_WEEKLY_REPORT_SENT')
        else:
            flash('Failed to send weekly report', 'error')
            log_admin_action('MANUAL_WEEKLY_REPORT_FAILED')
            
    except Exception as e:
        flash(f'Error: {str(e)}', 'error')
        log_admin_action('MANUAL_WEEKLY_REPORT_ERROR', {'error': str(e)})
    
    return redirect(url_for('admin.notifications')) 