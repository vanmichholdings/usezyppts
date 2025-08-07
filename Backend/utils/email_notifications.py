"""
Email Notification System for Admin Alerts
"""

import os
import logging
from datetime import datetime
from flask import current_app, render_template
from flask_mail import Message
from threading import Thread

# Set up logging
logger = logging.getLogger(__name__)

def send_async_email(app, msg):
    """Send email asynchronously"""
    with app.app_context():
        try:
            from app_config import mail
            mail.send(msg)
            logger.info(f"Email sent successfully: {msg.subject}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")

def send_email(subject, recipients, template, **kwargs):
    """Send email using template"""
    try:
        from app_config import mail
        
        msg = Message(
            subject=subject,
            recipients=recipients,
            sender=current_app.config.get('MAIL_DEFAULT_SENDER', 'Zyppts HQ <noreply@zyppts.com>')
        )
        
        # Render HTML template
        msg.html = render_template(f'emails/{template}.html', **kwargs)
        
        # Send asynchronously
        Thread(target=send_async_email, args=(current_app._get_current_object(), msg)).start()
        
        return True
    except Exception as e:
        logger.error(f"Failed to send email notification: {e}")
        return False

def send_new_account_notification(user):
    """Send notification when new account is created"""
    if not current_app.config.get('ADMIN_ALERT_EMAIL'):
        logger.warning("ADMIN_ALERT_EMAIL not configured, skipping notification")
        return False
    
    try:
        # Get user details
        user_info = {
            'username': user.username,
            'email': user.email,
            'created_at': user.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'ip_address': getattr(user, 'registration_ip', 'Unknown'),
            'user_agent': getattr(user, 'registration_user_agent', 'Unknown')
        }
        
        # Send admin notification
        send_email(
            subject=f"🆕 New User Registration: {user.username}",
            recipients=[current_app.config['ADMIN_ALERT_EMAIL']],
            template='new_account_notification',
            user=user_info,
            timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        )
        
        # Send welcome email to user
        send_welcome_email(user)
        
        logger.info(f"New account notification sent for user: {user.username}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to send new account notification: {e}")
        return False

def send_welcome_email(user):
    """Send welcome email to new user"""
    try:
        send_email(
            subject="🎉 Welcome to Zyppts!",
            recipients=[user.email],
            template='welcome_email',
            username=user.username,
            login_url=current_app.config.get('SITE_URL', 'https://zyppts.com') + '/login'
        )
        logger.info(f"Welcome email sent to: {user.email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send welcome email: {e}")
        return False

def send_payment_confirmation(user, subscription, amount, transaction_id, payment_date):
    """Send payment confirmation email to user"""
    try:
        send_email(
            subject="✅ Payment Confirmed - Zyppts",
            recipients=[user.email],
            template='payment_confirmation',
            user=user,
            subscription=subscription,
            amount=amount,
            transaction_id=transaction_id,
            payment_date=payment_date,
            timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        )
        logger.info(f"Payment confirmation email sent to: {user.email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send payment confirmation email: {e}")
        return False

def send_payment_failed(user, subscription, amount, transaction_id, payment_date, error_message):
    """Send payment failed email to user"""
    try:
        send_email(
            subject="❌ Payment Failed - Zyppts",
            recipients=[user.email],
            template='payment_failed',
            user=user,
            subscription=subscription,
            amount=amount,
            transaction_id=transaction_id,
            payment_date=payment_date,
            error_message=error_message,
            timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        )
        logger.info(f"Payment failed email sent to: {user.email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send payment failed email: {e}")
        return False

def send_subscription_upgrade(user, old_plan, new_plan, old_price, new_price, billing_cycle, 
                            effective_date, next_billing_date, new_features=None):
    """Send subscription upgrade email to user"""
    try:
        send_email(
            subject=f"🚀 Subscription Upgraded to {new_plan.title()} - Zyppts",
            recipients=[user.email],
            template='subscription_upgrade',
            user=user,
            old_plan=old_plan,
            new_plan=new_plan,
            old_price=old_price,
            new_price=new_price,
            billing_cycle=billing_cycle,
            effective_date=effective_date,
            next_billing_date=next_billing_date,
            new_features=new_features or [],
            timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        )
        logger.info(f"Subscription upgrade email sent to: {user.email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send subscription upgrade email: {e}")
        return False

def send_account_cancellation(user, subscription, cancellation_date, access_until_date, cancellation_reason=None):
    """Send account cancellation email to user"""
    try:
        send_email(
            subject="📝 Subscription Cancelled - Zyppts",
            recipients=[user.email],
            template='account_cancellation',
            user=user,
            subscription=subscription,
            cancellation_date=cancellation_date,
            access_until_date=access_until_date,
            cancellation_reason=cancellation_reason,
            timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        )
        logger.info(f"Account cancellation email sent to: {user.email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send account cancellation email: {e}")
        return False

def send_subscription_notification(user, subscription, action):
    """Send notification for subscription changes"""
    if not current_app.config.get('ADMIN_ALERT_EMAIL'):
        return False
    
    try:
        send_email(
            subject=f"💳 Subscription {action.title()}: {user.username}",
            recipients=[current_app.config['ADMIN_ALERT_EMAIL']],
            template='subscription_notification',
            user=user,
            subscription=subscription,
            action=action,
            timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        )
        return True
    except Exception as e:
        logger.error(f"Failed to send subscription notification: {e}")
        return False

def send_upload_notification(user, upload):
    """Send notification for new uploads"""
    if not current_app.config.get('ADMIN_ALERT_EMAIL'):
        return False
    
    try:
        send_email(
            subject=f"📁 New Upload: {user.username}",
            recipients=[current_app.config['ADMIN_ALERT_EMAIL']],
            template='upload_notification',
            user=user,
            upload=upload,
            timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        )
        return True
    except Exception as e:
        logger.error(f"Failed to send upload notification: {e}")
        return False

def send_security_alert(alert_type, details):
    """Send security alert notifications"""
    if not current_app.config.get('ADMIN_ALERT_EMAIL'):
        return False
    
    try:
        send_email(
            subject=f"🚨 Security Alert: {alert_type}",
            recipients=[current_app.config['ADMIN_ALERT_EMAIL']],
            template='security_alert',
            alert_type=alert_type,
            details=details,
            timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        )
        return True
    except Exception as e:
        logger.error(f"Failed to send security alert: {e}")
        return False

def send_daily_summary():
    """Send daily summary to admin"""
    if not current_app.config.get('ADMIN_ALERT_EMAIL'):
        return False
    
    try:
        from models import User, Subscription, LogoUpload
        from datetime import datetime, timedelta
        
        # Get yesterday's date
        yesterday = datetime.utcnow() - timedelta(days=1)
        
        # Get statistics
        new_users = User.query.filter(
            User.created_at >= yesterday
        ).count()
        
        new_subscriptions = Subscription.query.filter(
            Subscription.start_date >= yesterday
        ).count()
        
        new_uploads = LogoUpload.query.filter(
            LogoUpload.upload_date >= yesterday
        ).count()
        
        # Get top users
        top_users = User.query.order_by(User.created_at.desc()).limit(5).all()
        
        send_email(
            subject=f"📊 Daily Summary - {yesterday.strftime('%Y-%m-%d')}",
            recipients=[current_app.config['ADMIN_ALERT_EMAIL']],
            template='daily_summary',
            date=yesterday.strftime('%Y-%m-%d'),
            new_users=new_users,
            new_subscriptions=new_subscriptions,
            new_uploads=new_uploads,
            top_users=top_users,
            timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        )
        return True
    except Exception as e:
        logger.error(f"Failed to send daily summary: {e}")
        return False

def send_weekly_report():
    """Send weekly report to admin"""
    if not current_app.config.get('ADMIN_ALERT_EMAIL'):
        return False
    
    try:
        from models import User, Subscription, LogoUpload
        from datetime import datetime, timedelta
        
        # Get last week's date
        last_week = datetime.utcnow() - timedelta(weeks=1)
        
        # Get statistics
        new_users = User.query.filter(
            User.created_at >= last_week
        ).count()
        
        active_subscriptions = Subscription.query.filter(
            Subscription.status == 'active'
        ).count()
        
        total_uploads = LogoUpload.query.filter(
            LogoUpload.upload_date >= last_week
        ).count()
        
        # Get user growth trend
        user_growth = []
        for i in range(7):
            date = last_week + timedelta(days=i)
            count = User.query.filter(
                User.created_at >= date,
                User.created_at < date + timedelta(days=1)
            ).count()
            user_growth.append({'date': date.strftime('%Y-%m-%d'), 'count': count})
        
        send_email(
            subject=f"📈 Weekly Report - {last_week.strftime('%Y-%m-%d')} to {datetime.utcnow().strftime('%Y-%m-%d')}",
            recipients=[current_app.config['ADMIN_ALERT_EMAIL']],
            template='weekly_report',
            start_date=last_week.strftime('%Y-%m-%d'),
            end_date=datetime.utcnow().strftime('%Y-%m-%d'),
            new_users=new_users,
            active_subscriptions=active_subscriptions,
            total_uploads=total_uploads,
            user_growth=user_growth,
            timestamp=datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')
        )
        return True
    except Exception as e:
        logger.error(f"Failed to send weekly report: {e}")
        return False 