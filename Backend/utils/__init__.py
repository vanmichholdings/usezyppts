"""
Utils package for Zyppts
"""

from .logo_processor import LogoProcessor

# Import email notifications
try:
    from .email_notifications import (
        send_new_account_notification,
        send_welcome_email,
        send_payment_confirmation,
        send_payment_failed,
        send_account_cancellation,
        send_daily_summary,
        send_weekly_report,
        send_email
    )
    __all__ = [
        'LogoProcessor',
        'send_new_account_notification',
        'send_welcome_email', 
        'send_payment_confirmation',
        'send_payment_failed',
        'send_account_cancellation',
        'send_daily_summary',
        'send_weekly_report',
        'send_email'
    ]
except ImportError as e:
    print(f"Warning: Could not import email_notifications: {e}")
    __all__ = ['LogoProcessor'] 