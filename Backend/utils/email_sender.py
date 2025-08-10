"""
Email Sender Utility with Unified Template System

IMPORTANT: Daily and Weekly Summary Analytics are ADMIN-ONLY
- Daily summaries are sent only to admin emails (ADMIN_ALERT_EMAIL config)
- Weekly reports are sent only to admin emails (ADMIN_ALERT_EMAIL config)
- User-specific summary functions are deprecated and redirect to admin emails
- This ensures analytics data is only accessible to authorized administrators

Supported Email Types:
- Welcome emails (users)
- Payment confirmations (users)
- Subscription notifications (users)
- Admin alerts (admin only)
- Daily summaries (admin only)
- Weekly reports (admin only)
- Security alerts (admin only)
"""

import os
import logging
from datetime import datetime, date, timedelta
from flask import render_template, current_app
from flask_mail import Message, Mail
try:
    from ..app_config import mail
except ImportError:
    try:
        from app_config import mail
    except ImportError:
        from Backend.app_config import mail
try:
    from .analytics_collector import AnalyticsCollector
except ImportError:
    try:
        from utils.analytics_collector import AnalyticsCollector
    except ImportError:
        from Backend.utils.analytics_collector import AnalyticsCollector
import json

logger = logging.getLogger(__name__)

class EmailSender:
    """Email sender utility with unified template system"""
    
    @staticmethod
    def send_welcome_email(user):
        """Send welcome email to new user"""
        try:
            subject = "Welcome to Zyppts! 🎉"
            
            # Get user plan info
            user_plan = user.subscription.plan if user.subscription else 'Free'
            
            html_content = render_template(
                'emails/welcome_email.html',
                user_email=user.email,
                username=user.username,
                user_plan=user_plan,
                sent_date=datetime.now().strftime('%B %d, %Y at %I:%M %p')
            )
            
            msg = Message(
                subject=subject,
                recipients=[user.email],
                html=html_content
            )
            
            mail.send(msg)
            logger.info(f"Welcome email sent to {user.email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send welcome email to {user.email}: {str(e)}")
            return False
    
    @staticmethod
    def send_daily_summary(user, summary_data=None):
        """
        DEPRECATED: Daily summaries are now admin-only.
        This function is kept for backward compatibility but will only send to admin emails.
        """
        import warnings
        warnings.warn(
            "send_daily_summary(user) is deprecated. Daily summaries are now admin-only. "
            "Use send_admin_daily_summary() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Get admin email from config
        from flask import current_app
        try:
            from .email_notifications import get_admin_emails
        except ImportError:
            try:
                from email_notifications import get_admin_emails
            except ImportError:
                try:
                    from Backend.utils.email_notifications import get_admin_emails
                except ImportError:
                    logger.error("Could not import get_admin_emails from any location")
                    return False
        
        admin_emails = get_admin_emails()
        if not admin_emails:
            logger.error("No admin emails configured for daily summary")
            return False
        
        # Redirect to admin function (send to all admin emails)
        success = True
        for admin_email in admin_emails:
            if not EmailSender.send_admin_daily_summary(admin_email, date.today()):
                success = False
        return success
    
    @staticmethod
    def send_weekly_report(user, report_data=None):
        """
        DEPRECATED: Weekly reports are now admin-only.
        This function is kept for backward compatibility but will only send to admin emails.
        """
        import warnings
        warnings.warn(
            "send_weekly_report(user) is deprecated. Weekly reports are now admin-only. "
            "Use send_admin_weekly_report() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Get admin email from config
        from flask import current_app
        try:
            from .email_notifications import get_admin_emails
        except ImportError:
            try:
                from email_notifications import get_admin_emails
            except ImportError:
                try:
                    from Backend.utils.email_notifications import get_admin_emails
                except ImportError:
                    logger.error("Could not import get_admin_emails from any location")
                    return False
        
        admin_emails = get_admin_emails()
        if not admin_emails:
            logger.error("No admin emails configured for weekly report")
            return False
        
        # Get the start of the current week (Monday)
        today = date.today()
        week_start = today - timedelta(days=today.weekday())
        
        # Redirect to admin function (send to all admin emails)
        success = True
        for admin_email in admin_emails:
            if not EmailSender.send_admin_weekly_report(admin_email, week_start):
                success = False
        return success
    
    @staticmethod
    def send_admin_daily_summary(admin_email, report_date=None):
        """Send comprehensive daily summary to admin with live analytics"""
        try:
            subject = f"📊 Daily Summary Report - {report_date.strftime('%B %d, %Y') if report_date else 'Today'}"
            
            # Get comprehensive daily analytics data
            analytics_data = AnalyticsCollector.get_daily_summary_data(report_date)
            if not analytics_data:
                logger.error("Failed to get daily analytics data")
                return False
            
            html_content = render_template(
                'emails/daily_summary.html',
                user_email=admin_email,
                report_date=analytics_data['date'],
                total_uploads=analytics_data['total_uploads'],
                total_variations=analytics_data['total_variations'],
                total_processing_time=analytics_data['total_processing_time'],
                credits_used=analytics_data['total_credits_used'],
                credits_remaining=0,  # Admin view
                user_plan='Admin',
                processing_limit='System Overview',
                recent_uploads=analytics_data['recent_uploads'],
                top_variations=analytics_data['popular_variations'],
                # Additional admin data
                new_users=analytics_data['new_users'],
                new_subscriptions=analytics_data['new_subscriptions'],
                active_subscriptions=analytics_data['active_subscriptions'],
                total_revenue=analytics_data['total_revenue'],
                top_users=analytics_data['top_users'],
                plan_distribution=analytics_data['plan_distribution'],
                sent_date=datetime.now().strftime('%B %d, %Y at %I:%M %p')
            )
            
            msg = Message(
                subject=subject,
                recipients=[admin_email],
                html=html_content
            )
            
            mail.send(msg)
            logger.info(f"Admin daily summary sent to {admin_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send admin daily summary to {admin_email}: {str(e)}")
            return False
    
    @staticmethod
    def send_admin_weekly_report(admin_email, week_start=None):
        """Send comprehensive weekly report to admin with live analytics"""
        try:
            # Get comprehensive weekly analytics data
            analytics_data = AnalyticsCollector.get_weekly_report_data(week_start)
            if not analytics_data:
                logger.error("Failed to get weekly analytics data")
                return False
            
            subject = f"📈 Weekly Report - {analytics_data['week_start']} to {analytics_data['week_end']}"
            
            html_content = render_template(
                'emails/weekly_report.html',
                user_email=admin_email,
                start_date=analytics_data['week_start'],  # Template expects start_date
                end_date=analytics_data['week_end'],      # Template expects end_date
                week_start=analytics_data['week_start'],  # Backup compatibility
                total_uploads=analytics_data['total_uploads'],
                total_variations=analytics_data['total_variations'],
                total_processing_time=analytics_data['total_processing_time'],
                total_credits_used=analytics_data['total_credits_used'],  # Template expects total_credits_used
                credits_used=analytics_data['total_credits_used'],        # Backup compatibility
                credits_remaining=0,  # Admin view
                user_plan='Admin',
                weekly_usage_percentage=0,  # Admin view
                efficiency_score=analytics_data['efficiency_score'],
                daily_breakdown=analytics_data['daily_breakdown'],
                popular_variations=analytics_data['popular_variations'],
                efficiency_tips=[
                    "Monitor system performance and user engagement",
                    "Track revenue trends and subscription growth",
                    "Analyze popular variation types for feature development"
                ],
                # Additional admin data
                new_users=analytics_data['new_users'],
                new_subscriptions=analytics_data['new_subscriptions'],
                active_subscriptions=analytics_data['active_subscriptions'],
                subscription_upgrades=analytics_data['subscription_upgrades'],
                subscription_cancellations=analytics_data['subscription_cancellations'],
                weekly_revenue=analytics_data['total_revenue'],  # Template expects weekly_revenue
                total_revenue=analytics_data['total_revenue'],   # Backup compatibility
                avg_processing_time=analytics_data['avg_processing_time'],
                user_growth=analytics_data['user_growth'],
                top_users=analytics_data['top_users'],
                sent_date=datetime.now().strftime('%B %d, %Y at %I:%M %p')
            )
            
            msg = Message(
                subject=subject,
                recipients=[admin_email],
                html=html_content
            )
            
            mail.send(msg)
            logger.info(f"Admin weekly report sent to {admin_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send admin weekly report to {admin_email}: {str(e)}")
            return False
    
    @staticmethod
    def send_payment_confirmation(user, payment_data):
        """Send payment confirmation email"""
        try:
            subject = "Payment Confirmed - Thank You! ✅"
            
            html_content = render_template(
                'emails/payment_confirmation.html',
                user_email=user.email,
                order_id=payment_data.get('order_id'),
                plan_name=payment_data.get('plan_name', 'Premium'),
                amount=payment_data.get('amount', '0.00'),
                payment_date=payment_data.get('payment_date', datetime.now().strftime('%B %d, %Y')),
                processing_limit=payment_data.get('processing_limit', 'Unlimited'),
                sent_date=datetime.now().strftime('%B %d, %Y at %I:%M %p')
            )
            
            msg = Message(
                subject=subject,
                recipients=[user.email],
                html=html_content
            )
            
            mail.send(msg)
            logger.info(f"Payment confirmation sent to {user.email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send payment confirmation to {user.email}: {str(e)}")
            return False
    
    @staticmethod
    def send_payment_failed(user, payment_data):
        """Send payment failed notification"""
        try:
            subject = "Payment Failed - Action Required ⚠️"
            
            html_content = render_template(
                'emails/payment_failed.html',
                user_email=user.email,
                order_id=payment_data.get('order_id'),
                plan_name=payment_data.get('plan_name', 'Premium'),
                amount=payment_data.get('amount', '0.00'),
                payment_date=payment_data.get('payment_date', datetime.now().strftime('%B %d, %Y')),
                sent_date=datetime.now().strftime('%B %d, %Y at %I:%M %p')
            )
            
            msg = Message(
                subject=subject,
                recipients=[user.email],
                html=html_content
            )
            
            mail.send(msg)
            logger.info(f"Payment failed notification sent to {user.email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send payment failed notification to {user.email}: {str(e)}")
            return False
    
    @staticmethod
    def send_subscription_upgrade(user, upgrade_data):
        """Send subscription upgrade confirmation"""
        try:
            subject = "Subscription Upgraded - Welcome! 🚀"
            
            html_content = render_template(
                'emails/subscription_upgrade.html',
                user_email=user.email,
                previous_plan=upgrade_data.get('previous_plan', 'Free'),
                new_plan=upgrade_data.get('new_plan', 'Premium'),
                upgrade_date=upgrade_data.get('upgrade_date', datetime.now().strftime('%B %d, %Y')),
                next_billing=upgrade_data.get('next_billing', 'Next month'),
                processing_limit=upgrade_data.get('processing_limit', 'Unlimited'),
                sent_date=datetime.now().strftime('%B %d, %Y at %I:%M %p')
            )
            
            msg = Message(
                subject=subject,
                recipients=[user.email],
                html=html_content
            )
            
            mail.send(msg)
            logger.info(f"Subscription upgrade confirmation sent to {user.email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send subscription upgrade to {user.email}: {str(e)}")
            return False
    
    @staticmethod
    def send_account_cancellation(user, cancellation_data):
        """Send account cancellation confirmation"""
        try:
            subject = "Account Cancellation Confirmed 📋"
            
            html_content = render_template(
                'emails/account_cancellation.html',
                user_email=user.email,
                cancelled_plan=cancellation_data.get('cancelled_plan', 'Premium'),
                cancellation_date=cancellation_data.get('cancellation_date', datetime.now().strftime('%B %d, %Y')),
                access_until=cancellation_data.get('access_until', 'End of billing period'),
                cancellation_reason=cancellation_data.get('cancellation_reason', 'Not specified'),
                remaining_days=cancellation_data.get('remaining_days', 0),
                remaining_credits=cancellation_data.get('remaining_credits', 0),
                sent_date=datetime.now().strftime('%B %d, %Y at %I:%M %p')
            )
            
            msg = Message(
                subject=subject,
                recipients=[user.email],
                html=html_content
            )
            
            mail.send(msg)
            logger.info(f"Account cancellation confirmation sent to {user.email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send account cancellation to {user.email}: {str(e)}")
            return False
    
    @staticmethod
    def send_new_account_notification(admin_email, user_data):
        """Send new account notification to admin"""
        try:
            subject = "New User Registration - Zyppts 🎉"
            
            html_content = render_template(
                'emails/new_account_notification.html',
                user_email=admin_email,
                username=user_data.get('username'),
                email=user_data.get('email'),
                account_type=user_data.get('account_type', 'Free'),
                registration_date=user_data.get('registration_date', datetime.now().strftime('%B %d, %Y')),
                referral_source=user_data.get('referral_source', 'Direct'),
                total_users=user_data.get('total_users', 0),
                new_users_today=user_data.get('new_users_today', 0),
                new_users_week=user_data.get('new_users_week', 0),
                user_location=user_data.get('user_location', {}),
                system_load=user_data.get('system_load', 'Normal'),
                db_status=user_data.get('db_status', 'Healthy'),
                email_queue=user_data.get('email_queue', '0 pending'),
                sent_date=datetime.now().strftime('%B %d, %Y at %I:%M %p')
            )
            
            msg = Message(
                subject=subject,
                recipients=[admin_email],
                html=html_content
            )
            
            mail.send(msg)
            logger.info(f"New account notification sent to admin {admin_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send new account notification to {admin_email}: {str(e)}")
            return False
    
    @staticmethod
    def send_test_notification(admin_email, test_data=None):
        """Send test notification email"""
        try:
            subject = "Email System Test - Zyppts ✅"
            
            if not test_data:
                test_data = {}
            
            html_content = render_template(
                'emails/test_notification.html',
                user_email=admin_email,
                test_date=test_data.get('test_date', datetime.now().strftime('%B %d, %Y at %I:%M %p')),
                test_id=test_data.get('test_id', 'TEST-001'),
                sent_date=datetime.now().strftime('%B %d, %Y at %I:%M %p')
            )
            
            msg = Message(
                subject=subject,
                recipients=[admin_email],
                html=html_content
            )
            
            mail.send(msg)
            logger.info(f"Test notification sent to {admin_email}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send test notification to {admin_email}: {str(e)}")
            return False
    
    @staticmethod
    def send_custom_email(recipients, subject, template_name, template_data=None):
        """Send custom email using any template"""
        try:
            if not template_data:
                template_data = {}
            
            # Add default data
            template_data.update({
                'user_email': recipients[0] if isinstance(recipients, list) else recipients,
                'sent_date': datetime.now().strftime('%B %d, %Y at %I:%M %p')
            })
            
            html_content = render_template(
                f'emails/{template_name}.html',
                **template_data
            )
            
            msg = Message(
                subject=subject,
                recipients=recipients if isinstance(recipients, list) else [recipients],
                html=html_content
            )
            
            mail.send(msg)
            logger.info(f"Custom email sent to {recipients}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send custom email to {recipients}: {str(e)}")
            return False 