"""
Scheduled Tasks for Admin Email Reports

IMPORTANT: All scheduled analytics reports are ADMIN-ONLY
- Daily summaries are sent only to admin emails (ADMIN_ALERT_EMAIL config)
- Weekly reports are sent only to admin emails (ADMIN_ALERT_EMAIL config)
- No user-specific analytics are sent via scheduled tasks
- This ensures sensitive analytics data is only accessible to authorized administrators

Scheduled Tasks:
- Daily Summary: 9 AM EST daily (admin only)
- Weekly Report: Friday 9 AM EST (admin only)
- Security Cleanup: 2 AM UTC daily (system maintenance)
"""

import os
import logging
from datetime import datetime, timedelta, date
from flask import current_app
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

logger = logging.getLogger(__name__)

# Global scheduler instance
scheduler = None

# Quiet placeholders; imports are resolved lazily to avoid init-time issues
try:
    from .email_sender import EmailSender  # may succeed in some environments
except Exception:
    EmailSender = None

# Helper to lazily resolve EmailSender to avoid module import timing issues
def _resolve_email_sender():
    try:
        if 'EmailSender' in globals() and EmailSender is not None:
            return EmailSender
    except Exception:
        pass
    try:
        from .email_sender import EmailSender as _ES
        return _ES
    except Exception:
        try:
            from utils.email_sender import EmailSender as _ES
            return _ES
        except Exception:
            try:
                from Backend.utils.email_sender import EmailSender as _ES
                return _ES
            except Exception as e:
                logger.error(f"Failed to import EmailSender: {e}")
                return None

# AnalyticsCollector will be imported lazily where needed to avoid init-time issues
AnalyticsCollector = None

def init_scheduler():
    """Initialize the background scheduler"""
    global scheduler
    
    if scheduler is None:
        scheduler = BackgroundScheduler()
        scheduler.start()
        logger.info("Background scheduler started")
    
    return scheduler

def schedule_daily_summary():
    """Schedule daily summary email at 9 AM EST (2 PM UTC)"""
    try:
        scheduler = init_scheduler()
        
        # Schedule for 9 AM EST (2 PM UTC) daily
        trigger = CronTrigger(
            hour=14,  # 2 PM UTC = 9 AM EST
            minute=0,
            timezone=pytz.UTC
        )
        
        job = scheduler.add_job(
            send_daily_summary_with_context,
            trigger=trigger,
            id='daily_summary',
            name='Daily Summary Email (9 AM EST)',
            replace_existing=True
        )
        
        logger.info(f"Daily summary scheduled for 9 AM EST (2 PM UTC)")
        logger.info(f"Next run: {job.next_run_time}")
        
    except Exception as e:
        logger.error(f"Failed to schedule daily summary: {e}")

def schedule_weekly_report():
    """Schedule weekly report email on Friday at 9 AM EST (2 PM UTC)"""
    try:
        scheduler = init_scheduler()
        
        # Schedule for Friday 9 AM EST (2 PM UTC)
        trigger = CronTrigger(
            day_of_week='fri',
            hour=14,  # 2 PM UTC = 9 AM EST
            minute=0,
            timezone=pytz.UTC
        )
        
        job = scheduler.add_job(
            send_weekly_report_with_context,
            trigger=trigger,
            id='weekly_report',
            name='Weekly Report Email (Friday 9 AM EST)',
            replace_existing=True
        )
        
        logger.info(f"Weekly report scheduled for Friday 9 AM EST (Friday 2 PM UTC)")
        logger.info(f"Next run: {job.next_run_time}")
        
    except Exception as e:
        logger.error(f"Failed to schedule weekly report: {e}")

def schedule_security_cleanup():
    """Schedule security log cleanup at 2 AM UTC daily"""
    try:
        scheduler = init_scheduler()
        
        trigger = CronTrigger(
            hour=2,
            minute=0,
            timezone=pytz.UTC
        )
        
        job = scheduler.add_job(
            security_cleanup_with_context,
            trigger=trigger,
            id='security_cleanup',
            name='Security Log Cleanup',
            replace_existing=True
        )
        
        logger.info(f"Security cleanup scheduled for 2 AM UTC")
        logger.info(f"Next run: {job.next_run_time}")
        
    except Exception as e:
        logger.error(f"Failed to schedule security cleanup: {e}")

def send_daily_summary_with_context():
    """Send admin-only daily summary with proper app context"""
    try:
        # Try multiple import paths for different environments
        try:
            from app_config import create_app
        except ImportError:
            try:
                from .app_config import create_app
            except ImportError:
                from Backend.app_config import create_app
        
        app = create_app()
        
        with app.app_context():
            # Resolve AnalyticsCollector lazily
            global AnalyticsCollector
            if AnalyticsCollector is None:
                try:
                    from .analytics_collector import AnalyticsCollector as _AC
                except ImportError:
                    try:
                        from utils.analytics_collector import AnalyticsCollector as _AC
                    except ImportError:
                        from Backend.utils.analytics_collector import AnalyticsCollector as _AC
                AnalyticsCollector = _AC
            # Get admin emails from config
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
            
            # Send admin daily summary with live analytics to all admin emails
            success = True
            es = _resolve_email_sender()
            if es is None:
                logger.error("EmailSender unavailable for daily summary")
                return False
            for admin_email in admin_emails:
                result = es.send_admin_daily_summary(admin_email, date.today())
                if not result:
                    success = False
            
            if success:
                logger.info("✅ Scheduled admin daily summary sent successfully")
            else:
                logger.error("❌ Scheduled admin daily summary failed")
            return success
    except Exception as e:
        logger.error(f"❌ Scheduled admin daily summary failed with exception: {e}")
        return False

def send_weekly_report_with_context():
    """Send admin-only weekly report with proper app context"""
    try:
        # Try multiple import paths for different environments
        try:
            from app_config import create_app
        except ImportError:
            try:
                from .app_config import create_app
            except ImportError:
                from Backend.app_config import create_app
        
        app = create_app()
        
        with app.app_context():
            # Resolve AnalyticsCollector lazily
            global AnalyticsCollector
            if AnalyticsCollector is None:
                try:
                    from .analytics_collector import AnalyticsCollector as _AC
                except ImportError:
                    try:
                        from utils.analytics_collector import AnalyticsCollector as _AC
                    except ImportError:
                        from Backend.utils.analytics_collector import AnalyticsCollector as _AC
                AnalyticsCollector = _AC
            # Get admin emails from config
            try:
                from .email_notifications import get_admin_emails
            except ImportError:
                try:
                    from email_notifications import get_admin_emails
                except ImportError:
                    try:
                        from Backend.utils.email_notifications import get_admin_emails
                    except ImportError:
                        logger.error("Could not import get_admin_emails")
                        return False
            
            admin_emails = get_admin_emails()
            if not admin_emails:
                logger.error("No admin emails configured for weekly report")
                return False
            
            # Get the start of the current week (Monday)
            today = date.today()
            week_start = today - timedelta(days=today.weekday())
            
            # Send admin weekly report with live analytics to all admin emails
            success = True
            es = _resolve_email_sender()
            if es is None:
                logger.error("EmailSender unavailable for weekly report")
                return False
            for admin_email in admin_emails:
                result = es.send_admin_weekly_report(admin_email, week_start)
                if not result:
                    success = False
            
            if success:
                logger.info("✅ Scheduled admin weekly report sent successfully")
            else:
                logger.error("❌ Scheduled admin weekly report failed")
            return success
    except Exception as e:
        logger.error(f"❌ Scheduled admin weekly report failed with exception: {e}")
        return False

def security_cleanup_with_context():
    """Perform security cleanup with proper app context"""
    try:
        try:
            from ..app_config import create_app
        except ImportError:
            try:
                from app_config import create_app
            except ImportError:
                from Backend.app_config import create_app
        app = create_app()
        
        with app.app_context():
            # Clean up old analytics data (older than 90 days)
            cleanup_date = datetime.utcnow() - timedelta(days=90)
            
            try:
                from ..models import UserAnalytics, UserMetrics
            except ImportError:
                try:
                    from models import UserAnalytics, UserMetrics
                except ImportError:
                    from Backend.models import UserAnalytics, UserMetrics
            
            # Clean up old analytics
            deleted_analytics = UserAnalytics.query.filter(
                UserAnalytics.timestamp < cleanup_date
            ).delete()
            
            # Clean up old metrics (older than 30 days)
            cleanup_metrics_date = datetime.utcnow() - timedelta(days=30)
            deleted_metrics = UserMetrics.query.filter(
                UserMetrics.date < cleanup_metrics_date.date()
            ).delete()
            
            # Commit the cleanup
            try:
                from ..app_config import db
            except ImportError:
                try:
                    from app_config import db
                except ImportError:
                    from Backend.app_config import db
            db.session.commit()
            
            logger.info(f"✅ Security cleanup completed: {deleted_analytics} analytics, {deleted_metrics} metrics deleted")
            return True
            
    except Exception as e:
        logger.error(f"❌ Security cleanup failed with exception: {e}")
        return False

def get_scheduled_jobs():
    """Get list of scheduled jobs"""
    try:
        global scheduler
        if scheduler is None:
            scheduler = init_scheduler()
        return scheduler.get_jobs()
    except Exception as e:
        logger.error(f"Failed to get scheduled jobs: {e}")
        return []

def stop_scheduled_tasks():
    """Stop all scheduled tasks"""
    try:
        global scheduler
        if scheduler:
            scheduler.shutdown()
            scheduler = None
            logger.info("Scheduled tasks stopped")
    except Exception as e:
        logger.error(f"Failed to stop scheduled tasks: {e}")

def start_scheduled_tasks():
    """Start all scheduled tasks (admin-only analytics)"""
    try:
        # Initialize scheduler
        scheduler = init_scheduler()
        
        # Schedule all tasks
        schedule_daily_summary()
        schedule_weekly_report()
        schedule_security_cleanup()
        
        logger.info("All scheduled tasks started successfully")
        logger.info("📅 Admin Daily Summary: 9 AM EST daily")
        logger.info("📅 Admin Weekly Report: Friday 9 AM EST")
        logger.info("📅 Security Cleanup: 2 AM UTC daily")
        logger.info("🔒 All analytics reports are admin-only for security")
        
    except Exception as e:
        logger.error(f"Failed to start scheduled tasks: {e}")

def send_manual_daily_summary(admin_email=None, report_date=None):
    """Manually send admin-only daily summary (for testing or immediate sending)"""
    try:
        # Prefer using the existing Flask app context if available (called from admin request)
        try:
            app = current_app._get_current_object()
            use_existing_context = True
        except RuntimeError:
            use_existing_context = False
        
        if not use_existing_context:
            # Fallback: import factory and create a new app context
            try:
                from ..app_config import create_app
            except ImportError:
                try:
                    from app_config import create_app
                except ImportError:
                    from Backend.app_config import create_app
            app = create_app()
            ctx_manager = app.app_context()
        else:
            # Use current_app's context manager
            ctx_manager = app.app_context()
        
        with ctx_manager:
            if not admin_email:
                # Get admin emails from config
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
                    logger.error("No admin emails configured for manual daily summary")
                    return False
                
                # Send to all admin emails
                success = True
                es = _resolve_email_sender()
                if es is None:
                    logger.error("EmailSender unavailable for manual daily summary")
                    return False
                for email in admin_emails:
                    result = es.send_admin_daily_summary(email, report_date)
                    if not result:
                        success = False
                
                if success:
                    logger.info(f"✅ Manual admin daily summary sent to {len(admin_emails)} admin emails")
                else:
                    logger.error(f"❌ Manual admin daily summary failed for some admin emails")
                
                return success
            else:
                # Send to specific email
                es = _resolve_email_sender()
                if es is None:
                    logger.error("EmailSender unavailable for manual daily summary (single)")
                    return False
                result = es.send_admin_daily_summary(admin_email, report_date)
                
                if result:
                    logger.info(f"✅ Manual admin daily summary sent to {admin_email}")
                else:
                    logger.error(f"❌ Manual admin daily summary failed for {admin_email}")
                
                return result
            
    except Exception as e:
        logger.error(f"❌ Manual admin daily summary failed with exception: {e}")
        return False

def send_manual_weekly_report(admin_email=None, week_start=None):
    """Manually send admin-only weekly report (for testing or immediate sending)"""
    try:
        # Prefer using the existing Flask app context if available (called from admin request)
        try:
            app = current_app._get_current_object()
            use_existing_context = True
        except RuntimeError:
            use_existing_context = False
        
        if not use_existing_context:
            # Fallback: import factory and create a new app context
            try:
                from ..app_config import create_app
            except ImportError:
                try:
                    from app_config import create_app
                except ImportError:
                    from Backend.app_config import create_app
            app = create_app()
            ctx_manager = app.app_context()
        else:
            # Use current_app's context manager
            ctx_manager = app.app_context()
        
        with ctx_manager:
            if not admin_email:
                # Get admin emails from config
                try:
                    from .email_notifications import get_admin_emails
                except ImportError:
                    try:
                        from email_notifications import get_admin_emails
                    except ImportError:
                        logger.error("Could not import get_admin_emails")
                        return False
                
                admin_emails = get_admin_emails()
                if not admin_emails:
                    logger.error("No admin emails configured for manual weekly report")
                    return False
                
                # Send to all admin emails
                success = True
                es = _resolve_email_sender()
                if es is None:
                    logger.error("EmailSender unavailable for manual weekly report")
                    return False
                for email in admin_emails:
                    result = es.send_admin_weekly_report(email, week_start)
                    if not result:
                        success = False
                
                if success:
                    logger.info(f"✅ Manual admin weekly report sent to {len(admin_emails)} admin emails")
                else:
                    logger.error(f"❌ Manual admin weekly report failed for some admin emails")
                
                return success
            else:
                # Send to specific email
                es = _resolve_email_sender()
                if es is None:
                    logger.error("EmailSender unavailable for manual weekly report (single)")
                    return False
                result = es.send_admin_weekly_report(admin_email, week_start)
                
                if result:
                    logger.info(f"✅ Manual admin weekly report sent to {admin_email}")
                else:
                    logger.error(f"❌ Manual admin weekly report failed for {admin_email}")
                
                return result
            
    except Exception as e:
        logger.error(f"❌ Manual admin weekly report failed with exception: {e}")
        return False

def run_manual_daily_summary():
    """Run manual daily summary - wrapper for admin routes"""
    try:
        return send_manual_daily_summary()
    except Exception as e:
        logger.error(f"Failed to run manual daily summary: {e}")
        return False

def run_manual_weekly_report():
    """Run manual weekly report - wrapper for admin routes"""
    try:
        return send_manual_weekly_report()
    except Exception as e:
        logger.error(f"Failed to run manual weekly report: {e}")
        return False 