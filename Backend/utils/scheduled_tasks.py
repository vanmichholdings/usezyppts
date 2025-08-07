"""
Scheduled Tasks for Admin Email Reports

IMPORTANT: All scheduled analytics reports are ADMIN-ONLY
- Daily summaries are sent only to admin emails (ADMIN_ALERT_EMAIL config)
- Weekly reports are sent only to admin emails (ADMIN_ALERT_EMAIL config)
- No user-specific analytics are sent via scheduled tasks
- This ensures sensitive analytics data is only accessible to authorized administrators

Scheduled Tasks:
- Daily Summary: 8 AM EST daily (admin only)
- Weekly Report: Sunday 8 AM EST (admin only)
- Security Cleanup: 2 AM UTC daily (system maintenance)
"""

import os
import logging
from datetime import datetime, timedelta, date
from flask import current_app
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from utils.email_sender import EmailSender
from utils.analytics_collector import AnalyticsCollector

logger = logging.getLogger(__name__)

# Global scheduler instance
scheduler = None

def init_scheduler():
    """Initialize the background scheduler"""
    global scheduler
    
    if scheduler is None:
        scheduler = BackgroundScheduler()
        scheduler.start()
        logger.info("Background scheduler started")
    
    return scheduler

def schedule_daily_summary():
    """Schedule daily summary email at 8 AM EST (1 PM UTC)"""
    try:
        scheduler = init_scheduler()
        
        # Schedule for 8 AM EST (1 PM UTC) daily
        trigger = CronTrigger(
            hour=13,  # 1 PM UTC = 8 AM EST
            minute=0,
            timezone=pytz.UTC
        )
        
        job = scheduler.add_job(
            send_daily_summary_with_context,
            trigger=trigger,
            id='daily_summary',
            name='Daily Summary Email (8 AM EST)',
            replace_existing=True
        )
        
        logger.info(f"Daily summary scheduled for 8 AM EST (1 PM UTC)")
        logger.info(f"Next run: {job.next_run_time}")
        
    except Exception as e:
        logger.error(f"Failed to schedule daily summary: {e}")

def schedule_weekly_report():
    """Schedule weekly report email on Sunday at 8 AM EST (1 PM UTC)"""
    try:
        scheduler = init_scheduler()
        
        # Schedule for Sunday 8 AM EST (1 PM UTC)
        trigger = CronTrigger(
            day_of_week='sun',
            hour=13,  # 1 PM UTC = 8 AM EST
            minute=0,
            timezone=pytz.UTC
        )
        
        job = scheduler.add_job(
            send_weekly_report_with_context,
            trigger=trigger,
            id='weekly_report',
            name='Weekly Report Email (Sunday 8 AM EST)',
            replace_existing=True
        )
        
        logger.info(f"Weekly report scheduled for Sunday 8 AM EST (Sunday 1 PM UTC)")
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
        from app_config import create_app
        app = create_app()
        
        with app.app_context():
            # Get admin email from config
            admin_email = current_app.config.get('ADMIN_ALERT_EMAIL')
            if not admin_email:
                logger.error("No admin email configured for daily summary")
                return False
            
            # Send admin daily summary with live analytics
            result = EmailSender.send_admin_daily_summary(admin_email, date.today())
            
            if result:
                logger.info("✅ Scheduled admin daily summary sent successfully")
            else:
                logger.error("❌ Scheduled admin daily summary failed")
            return result
    except Exception as e:
        logger.error(f"❌ Scheduled admin daily summary failed with exception: {e}")
        return False

def send_weekly_report_with_context():
    """Send admin-only weekly report with proper app context"""
    try:
        from app_config import create_app
        app = create_app()
        
        with app.app_context():
            # Get admin email from config
            admin_email = current_app.config.get('ADMIN_ALERT_EMAIL')
            if not admin_email:
                logger.error("No admin email configured for weekly report")
                return False
            
            # Get the start of the current week (Monday)
            today = date.today()
            week_start = today - timedelta(days=today.weekday())
            
            # Send admin weekly report with live analytics
            result = EmailSender.send_admin_weekly_report(admin_email, week_start)
            
            if result:
                logger.info("✅ Scheduled admin weekly report sent successfully")
            else:
                logger.error("❌ Scheduled admin weekly report failed")
            return result
    except Exception as e:
        logger.error(f"❌ Scheduled admin weekly report failed with exception: {e}")
        return False

def security_cleanup_with_context():
    """Perform security cleanup with proper app context"""
    try:
        from app_config import create_app
        app = create_app()
        
        with app.app_context():
            # Clean up old analytics data (older than 90 days)
            cleanup_date = datetime.utcnow() - timedelta(days=90)
            
            from models import UserAnalytics, UserMetrics
            
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
            from app_config import db
            db.session.commit()
            
            logger.info(f"✅ Security cleanup completed: {deleted_analytics} analytics, {deleted_metrics} metrics deleted")
            return True
            
    except Exception as e:
        logger.error(f"❌ Security cleanup failed with exception: {e}")
        return False

def get_scheduled_jobs():
    """Get list of scheduled jobs"""
    try:
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
        logger.info("📅 Admin Daily Summary: 8 AM EST daily")
        logger.info("📅 Admin Weekly Report: Sunday 8 AM EST")
        logger.info("📅 Security Cleanup: 2 AM UTC daily")
        logger.info("🔒 All analytics reports are admin-only for security")
        
    except Exception as e:
        logger.error(f"Failed to start scheduled tasks: {e}")

def send_manual_daily_summary(admin_email=None, report_date=None):
    """Manually send admin-only daily summary (for testing or immediate sending)"""
    try:
        from app_config import create_app
        app = create_app()
        
        with app.app_context():
            if not admin_email:
                admin_email = current_app.config.get('ADMIN_ALERT_EMAIL')
            
            if not admin_email:
                logger.error("No admin email provided for manual daily summary")
                return False
            
            result = EmailSender.send_admin_daily_summary(admin_email, report_date)
            
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
        from app_config import create_app
        app = create_app()
        
        with app.app_context():
            if not admin_email:
                admin_email = current_app.config.get('ADMIN_ALERT_EMAIL')
            
            if not admin_email:
                logger.error("No admin email provided for manual weekly report")
                return False
            
            result = EmailSender.send_admin_weekly_report(admin_email, week_start)
            
            if result:
                logger.info(f"✅ Manual admin weekly report sent to {admin_email}")
            else:
                logger.error(f"❌ Manual admin weekly report failed for {admin_email}")
            
            return result
            
    except Exception as e:
        logger.error(f"❌ Manual admin weekly report failed with exception: {e}")
        return False 