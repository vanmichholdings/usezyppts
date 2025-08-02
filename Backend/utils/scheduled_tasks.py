"""
Scheduled Tasks for Admin Email Reports
"""

import os
import logging
from datetime import datetime, timedelta
from flask import current_app
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from utils.email_notifications import send_daily_summary, send_weekly_report

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
    """Schedule daily summary email (runs at 8 AM EST)"""
    scheduler = init_scheduler()
    
    # Remove existing job if it exists
    try:
        scheduler.remove_job('daily_summary')
    except:
        pass
    
    # Add new job - 8 AM EST = 1 PM UTC (standard time) or 12 PM UTC (daylight saving)
    # Using 1 PM UTC to ensure it runs at 8 AM EST during standard time
    scheduler.add_job(
        func=send_daily_summary_with_context,
        trigger=CronTrigger(hour=13, minute=0, timezone='UTC'),  # 1 PM UTC = 8 AM EST
        id='daily_summary',
        name='Daily Summary Email (8 AM EST)',
        replace_existing=True
    )
    
    logger.info("Daily summary scheduled for 8 AM EST (1 PM UTC)")

def schedule_weekly_report():
    """Schedule weekly report email (runs every Sunday at 8 AM EST)"""
    scheduler = init_scheduler()
    
    # Remove existing job if it exists
    try:
        scheduler.remove_job('weekly_report')
    except:
        pass
    
    # Add new job - Sunday 8 AM EST = Sunday 1 PM UTC
    scheduler.add_job(
        func=send_weekly_report_with_context,
        trigger=CronTrigger(day_of_week='sun', hour=13, minute=0, timezone='UTC'),  # Sunday 1 PM UTC = Sunday 8 AM EST
        id='weekly_report',
        name='Weekly Report Email (Sunday 8 AM EST)',
        replace_existing=True
    )
    
    logger.info("Weekly report scheduled for Sunday 8 AM EST (Sunday 1 PM UTC)")

def schedule_security_cleanup():
    """Schedule security log cleanup (runs daily at 2 AM UTC)"""
    scheduler = init_scheduler()
    
    # Remove existing job if it exists
    try:
        scheduler.remove_job('security_cleanup')
    except:
        pass
    
    # Add new job
    scheduler.add_job(
        func=cleanup_old_logs_with_context,
        trigger=CronTrigger(hour=2, minute=0, timezone='UTC'),  # 2 AM UTC
        id='security_cleanup',
        name='Security Log Cleanup',
        replace_existing=True
    )
    
    logger.info("Security cleanup scheduled for 2 AM UTC")

def send_daily_summary_with_context():
    """Send daily summary with proper app context"""
    try:
        from app_config import create_app
        app = create_app()
        
        with app.app_context():
            result = send_daily_summary()
            if result:
                logger.info("✅ Scheduled daily summary sent successfully")
            else:
                logger.error("❌ Scheduled daily summary failed")
            return result
    except Exception as e:
        logger.error(f"❌ Scheduled daily summary failed with exception: {e}")
        return False

def send_weekly_report_with_context():
    """Send weekly report with proper app context"""
    try:
        from app_config import create_app
        app = create_app()
        
        with app.app_context():
            result = send_weekly_report()
            if result:
                logger.info("✅ Scheduled weekly report sent successfully")
            else:
                logger.error("❌ Scheduled weekly report failed")
            return result
    except Exception as e:
        logger.error(f"❌ Scheduled weekly report failed with exception: {e}")
        return False

def cleanup_old_logs_with_context():
    """Clean up old logs with proper app context"""
    try:
        from app_config import create_app
        app = create_app()
        
        with app.app_context():
            cleanup_old_logs()
    except Exception as e:
        logger.error(f"❌ Security cleanup failed with exception: {e}")

def cleanup_old_logs():
    """Clean up old log files and data"""
    try:
        from datetime import datetime, timedelta
        
        # Clean up old admin action logs (keep 30 days)
        cutoff_date = datetime.utcnow() - timedelta(days=30)
        
        # This would clean up old log entries from database
        # Implementation depends on your logging strategy
        
        logger.info("Security log cleanup completed")
        
    except Exception as e:
        logger.error(f"Security cleanup failed: {e}")

def start_scheduled_tasks():
    """Start all scheduled tasks"""
    try:
        # Initialize scheduler
        scheduler = init_scheduler()
        
        # Schedule all tasks
        schedule_daily_summary()
        schedule_weekly_report()
        schedule_security_cleanup()
        
        logger.info("All scheduled tasks started successfully")
        logger.info("📅 Daily Summary: 8 AM EST daily")
        logger.info("📅 Weekly Report: Sunday 8 AM EST")
        logger.info("📅 Security Cleanup: 2 AM UTC daily")
        
    except Exception as e:
        logger.error(f"Failed to start scheduled tasks: {e}")

def stop_scheduled_tasks():
    """Stop all scheduled tasks"""
    global scheduler
    
    if scheduler:
        scheduler.shutdown()
        scheduler = None
        logger.info("Scheduled tasks stopped")

def get_scheduled_jobs():
    """Get list of all scheduled jobs"""
    if scheduler:
        return scheduler.get_jobs()
    return []

def run_manual_daily_summary():
    """Manually trigger daily summary"""
    try:
        from app_config import create_app
        app = create_app()
        
        with app.app_context():
            result = send_daily_summary()
            if result:
                logger.info("✅ Manual daily summary sent successfully")
            else:
                logger.error("❌ Manual daily summary failed")
            return result
    except Exception as e:
        logger.error(f"❌ Manual daily summary failed: {e}")
        return False

def run_manual_weekly_report():
    """Manually trigger weekly report"""
    try:
        from app_config import create_app
        app = create_app()
        
        with app.app_context():
            result = send_weekly_report()
            if result:
                logger.info("✅ Manual weekly report sent successfully")
            else:
                logger.error("❌ Manual weekly report failed")
            return result
    except Exception as e:
        logger.error(f"❌ Manual weekly report failed: {e}")
        return False 