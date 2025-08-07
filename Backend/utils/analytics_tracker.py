"""
Analytics tracking utilities for admin dashboard
"""

import json
from datetime import datetime, date
from flask import request, current_app
from app_config import db
from models import UserAnalytics, UserSession, UserMetrics, LogoUpload, LogoVariation
import logging

logger = logging.getLogger(__name__)

def track_user_action(user_id, action_type, details=None):
    """Track a user action for analytics"""
    try:
        # Get request info if available, otherwise use defaults
        try:
            ip_address = request.remote_addr
            user_agent = request.headers.get('User-Agent', '')
        except RuntimeError:
            # Working outside of request context
            ip_address = 'unknown'
            user_agent = 'system'
        
        analytics = UserAnalytics(
            user_id=user_id,
            action_type=action_type,
            timestamp=datetime.utcnow(),
            details=json.dumps(details) if details else None,
            ip_address=ip_address,
            user_agent=user_agent
        )
        db.session.add(analytics)
        db.session.commit()
        logger.info(f"Tracked action: {action_type} for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to track user action: {e}")
        db.session.rollback()
        return False

def track_user_session(user_id, session_id, start=True):
    """Track user session start/end"""
    try:
        # Get request info if available, otherwise use defaults
        try:
            ip_address = request.remote_addr
            user_agent = request.headers.get('User-Agent', '')
        except RuntimeError:
            # Working outside of request context
            ip_address = 'unknown'
            user_agent = 'system'
        
        if start:
            # End any existing active sessions
            UserSession.query.filter_by(
                user_id=user_id, 
                is_active=True
            ).update({'is_active': False, 'end_time': datetime.utcnow()})
            
            # Start new session
            session = UserSession(
                user_id=user_id,
                session_id=session_id,
                start_time=datetime.utcnow(),
                ip_address=ip_address,
                user_agent=user_agent,
                is_active=True
            )
            db.session.add(session)
        else:
            # End session
            session = UserSession.query.filter_by(
                user_id=user_id,
                session_id=session_id,
                is_active=True
            ).first()
            if session:
                session.is_active = False
                session.end_time = datetime.utcnow()
        
        db.session.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to track user session: {e}")
        db.session.rollback()
        return False

def track_upload(user_id, filename, original_filename, file_size, file_type, status='pending'):
    """Track a file upload"""
    try:
        upload = LogoUpload(
            user_id=user_id,
            filename=filename,
            original_filename=original_filename,
            upload_date=datetime.utcnow(),
            file_size=file_size,
            file_type=file_type,
            status=status
        )
        db.session.add(upload)
        db.session.commit()
        
        # Track the action
        track_user_action(user_id, 'upload', {
            'filename': original_filename,
            'file_size': file_size,
            'file_type': file_type,
            'upload_id': upload.id
        })
        
        # Update daily metrics
        update_daily_metrics(user_id, 'uploads_count', 1)
        
        logger.info(f"Tracked upload: {original_filename} for user {user_id}")
        return upload
    except Exception as e:
        logger.error(f"Failed to track upload: {e}")
        db.session.rollback()
        return None

def track_variation(upload_id, variation_type, file_path, file_size):
    """Track a generated variation"""
    try:
        variation = LogoVariation(
            upload_id=upload_id,
            variation_type=variation_type,
            file_path=file_path,
            file_size=file_size,
            created_at=datetime.utcnow()
        )
        db.session.add(variation)
        db.session.commit()
        
        # Get user_id from upload
        upload = LogoUpload.query.get(upload_id)
        if upload:
            # Update daily metrics
            update_daily_metrics(upload.user_id, 'variations_generated', 1)
        
        logger.info(f"Tracked variation: {variation_type} for upload {upload_id}")
        return variation
    except Exception as e:
        logger.error(f"Failed to track variation: {e}")
        db.session.rollback()
        return None

def update_daily_metrics(user_id, metric_type, increment=1):
    """Update daily metrics for a user"""
    try:
        today = date.today()
        metrics = UserMetrics.query.filter_by(
            user_id=user_id,
            date=today
        ).first()
        
        if not metrics:
            metrics = UserMetrics(
                user_id=user_id,
                date=today,
                uploads_count=0,
                processing_time_total=0.0,
                files_processed=0,
                variations_generated=0,
                credits_used=0
            )
            db.session.add(metrics)
        
        # Update the specific metric
        if metric_type == 'uploads_count':
            metrics.uploads_count = (metrics.uploads_count or 0) + increment
        elif metric_type == 'variations_generated':
            metrics.variations_generated = (metrics.variations_generated or 0) + increment
        elif metric_type == 'files_processed':
            metrics.files_processed = (metrics.files_processed or 0) + increment
        elif metric_type == 'credits_used':
            metrics.credits_used = (metrics.credits_used or 0) + increment
        elif metric_type == 'processing_time_total':
            metrics.processing_time_total = (metrics.processing_time_total or 0.0) + increment
        
        db.session.commit()
        return True
    except Exception as e:
        logger.error(f"Failed to update daily metrics: {e}")
        db.session.rollback()
        return False

def track_processing_completion(user_id, processing_time, files_processed, variations_generated, credits_used):
    """Track processing completion metrics"""
    try:
        # Track the action
        track_user_action(user_id, 'process', {
            'processing_time': processing_time,
            'files_processed': files_processed,
            'variations_generated': variations_generated,
            'credits_used': credits_used
        })
        
        # Update daily metrics
        update_daily_metrics(user_id, 'processing_time_total', processing_time)
        update_daily_metrics(user_id, 'files_processed', files_processed)
        update_daily_metrics(user_id, 'variations_generated', variations_generated)
        update_daily_metrics(user_id, 'credits_used', credits_used)
        
        logger.info(f"Tracked processing completion for user {user_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to track processing completion: {e}")
        return False

def get_user_activity_summary(user_id, days=30):
    """Get activity summary for a user"""
    try:
        from datetime import timedelta
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Get analytics data
        analytics = UserAnalytics.query.filter_by(user_id=user_id).filter(
            UserAnalytics.timestamp >= start_date
        ).order_by(UserAnalytics.timestamp.desc()).all()
        
        # Get metrics data
        metrics = UserMetrics.query.filter_by(user_id=user_id).filter(
            UserMetrics.date >= start_date.date()
        ).all()
        
        # Get uploads
        uploads = LogoUpload.query.filter_by(user_id=user_id).filter(
            LogoUpload.upload_date >= start_date
        ).order_by(LogoUpload.upload_date.desc()).all()
        
        return {
            'analytics': analytics,
            'metrics': metrics,
            'uploads': uploads,
            'total_actions': len(analytics),
            'total_uploads': len(uploads),
            'total_variations': sum(m.variations_generated for m in metrics),
            'total_processing_time': sum(m.processing_time_total for m in metrics)
        }
    except Exception as e:
        logger.error(f"Failed to get user activity summary: {e}")
        return None 