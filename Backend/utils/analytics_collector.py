"""
Analytics data collector for accurate live admin reports
"""

import logging
from datetime import datetime, date, timedelta
from sqlalchemy import func, desc, and_
from app_config import db
from models import User, Subscription, LogoUpload, LogoVariation, UserAnalytics, UserMetrics, SubscriptionAnalytics

logger = logging.getLogger(__name__)

class AnalyticsCollector:
    """Collects accurate live analytics data for admin reports"""
    
    @staticmethod
    def get_daily_summary_data(report_date=None):
        """Get comprehensive daily summary data"""
        try:
            if not report_date:
                report_date = date.today()
            
            # Convert to datetime for queries
            start_datetime = datetime.combine(report_date, datetime.min.time())
            end_datetime = datetime.combine(report_date, datetime.max.time())
            
            # User metrics
            new_users = User.query.filter(
                and_(
                    User.created_at >= start_datetime,
                    User.created_at <= end_datetime
                )
            ).count()
            
            # Subscription metrics
            new_subscriptions = Subscription.query.filter(
                and_(
                    Subscription.start_date >= start_datetime,
                    Subscription.start_date <= end_datetime
                )
            ).count()
            
            active_subscriptions = Subscription.query.filter(
                Subscription.status == 'active'
            ).count()
            
            # Upload metrics
            total_uploads = LogoUpload.query.filter(
                and_(
                    LogoUpload.upload_date >= start_datetime,
                    LogoUpload.upload_date <= end_datetime
                )
            ).count()
            
            # Processing metrics
            total_variations = LogoVariation.query.join(LogoUpload).filter(
                and_(
                    LogoUpload.upload_date >= start_datetime,
                    LogoUpload.upload_date <= end_datetime
                )
            ).count()
            
            # Get processing time from UserMetrics
            daily_metrics = UserMetrics.query.filter(
                UserMetrics.date == report_date
            ).all()
            
            total_processing_time = sum(m.processing_time_total or 0 for m in daily_metrics)
            total_credits_used = sum(m.credits_used or 0 for m in daily_metrics)
            
            # User activity
            daily_analytics = UserAnalytics.query.filter(
                and_(
                    UserAnalytics.timestamp >= start_datetime,
                    UserAnalytics.timestamp <= end_datetime
                )
            ).count()
            
            # Top active users for the day
            top_users = db.session.query(
                User.username,
                User.email,
                func.count(UserAnalytics.id).label('activity_count')
            ).join(UserAnalytics).filter(
                and_(
                    UserAnalytics.timestamp >= start_datetime,
                    UserAnalytics.timestamp <= end_datetime
                )
            ).group_by(User.id, User.username, User.email).order_by(
                desc('activity_count')
            ).limit(10).all()
            
            # Recent uploads with details
            recent_uploads = db.session.query(
                LogoUpload.filename,
                LogoUpload.upload_date,
                User.username
            ).join(User).filter(
                and_(
                    LogoUpload.upload_date >= start_datetime,
                    LogoUpload.upload_date <= end_datetime
                )
            ).order_by(LogoUpload.upload_date.desc()).limit(20).all()
            
            # Popular variation types
            popular_variations = db.session.query(
                LogoVariation.variation_type,
                func.count(LogoVariation.id).label('count')
            ).join(LogoUpload).filter(
                and_(
                    LogoUpload.upload_date >= start_datetime,
                    LogoUpload.upload_date <= end_datetime
                )
            ).group_by(LogoVariation.variation_type).order_by(
                desc('count')
            ).limit(5).all()
            
            # Plan distribution
            plan_distribution = db.session.query(
                Subscription.plan,
                func.count(Subscription.id).label('count')
            ).filter(Subscription.status == 'active').group_by(
                Subscription.plan
            ).all()
            
            # Revenue metrics (if available)
            total_revenue = db.session.query(
                func.sum(SubscriptionAnalytics.revenue_generated)
            ).filter(
                SubscriptionAnalytics.date == report_date
            ).scalar() or 0.0
            
            return {
                'date': report_date.strftime('%B %d, %Y'),
                'new_users': new_users,
                'new_subscriptions': new_subscriptions,
                'active_subscriptions': active_subscriptions,
                'total_uploads': total_uploads,
                'total_variations': total_variations,
                'total_processing_time': round(total_processing_time, 2),
                'total_credits_used': total_credits_used,
                'daily_analytics': daily_analytics,
                'total_revenue': round(total_revenue, 2),
                'top_users': [
                    {
                        'username': user.username,
                        'email': user.email,
                        'activity_count': user.activity_count
                    } for user in top_users
                ],
                'recent_uploads': [
                    {
                        'filename': upload.filename,
                        'upload_date': upload.upload_date,
                        'username': upload.username
                    } for upload in recent_uploads
                ],
                'popular_variations': [
                    {
                        'type': variation.variation_type,
                        'count': variation.count
                    } for variation in popular_variations
                ],
                'plan_distribution': [
                    {
                        'plan': plan.plan,
                        'count': plan.count
                    } for plan in plan_distribution
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get daily summary data: {e}")
            return None
    
    @staticmethod
    def get_weekly_report_data(week_start=None):
        """Get comprehensive weekly report data"""
        try:
            if not week_start:
                # Get the start of the current week (Monday)
                today = date.today()
                week_start = today - timedelta(days=today.weekday())
            
            week_end = week_start + timedelta(days=6)
            start_datetime = datetime.combine(week_start, datetime.min.time())
            end_datetime = datetime.combine(week_end, datetime.max.time())
            
            # Weekly user metrics
            new_users = User.query.filter(
                and_(
                    User.created_at >= start_datetime,
                    User.created_at <= end_datetime
                )
            ).count()
            
            # Weekly subscription metrics
            new_subscriptions = Subscription.query.filter(
                and_(
                    Subscription.start_date >= start_datetime,
                    Subscription.start_date <= end_datetime
                )
            ).count()
            
            active_subscriptions = Subscription.query.filter(
                Subscription.status == 'active'
            ).count()
            
            # Weekly upload metrics
            total_uploads = LogoUpload.query.filter(
                and_(
                    LogoUpload.upload_date >= start_datetime,
                    LogoUpload.upload_date <= end_datetime
                )
            ).count()
            
            # Weekly processing metrics
            total_variations = LogoVariation.query.join(LogoUpload).filter(
                and_(
                    LogoUpload.upload_date >= start_datetime,
                    LogoUpload.upload_date <= end_datetime
                )
            ).count()
            
            # Get weekly metrics
            weekly_metrics = UserMetrics.query.filter(
                and_(
                    UserMetrics.date >= week_start,
                    UserMetrics.date <= week_end
                )
            ).all()
            
            total_processing_time = sum(m.processing_time_total or 0 for m in weekly_metrics)
            total_credits_used = sum(m.credits_used or 0 for m in weekly_metrics)
            
            # Daily breakdown for the week
            daily_breakdown = []
            for i in range(7):
                day_date = week_start + timedelta(days=i)
                day_start = datetime.combine(day_date, datetime.min.time())
                day_end = datetime.combine(day_date, datetime.max.time())
                
                day_uploads = LogoUpload.query.filter(
                    and_(
                        LogoUpload.upload_date >= day_start,
                        LogoUpload.upload_date <= day_end
                    )
                ).count()
                
                day_variations = LogoVariation.query.join(LogoUpload).filter(
                    and_(
                        LogoUpload.upload_date >= day_start,
                        LogoUpload.upload_date <= day_end
                    )
                ).count()
                
                daily_breakdown.append({
                    'date': day_date.strftime('%A'),
                    'uploads': day_uploads,
                    'variations': day_variations
                })
            
            # Weekly user activity
            weekly_analytics = UserAnalytics.query.filter(
                and_(
                    UserAnalytics.timestamp >= start_datetime,
                    UserAnalytics.timestamp <= end_datetime
                )
            ).count()
            
            # Top active users for the week
            top_users = db.session.query(
                User.username,
                User.email,
                func.count(UserAnalytics.id).label('activity_count')
            ).join(UserAnalytics).filter(
                and_(
                    UserAnalytics.timestamp >= start_datetime,
                    UserAnalytics.timestamp <= end_datetime
                )
            ).group_by(User.id, User.username, User.email).order_by(
                desc('activity_count')
            ).limit(10).all()
            
            # Popular variation types for the week
            popular_variations = db.session.query(
                LogoVariation.variation_type,
                func.count(LogoVariation.id).label('count')
            ).join(LogoUpload).filter(
                and_(
                    LogoUpload.upload_date >= start_datetime,
                    LogoUpload.upload_date <= end_datetime
                )
            ).group_by(LogoVariation.variation_type).order_by(
                desc('count')
            ).limit(5).all()
            
            # Weekly revenue
            weekly_revenue = db.session.query(
                func.sum(SubscriptionAnalytics.revenue_generated)
            ).filter(
                and_(
                    SubscriptionAnalytics.date >= week_start,
                    SubscriptionAnalytics.date <= week_end
                )
            ).scalar() or 0.0
            
            # User growth trend
            user_growth = []
            for i in range(7):
                day_date = week_start + timedelta(days=i)
                day_start = datetime.combine(day_date, datetime.min.time())
                day_end = datetime.combine(day_date, datetime.max.time())
                
                day_users = User.query.filter(
                    and_(
                        User.created_at >= day_start,
                        User.created_at <= day_end
                    )
                ).count()
                
                user_growth.append({
                    'date': day_date.strftime('%A'),
                    'count': day_users
                })
            
            # Efficiency metrics
            avg_processing_time = total_processing_time / max(total_uploads, 1)
            efficiency_score = 'Excellent' if avg_processing_time < 30 else 'Good' if avg_processing_time < 60 else 'Fair'
            
            return {
                'week_start': week_start.strftime('%B %d, %Y'),
                'week_end': week_end.strftime('%B %d, %Y'),
                'new_users': new_users,
                'new_subscriptions': new_subscriptions,
                'active_subscriptions': active_subscriptions,
                'total_uploads': total_uploads,
                'total_variations': total_variations,
                'total_processing_time': round(total_processing_time, 2),
                'total_credits_used': total_credits_used,
                'weekly_analytics': weekly_analytics,
                'total_revenue': round(weekly_revenue, 2),
                'avg_processing_time': round(avg_processing_time, 2),
                'efficiency_score': efficiency_score,
                'daily_breakdown': daily_breakdown,
                'user_growth': user_growth,
                'top_users': [
                    {
                        'username': user.username,
                        'email': user.email,
                        'activity_count': user.activity_count
                    } for user in top_users
                ],
                'popular_variations': [
                    {
                        'type': variation.variation_type,
                        'count': variation.count
                    } for variation in popular_variations
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to get weekly report data: {e}")
            return None
    
    @staticmethod
    def get_user_specific_data(user_id, days=1):
        """Get user-specific data for personalized reports"""
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=days)
            start_datetime = datetime.combine(start_date, datetime.min.time())
            end_datetime = datetime.combine(end_date, datetime.max.time())
            
            # User's uploads
            user_uploads = LogoUpload.query.filter(
                and_(
                    LogoUpload.user_id == user_id,
                    LogoUpload.upload_date >= start_datetime,
                    LogoUpload.upload_date <= end_datetime
                )
            ).order_by(LogoUpload.upload_date.desc()).all()
            
            # User's variations
            user_variations = LogoVariation.query.join(LogoUpload).filter(
                and_(
                    LogoUpload.user_id == user_id,
                    LogoUpload.upload_date >= start_datetime,
                    LogoUpload.upload_date <= end_datetime
                )
            ).all()
            
            # User's metrics
            user_metrics = UserMetrics.query.filter(
                and_(
                    UserMetrics.user_id == user_id,
                    UserMetrics.date >= start_date
                )
            ).all()
            
            # Calculate totals
            total_uploads = len(user_uploads)
            total_variations = len(user_variations)
            total_processing_time = sum(m.processing_time_total or 0 for m in user_metrics)
            total_credits_used = sum(m.credits_used or 0 for m in user_metrics)
            
            # Recent uploads
            recent_uploads = [
                {
                    'filename': upload.filename,
                    'created_at': upload.upload_date
                } for upload in user_uploads[:10]
            ]
            
            # Popular variations
            variation_counts = {}
            for variation in user_variations:
                variation_counts[variation.variation_type] = variation_counts.get(variation.variation_type, 0) + 1
            
            top_variations = [
                {
                    'type': var_type,
                    'count': count
                } for var_type, count in sorted(variation_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            ]
            
            return {
                'total_uploads': total_uploads,
                'total_variations': total_variations,
                'processing_time': round(total_processing_time, 2),
                'credits_used': total_credits_used,
                'recent_uploads': recent_uploads,
                'top_variations': top_variations
            }
            
        except Exception as e:
            logger.error(f"Failed to get user-specific data: {e}")
            return None 