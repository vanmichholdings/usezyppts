"""
Database models for the logo processing application.
"""

from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from flask_sqlalchemy import SQLAlchemy

# Create SQLAlchemy instance
db = SQLAlchemy()

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    is_active = db.Column(db.Boolean, default=True)
    is_admin = db.Column(db.Boolean, default=False)  # Add admin flag
    is_beta = db.Column(db.Boolean, default=False)
    
    # Registration metadata for admin tracking
    registration_ip = db.Column(db.String(45))  # IPv6 compatible
    registration_user_agent = db.Column(db.Text)
    
    subscription = db.relationship('Subscription', backref='user', uselist=False)
    uploads = db.relationship('LogoUpload', backref='user', lazy=True)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def get_id(self):
        return str(self.id)
    
    def has_active_subscription(self):
        return self.subscription and self.subscription.is_active()
    
    def can_generate_files(self):
        if not self.subscription:
            return False
        return self.subscription.has_credits() and not self.subscription.is_expired()
    
    def is_studio_plan(self):
        """Check if the user has an active Studio or Enterprise subscription for batch processing."""
        return self.has_active_subscription() and self.subscription.plan in ['studio', 'enterprise']
    
    def is_administrator(self):
        """Check if user is an administrator"""
        return self.is_admin
    
    def __repr__(self):
        return f'<User {self.username}>'

class Subscription(db.Model):
    __tablename__ = 'subscriptions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    plan = db.Column(db.String(20), nullable=False)  # 'free', 'pro', 'studio', 'enterprise'
    status = db.Column(db.String(20), default='active')  # 'active', 'cancelled', 'expired'
    monthly_credits = db.Column(db.Integer, nullable=False)
    used_credits = db.Column(db.Integer, default=0)
    start_date = db.Column(db.DateTime, default=datetime.utcnow)
    end_date = db.Column(db.DateTime)
    last_reset = db.Column(db.DateTime, default=datetime.utcnow)
    auto_renew = db.Column(db.Boolean, default=True)
    payment_id = db.Column(db.String(100))  # For storing payment processor ID
    billing_cycle = db.Column(db.String(20), default='monthly')  # 'monthly' or 'annual'
    
    def is_active(self):
        return self.status == 'active' and (not self.end_date or self.end_date > datetime.utcnow())
    
    def is_expired(self):
        return self.end_date and self.end_date <= datetime.utcnow()
    
    def has_credits(self):
        """Check if subscription has available credits"""
        return self.monthly_credits > self.used_credits
    
    def get_next_billing_date(self):
        """Calculate the next billing date based on account creation"""
        if not self.user or not self.start_date:
            return None
            
        # Get the day of the month from the start date
        billing_day = self.start_date.day
        
        # Calculate next billing date
        today = datetime.utcnow()
        next_billing = today.replace(day=billing_day)
        
        # If the billing day has passed this month, move to next month
        if next_billing < today:
            if next_billing.month == 12:
                next_billing = next_billing.replace(year=next_billing.year + 1, month=1)
            else:
                next_billing = next_billing.replace(month=next_billing.month + 1)
        
        return next_billing
    
    def reset_credits(self):
        """Reset used credits at the start of each billing cycle"""
        self.used_credits = 0
        self.last_reset = datetime.utcnow()
    
    def use_credits(self, amount):
        """Use credits and return True if successful"""
        if self.monthly_credits - self.used_credits >= amount:
            self.used_credits += amount
            return True
        return False
    
    def __repr__(self):
        return f'<Subscription {self.plan} for User {self.user_id}>'

class LogoUpload(db.Model):
    __tablename__ = 'logo_uploads'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    file_size = db.Column(db.Integer)  # in bytes
    file_type = db.Column(db.String(50))
    status = db.Column(db.String(20), default='pending')  # 'pending', 'processed', 'failed'
    preview_url = db.Column(db.String(255))
    variations = db.relationship('LogoVariation', backref='upload', lazy=True)
    
    def __repr__(self):
        return f'<LogoUpload {self.original_filename} by User {self.user_id}>'

class LogoVariation(db.Model):
    __tablename__ = 'logo_variations'
    
    id = db.Column(db.Integer, primary_key=True)
    upload_id = db.Column(db.Integer, db.ForeignKey('logo_uploads.id'), nullable=False)
    variation_type = db.Column(db.String(50), nullable=False)  # 'transparent', 'black', 'white', etc.
    file_path = db.Column(db.String(255), nullable=False)
    file_size = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_watermarked = db.Column(db.Boolean, default=False)
    
    def __repr__(self):
        return f'<LogoVariation {self.variation_type} for Upload {self.upload_id}>'


# Analytics Models for Admin Dashboard
class UserAnalytics(db.Model):
    __tablename__ = 'user_analytics'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    action_type = db.Column(db.String(50), nullable=False)  # 'login', 'upload', 'process', 'download'
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    details = db.Column(db.Text)  # JSON string for additional data
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    
    user = db.relationship('User', backref='analytics')
    
    def __repr__(self):
        return f'<UserAnalytics {self.action_type} by User {self.user_id}>'


class UserSession(db.Model):
    __tablename__ = 'user_sessions'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    session_id = db.Column(db.String(255), unique=True, nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow)
    end_time = db.Column(db.DateTime)
    ip_address = db.Column(db.String(45))
    user_agent = db.Column(db.Text)
    is_active = db.Column(db.Boolean, default=True)
    
    user = db.relationship('User', backref='sessions')
    
    def __repr__(self):
        return f'<UserSession {self.session_id} for User {self.user_id}>'


class UserMetrics(db.Model):
    __tablename__ = 'user_metrics'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    uploads_count = db.Column(db.Integer, default=0)
    processing_time_total = db.Column(db.Float, default=0.0)  # in seconds
    files_processed = db.Column(db.Integer, default=0)
    variations_generated = db.Column(db.Integer, default=0)
    credits_used = db.Column(db.Integer, default=0)
    
    user = db.relationship('User', backref='metrics')
    
    class Meta:
        unique_together = ('user_id', 'date')
    
    def __repr__(self):
        return f'<UserMetrics {self.date} for User {self.user_id}>'


class SubscriptionAnalytics(db.Model):
    __tablename__ = 'subscription_analytics'
    
    id = db.Column(db.Integer, primary_key=True)
    subscription_id = db.Column(db.Integer, db.ForeignKey('subscriptions.id'), nullable=False)
    date = db.Column(db.Date, nullable=False)
    credits_used = db.Column(db.Integer, default=0)
    uploads_processed = db.Column(db.Integer, default=0)
    revenue_generated = db.Column(db.Float, default=0.0)
    
    subscription = db.relationship('Subscription', backref='analytics')
    
    class Meta:
        unique_together = ('subscription_id', 'date')
    
    def __repr__(self):
        return f'<SubscriptionAnalytics {self.date} for Subscription {self.subscription_id}>'