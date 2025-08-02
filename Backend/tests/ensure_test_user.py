import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app_config import db, create_app
from models import User, Subscription
from datetime import datetime, timedelta

EMAIL = 'testuser@example.com'
PASSWORD = 'password123'
PLAN = 'studio'
CREDITS = 1000

def ensure_test_user():
    app = create_app()
    with app.app_context():
        user = User.query.filter_by(email=EMAIL).first()
        if not user:
            user = User(email=EMAIL, username='testuser', is_active=True, created_at=datetime.utcnow())
            user.set_password(PASSWORD)
            db.session.add(user)
            db.session.flush()
            print('Created new test user.')
        else:
            user.set_password(PASSWORD)
            print('Test user already exists, password reset.')
        # Ensure subscription
        sub = Subscription.query.filter_by(user_id=user.id).first()
        if not sub:
            sub = Subscription(user_id=user.id, plan=PLAN, status='active', monthly_credits=CREDITS, used_credits=0, start_date=datetime.utcnow(), end_date=datetime.utcnow() + timedelta(days=30))
            db.session.add(sub)
            print('Created new studio subscription.')
        else:
            sub.plan = PLAN
            sub.status = 'active'
            sub.monthly_credits = CREDITS
            sub.used_credits = 0
            sub.start_date = datetime.utcnow()
            sub.end_date = datetime.utcnow() + timedelta(days=30)
            print('Updated existing subscription.')
        db.session.commit()
        print('Test user and subscription ensured.')

if __name__ == '__main__':
    ensure_test_user() 