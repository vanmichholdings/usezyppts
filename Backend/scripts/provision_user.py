import sys
from datetime import datetime, timedelta
from zyppts import create_app, db
from zyppts.models import User, Subscription

def provision_user(email):
    """Gives a user a 'studio' plan subscription with 1000 credits."""
    app = create_app()
    with app.app_context():
        user = User.query.filter_by(email=email).first()
        if not user:
            print(f"Error: User with email {email} not found.")
            return

        if user.subscription:
            print(f"User {email} already has a subscription. Updating it.")
            sub = user.subscription
        else:
            print(f"Creating a new subscription for {email}.")
            sub = Subscription(user_id=user.id)
            db.session.add(sub)

        sub.plan = 'studio'
        sub.status = 'active'
        sub.monthly_credits = 1000
        sub.used_credits = 0
        sub.start_date = datetime.utcnow()
        sub.end_date = datetime.utcnow() + timedelta(days=30)
        sub.auto_renew = False

        try:
            db.session.commit()
            print(f"Successfully provisioned 'studio' plan for {email}.")
        except Exception as e:
            db.session.rollback()
            print(f"Error provisioning user: {e}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python provision_user.py <user_email>")
        sys.exit(1)
    
    user_email = sys.argv[1]
    provision_user(user_email)
