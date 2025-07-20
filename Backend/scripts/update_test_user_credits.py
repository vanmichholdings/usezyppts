from zyppts import create_app, db
from zyppts.models import User, Subscription
from datetime import datetime, timedelta

def update_test_user_credits():
    app = create_app()
    with app.app_context():
        # Find the test user
        test_user = User.query.filter_by(username='test_user').first()
        
        if not test_user:
            print("Test user not found.")
            return
            
        # Get or create subscription
        subscription = test_user.subscription
        if not subscription:
            subscription = Subscription(user_id=test_user.id)
            db.session.add(subscription)
        
        # Update subscription to have unlimited credits
        subscription.plan = 'enterprise'
        subscription.status = 'active'
        subscription.monthly_credits = 1000000  # Effectively unlimited
        subscription.used_credits = 0
        subscription.start_date = datetime.utcnow()
        subscription.end_date = datetime.utcnow() + timedelta(days=3650)  # 10 years
        subscription.auto_renew = True
        
        db.session.commit()
        
        print(f"Updated subscription for {test_user.username}:")
        print(f"Plan: {subscription.plan}")
        print(f"Monthly Credits: {subscription.monthly_credits}")
        print(f"Used Credits: {subscription.used_credits}")
        print(f"Status: {subscription.status}")
        print(f"End Date: {subscription.end_date}")

if __name__ == "__main__":
    update_test_user_credits()
