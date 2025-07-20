"""
Script to update the test user's subscription to have unlimited credits.
Run this script with: python update_test_user.py
"""

from zyppts import create_app, db
from zyppts.models import User, Subscription
from datetime import datetime, timedelta

def update_test_user():
    # Create the Flask application context
    app = create_app()
    
    with app.app_context():
        # Find the test user (assuming username is 'test_user')
        test_user = User.query.filter_by(username='test_user').first()
        
        if not test_user:
            print("Test user not found. Please ensure a user with username 'test_user' exists.")
            return
        
        # Check if the user already has a subscription
        if test_user.subscription:
            subscription = test_user.subscription
            print(f"Updating existing subscription for user: {test_user.username}")
        else:
            # Create a new subscription if one doesn't exist
            subscription = Subscription(user_id=test_user.id)
            db.session.add(subscription)
            print(f"Created new subscription for user: {test_user.username}")
        
        # Set subscription to have unlimited credits
        subscription.plan = 'enterprise'  # Assuming 'enterprise' is the highest plan
        subscription.status = 'active'
        subscription.monthly_credits = 1000000  # Effectively unlimited
        subscription.used_credits = 0
        subscription.start_date = datetime.utcnow()
        subscription.end_date = datetime.utcnow() + timedelta(days=3650)  # 10 years from now
        subscription.auto_renew = True
        
        # Commit the changes
        db.session.commit()
        
        print(f"Successfully updated subscription for {test_user.username}:")
        print(f"- Plan: {subscription.plan}")
        print(f"- Monthly Credits: {subscription.monthly_credits}")
        print(f"- Used Credits: {subscription.used_credits}")
        print(f"- Status: {subscription.status}")
        print(f"- End Date: {subscription.end_date}")

if __name__ == "__main__":
    update_test_user()
