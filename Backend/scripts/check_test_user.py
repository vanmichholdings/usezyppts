from zyppts import create_app
from zyppts.models import User, Subscription

def check_test_user():
    app = create_app()
    with app.app_context():
        # Find the test user (assuming username is 'test_user')
        test_user = User.query.filter_by(username='test_user').first()
        
        if not test_user:
            print("Test user not found. Do you want to create one? (y/n)")
            if input().lower() == 'y':
                from werkzeug.security import generate_password_hash
                test_user = User(
                    username='test_user',
                    email='test@example.com',
                    password_hash=generate_password_hash('testpassword')
                )
                db.session.add(test_user)
                db.session.commit()
                print("Created test user with username: test_user and password: testpassword")
            else:
                return
        
        print(f"\nTest User Details:")
        print(f"ID: {test_user.id}")
        print(f"Username: {test_user.username}")
        print(f"Email: {test_user.email}")
        
        if test_user.subscription:
            print("\nCurrent Subscription:")
            print(f"Plan: {test_user.subscription.plan}")
            print(f"Monthly Credits: {test_user.subscription.monthly_credits}")
            print(f"Used Credits: {test_user.subscription.used_credits}")
            print(f"Status: {test_user.subscription.status}")
        else:
            print("\nNo subscription found. Would you like to create one with unlimited credits? (y/n)")
            if input().lower() == 'y':
                subscription = Subscription(
                    user_id=test_user.id,
                    plan='enterprise',
                    status='active',
                    monthly_credits=1000000,
                    used_credits=0,
                    auto_renew=True
                )
                db.session.add(subscription)
                db.session.commit()
                print("Created new subscription with unlimited credits")

if __name__ == "__main__":
    check_test_user()
