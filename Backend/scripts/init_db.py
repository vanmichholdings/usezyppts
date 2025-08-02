from app_config import create_app
from models import User, Subscription
from datetime import datetime, timedelta

def init_db():
    app = create_app()
    with app.app_context():
        # Create all tables
        from app_config import db
        db.create_all()
        
        try:
            # Create test user if it doesn't exist
            if not User.query.filter_by(email='test@zyppts.com').first():
                test_user = User(
                    username='test_user',
                    email='test@zyppts.com',
                    created_at=datetime.utcnow(),
                    last_login=None,
                    is_active=True,
                    is_beta=False
                )
                test_user.set_password('test123')
                
                # Create a free subscription for test user
                test_subscription = Subscription(
                    user=test_user,
                    plan='free',
                    status='active',
                    monthly_credits=3,
                    start_date=datetime.utcnow(),
                    end_date=datetime.utcnow() + timedelta(days=30)
                )
                
                db.session.add(test_user)
                db.session.add(test_subscription)
                print("Test user created successfully!")
            
            # Create beta user if it doesn't exist
            if not User.query.filter_by(email='beta@zyppts.com').first():
                beta_user = User(
                    username='beta_tester',
                    email='beta@zyppts.com',
                    created_at=datetime.utcnow(),
                    last_login=None,
                    is_active=True,
                    is_beta=True
                )
                beta_user.set_password('beta2024')
                
                # Create a pro subscription for beta user
                beta_subscription = Subscription(
                    user=beta_user,
                    plan='pro',
                    status='active',
                    monthly_credits=500,
                    start_date=datetime.utcnow(),
                    end_date=datetime.utcnow() + timedelta(days=365)
                )
                
                db.session.add(beta_user)
                db.session.add(beta_subscription)
                print("Beta user created successfully!")
            
            db.session.commit()
            print("Database initialized successfully!")
            
        except Exception as e:
            print(f"Error initializing database: {str(e)}")
            db.session.rollback()

if __name__ == '__main__':
    init_db()