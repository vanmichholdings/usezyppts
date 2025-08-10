# ZYPPTS Environment Setup Guide

## 🎉 **Environment Setup Complete!**

Your ZYPPTS application is now properly configured with a virtual environment and all necessary dependencies.

---

## **✅ What's Been Set Up:**

### **🐍 Virtual Environment**
- **Created:** `venv/` directory in the Backend folder
- **Status:** ✅ Active and working
- **Python Version:** 3.13.3 (compatible)

### **📧 Email System Dependencies**
- **Flask:** 3.1.1 ✅
- **Flask-Mail:** 0.10.0 ✅
- **Flask-Login:** 0.6.3 ✅
- **Flask-SQLAlchemy:** 3.1.1 ✅
- **APScheduler:** 3.10.4 ✅
- **email-validator:** 2.2.0 ✅
- **python-dotenv:** 1.1.1 ✅

### **📧 Email System Status**
- **Configuration:** ✅ Complete
- **Gmail App Password:** ✅ Set up
- **Test Emails:** ✅ All 5 types working
- **Admin Email:** zyppts@gmail.com ✅

---

## **🚀 Quick Start Commands:**

### **1. Activate Virtual Environment**
```bash
cd Backend
source venv/bin/activate
```

### **2. Test Email System**
```bash
python3 scripts/test_all_emails.py
```

### **3. Check Dependencies**
```bash
python3 scripts/check_dependencies.py
```

### **4. Run Flask Application**
```bash
python3 run.py
```

---

## **📦 Dependency Management:**

### **Minimal Dependencies (Email System Only)**
```bash
# Already installed ✅
pip install -r requirements_email_minimal.txt
```

### **Full Dependencies (Complete Application)**
```bash
# Install all dependencies for full functionality
pip install -r requirements.txt
```

### **Individual Package Installation**
```bash
# Install specific packages as needed
pip install Flask Flask-Mail Flask-Login Flask-SQLAlchemy
pip install APScheduler email-validator python-dotenv
```

---

## **🔧 Available Scripts:**

### **Setup Scripts**
- `scripts/setup_environment.py` - Create virtual environment
- `scripts/configure_email.py` - Configure email settings
- `scripts/check_dependencies.py` - Check installed packages

### **Email Testing Scripts**
- `scripts/test_email_simple.py` - Basic email connection test
- `scripts/test_all_emails.py` - Comprehensive email type testing

### **Database Scripts**
- `scripts/init_db.py` - Initialize database
- `scripts/add_registration_metadata.py` - Add user metadata fields

---

## **📧 Email Notification Types:**

### **✅ Working Email Types:**
1. **🆕 New Account Notification** - Admin alert for new registrations
2. **🎉 Welcome Email** - Welcome message for new users
3. **📊 Daily Summary** - Daily statistics report
4. **📈 Weekly Report** - Weekly analytics report
5. **🚨 Security Alert** - Security event notifications

### **📅 Scheduled Tasks:**
- **Daily Summary:** 9 AM UTC daily
- **Weekly Report:** Monday 10 AM UTC
- **Security Cleanup:** 2 AM UTC daily

---

## **🛡️ Security Features:**

### **Email Security**
- **Asynchronous sending** - doesn't block user registration
- **Error handling** - registration continues even if email fails
- **IP tracking** - for security monitoring
- **Audit logging** - all email activities logged

### **Admin Security**
- **IP whitelisting** - restrict admin access
- **Email authorization** - restrict to specific emails
- **Session timeouts** - automatic logout
- **Rate limiting** - prevent brute force attacks

---

## **📊 Admin Dashboard Features:**

### **Available Sections:**
- **📊 Dashboard** - Overview and statistics
- **👥 Users** - User management and details
- **💳 Subscriptions** - Subscription management
- **📈 Analytics** - User activity analytics
- **⚙️ System** - System health and logs
- **📧 Notifications** - Email management and testing

### **Admin Access:**
- **URL:** `/admin`
- **Test User:** test@zyppts.com
- **Admin Button:** Visible in header for admin users

---

## **🔍 Troubleshooting:**

### **Virtual Environment Issues**
```bash
# Recreate virtual environment
python3 scripts/setup_environment.py

# Activate manually
source venv/bin/activate
```

### **Email Issues**
```bash
# Test basic connection
python3 scripts/test_email_simple.py

# Test all email types
python3 scripts/test_all_emails.py

# Check Gmail App Password setup
# See email_setup_guide.md for detailed instructions
```

### **Dependency Issues**
```bash
# Check what's installed
python3 scripts/check_dependencies.py

# Install missing packages
pip install package_name

# Update requirements
pip install -r requirements.txt
```

---

## **🚀 Production Deployment:**

### **For Render Deployment:**
1. **Set environment variables** in Render dashboard
2. **Use PostgreSQL** for production database
3. **Configure Redis** for session management
4. **Set up monitoring** with Sentry and logging

### **Environment Variables:**
```bash
# Email Configuration
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USE_TLS=True
MAIL_USERNAME=zyppts@gmail.com
MAIL_PASSWORD=your-app-password
ADMIN_ALERT_EMAIL=mike@usezyppts.com,zyppts@gmail.com

# Database
DATABASE_URL=postgresql://user:pass@host:port/db

# Security
SECRET_KEY=your-secret-key
ADMIN_SECRET_KEY=your-admin-secret
```

---

## **📋 File Structure:**

```
Backend/
├── venv/                          # Virtual environment ✅
├── requirements.txt               # Full dependencies
├── requirements_email_minimal.txt # Email-only dependencies ✅
├── .env                          # Environment variables ✅
├── scripts/                      # Setup and utility scripts
│   ├── setup_environment.py      # Environment setup ✅
│   ├── check_dependencies.py     # Dependency checker ✅
│   ├── test_all_emails.py        # Email testing ✅
│   └── configure_email.py        # Email configuration ✅
├── utils/                        # Utility modules
│   ├── email_notifications.py    # Email system ✅
│   └── scheduled_tasks.py        # Scheduled tasks ✅
└── templates/                    # Email templates ✅
    └── emails/
        ├── new_account_notification.html
        ├── welcome_email.html
        └── test_notification.html
```

---

## **✅ System Status:**

- **✅ Virtual Environment:** Created and active
- **✅ Email System:** Fully operational
- **✅ Dependencies:** Essential packages installed
- **✅ Configuration:** Environment variables set
- **✅ Testing:** All email types working
- **✅ Documentation:** Complete setup guides available

---

## **🎯 Next Steps:**

1. **✅ Virtual environment** - Created and active
2. **✅ Email system** - Configured and tested
3. **🔄 Full application** - Install complete dependencies when needed
4. **🔄 Database setup** - Initialize when ready
5. **🔄 Admin panel** - Access via `/admin` when app is running

---

## **📞 Support:**

If you encounter any issues:

1. **Check virtual environment** is activated
2. **Verify email configuration** in `.env` file
3. **Test email system** using provided scripts
4. **Check dependencies** with dependency checker
5. **Review logs** for error details

---

**🎉 Your ZYPPTS environment is ready! The email notification system is fully operational and you can now install additional dependencies as needed for the complete application functionality.** 