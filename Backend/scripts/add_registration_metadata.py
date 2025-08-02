#!/usr/bin/env python3
"""
Add registration metadata fields to users table
"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '../instance/app.db')

def column_exists(cursor, table, column):
    """Check if a column exists in a table"""
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())

def main():
    """Add registration metadata columns to users table"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    print("🔧 Adding registration metadata fields to users table...")
    
    # Add registration_ip column if it doesn't exist
    if not column_exists(cursor, 'users', 'registration_ip'):
        print('Adding registration_ip column...')
        cursor.execute('ALTER TABLE users ADD COLUMN registration_ip VARCHAR(45)')
        conn.commit()
        print('✅ registration_ip column added')
    else:
        print('registration_ip column already exists')
    
    # Add registration_user_agent column if it doesn't exist
    if not column_exists(cursor, 'users', 'registration_user_agent'):
        print('Adding registration_user_agent column...')
        cursor.execute('ALTER TABLE users ADD COLUMN registration_user_agent TEXT')
        conn.commit()
        print('✅ registration_user_agent column added')
    else:
        print('registration_user_agent column already exists')
    
    # Update existing users with placeholder data
    cursor.execute("""
        UPDATE users 
        SET registration_ip = 'Unknown', 
            registration_user_agent = 'Unknown' 
        WHERE registration_ip IS NULL OR registration_user_agent IS NULL
    """)
    conn.commit()
    
    print(f"✅ Updated {cursor.rowcount} existing users with placeholder metadata")
    
    conn.close()
    print("🎉 Migration complete! Email notifications are now ready.")

if __name__ == '__main__':
    main() 