import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), '../instance/app.db')

def column_exists(cursor, table, column):
    cursor.execute(f"PRAGMA table_info({table})")
    return any(row[1] == column for row in cursor.fetchall())

def main():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    # Add is_admin column if it doesn't exist
    if not column_exists(cursor, 'users', 'is_admin'):
        print('Adding is_admin column to users table...')
        cursor.execute('ALTER TABLE users ADD COLUMN is_admin BOOLEAN DEFAULT 0')
        conn.commit()
    else:
        print('is_admin column already exists.')
    # Set is_admin=1 for test user
    cursor.execute("UPDATE users SET is_admin=1 WHERE email='test@zyppts.com'")
    conn.commit()
    print('Set is_admin=1 for test@zyppts.com')
    conn.close()

if __name__ == '__main__':
    main() 