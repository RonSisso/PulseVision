import os
import sqlite3
import bcrypt

def init_database():
    """Initialize the database and create admin user."""
    # Get database path
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'users.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL,
        last_login DATETIME,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    # Create admin user if not exists
    cursor.execute('SELECT COUNT(*) FROM users WHERE username = ?', ('admin',))
    if cursor.fetchone()[0] == 0:
        # Create admin user (username: admin, password: admin123)
        password_hash = bcrypt.hashpw('admin123'.encode('utf-8'), bcrypt.gensalt())
        cursor.execute('''
        INSERT INTO users (username, password_hash, role)
        VALUES (?, ?, ?)
        ''', ('admin', password_hash, 'Administrator'))
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_database()
    print("Database initialized successfully!") 