import sqlite3
import bcrypt
from datetime import datetime
import os
import json
import logging

class Database:
    def __init__(self):
        """Initialize database connection."""
        try:
            # Set up logging
            self.logger = logging.getLogger(__name__)
            
            # Create data directory if it doesn't exist
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)
                self.logger.info(f"Created data directory at {data_dir}")
            
            # Set up database path
            self.db_path = os.path.join(data_dir, 'users.db')
            self.logger.info(f"Database path: {self.db_path}")
            
            # Create database connection
            self.conn = sqlite3.connect(self.db_path)
            self.logger.info("Database connection established")
            
            # Create tables and admin user
            self.create_tables()
            self.create_admin_if_not_exists()
            
        except Exception as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            raise Exception(f"Failed to initialize database: {str(e)}")

    def create_tables(self):
        """Create necessary database tables if they don't exist."""
        cursor = self.conn.cursor()
        
        # Users table
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
        
        # Patients table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            dob DATE,
            assigned_doctor TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (assigned_doctor) REFERENCES users(username)
        )
        ''')
        
        # Measurements table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS measurements (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            heart_rate INTEGER NOT NULL,
            status TEXT NOT NULL,
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
        ''')
        
        # Measurement sessions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS measurement_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT NOT NULL,
            avg_heart_rate REAL NOT NULL,
            status TEXT NOT NULL,
            measurement_date DATE NOT NULL,
            measurement_time TIME NOT NULL,
            duration_seconds REAL NOT NULL,
            total_measurements INTEGER NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
        ''')
        
        self.conn.commit()

    def create_admin_if_not_exists(self):
        """Create default admin user if no users exist."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM users')
            if cursor.fetchone()[0] == 0:
                # Create default admin user (username: admin, password: admin123)
                self.logger.info("Creating default admin user")
                success = self.add_user('admin', 'admin123', 'Administrator')
                if success:
                    self.logger.info("Default admin user created successfully")
                    self.conn.commit()
                else:
                    self.logger.error("Failed to create default admin user")
        except Exception as e:
            self.logger.error(f"Error checking/creating admin user: {str(e)}")

    def add_user(self, username, password, role):
        """Add a new user to the database."""
        try:
            cursor = self.conn.cursor()
            # Hash the password
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            cursor.execute('''
            INSERT INTO users (username, password_hash, role)
            VALUES (?, ?, ?)
            ''', (username, password_hash, role))
            self.conn.commit()
            self.logger.info(f"User {username} added successfully")
            return True
        except sqlite3.IntegrityError:
            self.logger.warning(f"User {username} already exists")
            return False
        except Exception as e:
            self.logger.error(f"Error adding user: {str(e)}")
            return False

    def verify_user(self, username, password):
        """Verify user credentials and update last login."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
            SELECT password_hash, role FROM users
            WHERE username = ?
            ''', (username,))
            result = cursor.fetchone()
            
            if result:
                stored_hash = result[0]
                if isinstance(stored_hash, str):
                    stored_hash = stored_hash.encode('utf-8')
                
                if bcrypt.checkpw(password.encode('utf-8'), stored_hash):
                    # Update last login time
                    cursor.execute('''
                    UPDATE users SET last_login = ? WHERE username = ?
                    ''', (datetime.utcnow().isoformat(), username))
                    self.conn.commit()
                    self.logger.info(f"User {username} logged in successfully")
                    return True, result[1]  # Return success and user role
                else:
                    self.logger.warning(f"Invalid password for user {username}")
            else:
                self.logger.warning(f"User {username} not found")
            
            return False, None
            
        except Exception as e:
            self.logger.error(f"Error verifying user: {str(e)}")
            return False, None

    def get_user_role(self, username):
        """Get user's role."""
        cursor = self.conn.cursor()
        cursor.execute('SELECT role FROM users WHERE username = ?', (username,))
        result = cursor.fetchone()
        return result[0] if result else None

    def add_patient(self, patient_id, first_name, last_name, dob=None, assigned_doctor=None):
        """Add a new patient to the database."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
            INSERT INTO patients (id, first_name, last_name, dob, assigned_doctor)
            VALUES (?, ?, ?, ?, ?)
            ''', (patient_id, first_name, last_name, dob, assigned_doctor))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def get_patient(self, patient_id):
        """Get patient details by ID."""
        cursor = self.conn.cursor()
        cursor.execute('''
        SELECT p.*, json_group_array(
            json_object(
                'timestamp', m.timestamp,
                'heartRate', m.heart_rate,
                'status', m.status
            )
        ) as measurements
        FROM patients p
        LEFT JOIN measurements m ON p.id = m.patient_id
        WHERE p.id = ?
        GROUP BY p.id
        ''', (patient_id,))
        result = cursor.fetchone()
        
        if result:
            patient_data = {
                'id': result[0],
                'firstName': result[1],
                'lastName': result[2],
                'dob': result[3],
                'assignedDoctor': result[4],
                'measurements': json.loads(result[6])
            }
            return patient_data
        return None

    def get_all_patients(self, doctor_username=None):
        """Get all patients, optionally filtered by assigned doctor."""
        cursor = self.conn.cursor()
        if doctor_username:
            cursor.execute('''
            SELECT p.*, json_group_array(
                json_object(
                    'timestamp', m.timestamp,
                    'heartRate', m.heart_rate,
                    'status', m.status
                )
            ) as measurements
            FROM patients p
            LEFT JOIN measurements m ON p.id = m.patient_id
            WHERE p.assigned_doctor = ?
            GROUP BY p.id
            ''', (doctor_username,))
        else:
            cursor.execute('''
            SELECT p.*, json_group_array(
                json_object(
                    'timestamp', m.timestamp,
                    'heartRate', m.heart_rate,
                    'status', m.status
                )
            ) as measurements
            FROM patients p
            LEFT JOIN measurements m ON p.id = m.patient_id
            GROUP BY p.id
            ''')
        
        results = cursor.fetchall()
        patients = []
        for result in results:
            patient_data = {
                'id': result[0],
                'firstName': result[1],
                'lastName': result[2],
                'dob': result[3],
                'assignedDoctor': result[4],
                'measurements': json.loads(result[6])
            }
            patients.append(patient_data)
        return patients

    def add_measurement(self, patient_id, heart_rate, status):
        """Add a new heart rate measurement for a patient."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
            INSERT INTO measurements (patient_id, timestamp, heart_rate, status)
            VALUES (?, ?, ?, ?)
            ''', (patient_id, datetime.utcnow().isoformat(), heart_rate, status))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def add_measurement_session(self, patient_id, avg_heart_rate, status, measurement_date, measurement_time, duration_seconds, total_measurements):
        """Add a measurement session with average heart rate to the database."""
        try:
            cursor = self.conn.cursor()
            
            # Convert datetime objects to strings for SQLite compatibility
            date_str = measurement_date.isoformat() if hasattr(measurement_date, 'isoformat') else str(measurement_date)
            time_str = measurement_time.isoformat() if hasattr(measurement_time, 'isoformat') else str(measurement_time)
            
            cursor.execute('''
            INSERT INTO measurement_sessions (patient_id, avg_heart_rate, status, measurement_date, measurement_time, duration_seconds, total_measurements)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (patient_id, avg_heart_rate, status, date_str, time_str, duration_seconds, total_measurements))
            self.conn.commit()
            self.logger.info(f"Measurement session saved for patient {patient_id}: {avg_heart_rate} BPM avg over {duration_seconds}s")
            return True
        except Exception as e:
            self.logger.error(f"Error adding measurement session: {str(e)}")
            return False

    def get_measurement_sessions(self, patient_id=None):
        """Get measurement sessions, optionally filtered by patient."""
        try:
            cursor = self.conn.cursor()
            if patient_id:
                cursor.execute('''
                SELECT ms.*, p.first_name, p.last_name
                FROM measurement_sessions ms
                JOIN patients p ON ms.patient_id = p.id
                WHERE ms.patient_id = ?
                ORDER BY ms.created_at DESC
                ''', (patient_id,))
            else:
                cursor.execute('''
                SELECT ms.*, p.first_name, p.last_name
                FROM measurement_sessions ms
                JOIN patients p ON ms.patient_id = p.id
                ORDER BY ms.created_at DESC
                ''')
            
            results = cursor.fetchall()
            sessions = []
            for result in results:
                session_data = {
                    'id': result[0],
                    'patient_id': result[1],
                    'avg_heart_rate': result[2],
                    'status': result[3],
                    'measurement_date': result[4],
                    'measurement_time': result[5],
                    'duration_seconds': result[6],
                    'total_measurements': result[7],
                    'created_at': result[8],
                    'patient_name': f"{result[9]} {result[10]}"
                }
                sessions.append(session_data)
            return sessions
        except Exception as e:
            self.logger.error(f"Error getting measurement sessions: {str(e)}")
            return []

    def __del__(self):
        """Close database connection."""
        if hasattr(self, 'conn'):
            self.conn.close() 