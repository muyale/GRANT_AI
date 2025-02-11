import sqlite3
import os

# Ensure the "database" directory exists
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_DIR = os.path.join(BASE_DIR, "database")
if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

# Paths from config (alternatively, you can import from config.py)
HR_DB_PATH = os.path.join(DB_DIR, "hr_tool.db")
ADDITIONAL_DB_PATH = os.path.join(DB_DIR, "hr_additional_db.db")

# Initialize primary HR database
conn_hr = sqlite3.connect(HR_DB_PATH)
cursor_hr = conn_hr.cursor()
cursor_hr.execute("""
CREATE TABLE IF NOT EXISTS employees (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    department TEXT,
    hours_available INTEGER,
    skill_tags TEXT,
    task_status TEXT,
    on_leave INTEGER,
    logged_in INTEGER
)
""")
conn_hr.commit()
conn_hr.close()

# Initialize additional database (for performance, attendance, burnout risk, etc.)
conn_add = sqlite3.connect(ADDITIONAL_DB_PATH)
cursor_add = conn_add.cursor()
cursor_add.execute("""
CREATE TABLE IF NOT EXISTS performance_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER,
    performance_score REAL,
    sentiment_score REAL,
    burnout_risk REAL,
    FOREIGN KEY (employee_id) REFERENCES employees (id)
)
""")
cursor_add.execute("""
CREATE TABLE IF NOT EXISTS attendance_tracking (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id INTEGER,
    check_in_time TEXT,
    check_out_time TEXT,
    attendance_date TEXT,
    FOREIGN KEY (employee_id) REFERENCES employees (id)
)
""")
conn_add.commit()
conn_add.close()

print("Databases initialized successfully!")
