import sqlite3
import os

# Construct the absolute path to the database
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "../database/hr_tool.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# Execute query
cursor.execute("SELECT * FROM employees LIMIT 10;")
rows = cursor.fetchall()

# Print the rows
for row in rows:
    print(row)

conn.close()
