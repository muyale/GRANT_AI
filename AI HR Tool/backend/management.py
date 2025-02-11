import sqlite3
from config import HR_DB_PATH, ADDITIONAL_DB_PATH

def mark_employee_leave(employee_id, on_leave_status):
    """Update an employee's leave status in the HR database."""
    conn = sqlite3.connect(HR_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE employees SET on_leave = ? WHERE id = ?", (on_leave_status, employee_id))
    conn.commit()
    conn.close()
    return True

def record_attendance(employee_id, check_in_time, check_out_time, attendance_date):
    """Insert a new attendance record into the additional database."""
    conn = sqlite3.connect(ADDITIONAL_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO attendance_tracking (employee_id, check_in_time, check_out_time, attendance_date)
        VALUES (?, ?, ?, ?)
    """, (employee_id, check_in_time, check_out_time, attendance_date))
    conn.commit()
    conn.close()
    return True
