import sqlite3
import pandas as pd
from config import HR_DB_PATH, ADDITIONAL_DB_PATH, HRMS_DB_PATH,ADDITIONAL_HR_DB_PATH


def fetch_employee_data():
    """
    Fetch employee data from the main HR database.
    """
    conn = sqlite3.connect(HR_DB_PATH)
    query = "SELECT * FROM employees"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def fetch_additional_data():
    """
    Fetch additional predictions from the additional HR database.
    """
    conn = sqlite3.connect(ADDITIONAL_DB_PATH)
    query = "SELECT * FROM additional_predictions"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def fetch_additional_hr_data():
    """
    Fetch additional predictions from the additional HR database.
    """
    conn = sqlite3.connect(ADDITIONAL_HR_DB_PATH)
    query = "SELECT * FROM additional_predictions"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df




def fetch_employee_monitoring_data():
    """
    Fetches employee monitoring data, including department and hours logged.
    """
    try:
        conn = sqlite3.connect(HRMS_DB_PATH)
        query = "SELECT employee_name, department, hours_logged, task_completion_rate FROM employee_monitoring"
        monitoring_data = pd.read_sql_query(query, conn)
        conn.close()
        return monitoring_data
    except Exception as e:
        print("Error fetching employee monitoring data:", e)
        return pd.DataFrame()


def fetch_attendance_data():
    """
    Fetch attendance records from the HRMS database.
    """
    try:
        conn = sqlite3.connect(HRMS_DB_PATH)
        query = "SELECT * FROM attendance_tracking"
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        print("Error fetching attendance data:", e)
        return pd.DataFrame()
