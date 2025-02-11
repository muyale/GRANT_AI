import sqlite3
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from config import HR_DB_PATH, ADDITIONAL_DB_PATH
from models.employee_model import EmployeeTaskModel

def fetch_employee_data():
    """Fetch all employee data from the primary HR database."""
    conn = sqlite3.connect(HR_DB_PATH)
    query = "SELECT * FROM employees"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def fetch_attendance_data():
    """Fetch all attendance records from the additional database."""
    conn = sqlite3.connect(ADDITIONAL_DB_PATH)
    query = "SELECT * FROM attendance_tracking"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def fetch_performance_data():
    """Fetch all performance tracking records from the additional database."""
    conn = sqlite3.connect(ADDITIONAL_DB_PATH)
    query = "SELECT * FROM performance_tracking"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_leave_by_department():
    """Compute and return the count of employees on leave grouped by department."""
    df = fetch_employee_data()
    if df.empty or "department" not in df.columns:
        return pd.DataFrame()
    df_leave = df[df["on_leave"] == 1]
    result = df_leave.groupby("department")["id"].count().reset_index()
    result.rename(columns={"id": "on_leave_count"}, inplace=True)
    return result

def hr_agent():
    """
    Predict task suitability using a trained PyTorch model.
    """
    df = fetch_employee_data()

    if df.empty:
        raise ValueError("No employee data available for predictions.")

    # Encode categorical feature (task_status)
    le_task_status = LabelEncoder()
    df['task_status'] = le_task_status.fit_transform(df['task_status'])
    
    # Compute input features
    features = df[['hours_available', 'task_status', 'on_leave', 'logged_in']].values.astype(float)

    # Load the PyTorch model
    model = EmployeeTaskModel(input_dim=4)
    #model.load_state_dict(torch.load("models/weights/hr_task_model.pth"))
    #model.eval()

    # Make predictions
    with torch.no_grad():
        inputs = torch.tensor(features, dtype=torch.float32)
        outputs = model(inputs).squeeze()

    df['suitability_score'] = outputs.numpy()
    return df[['name', 'suitability_score']].values.tolist()
hr_agent()