import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from backend.agent import fetch_employee_data, get_leave_by_department, fetch_attendance_data
from backend.management import record_attendance, mark_employee_leave
from models.ai_agent import EmployeeTaskModel, AdditionalPredictionsModel

st.title("Revolutionary AI-Powered HR Management System")
st.write("A one-stop dashboard for employee monitoring, attendance tracking, performance prediction, and more.")

# Sidebar Navigation
st.sidebar.title("Navigation")
view = st.sidebar.selectbox("Select View", 
                            ["Dashboard", "Employee Monitoring", "Attendance Tracking", "HR Predictions"])

if view == "Dashboard":
    st.header("Dashboard")
    st.write("Welcome to the HRMS Dashboard. Use the sidebar to navigate through different modules.")
    
elif view == "Employee Monitoring":
    st.header("Employee Monitoring")
    df_emp = fetch_employee_data()
    st.subheader("Employee Data")
    st.dataframe(df_emp)
    
    st.subheader("Employees on Leave by Department")
    leave_dept = get_leave_by_department()
    if not leave_dept.empty:
        fig, ax = plt.subplots()
        sns.barplot(data=leave_dept, x="department", y="on_leave_count", ax=ax)
        ax.set_title("Employees on Leave by Department")
        st.pyplot(fig)
    else:
        st.info("No leave data available or no department information found.")

elif view == "Attendance Tracking":
    st.header("Attendance Tracking")
    df_attendance = fetch_attendance_data()
    st.subheader("Attendance Records")
    st.dataframe(df_attendance)
    
    st.subheader("Record New Attendance")
    employee_id = st.number_input("Employee ID", min_value=1, step=1)
    check_in = st.text_input("Check-in Time (YYYY-MM-DD HH:MM:SS)")
    check_out = st.text_input("Check-out Time (YYYY-MM-DD HH:MM:SS)")
    att_date = st.text_input("Attendance Date (YYYY-MM-DD)")
    if st.button("Record Attendance"):
        if check_in and check_out and att_date:
            if record_attendance(employee_id, check_in, check_out, att_date):
                st.success("Attendance recorded successfully!")
            else:
                st.error("Failed to record attendance.")
        else:
            st.warning("Please fill in all attendance fields.")

elif view == "HR Predictions":
    st.header("HR Predictions")
    st.write("Future modules for task suitability and burnout risk predictions will be integrated here.")
    st.info("Advanced predictions coming soon!")

st.write("---")
st.write("Revolutionary HRMS – Empowering data-driven HR management with real-time insights.")
