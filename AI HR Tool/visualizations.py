import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import sqlite3

def plot_leave_by_department():
    conn = sqlite3.connect("database/hr_tool.db")
    query = "SELECT department, COUNT(*) as leave_count FROM employees WHERE on_leave = 1 GROUP BY department"
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        st.warning("No leave data available by department.")
    else:
        fig, ax = plt.subplots()
        ax.bar(df['department'], df['leave_count'], color='skyblue')
        plt.xticks(rotation=45)
        plt.xlabel("Department")
        plt.ylabel("Number of Employees on Leave")
        plt.title("Leave by Department")
        st.pyplot(fig)


def plot_attendance_summary(attendance_data):
    """
    Plot a summary of attendance data using Streamlit.
    """
    try:
        # Example plotting logic
        summary = attendance_data.groupby('status')['employee_name'].count()
        st.bar_chart(summary)
    except Exception as e:
        st.warning(f"Error generating attendance summary plot: {e}")
