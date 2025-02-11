import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from models.ai_agent import fetch_employee_data
from models.additional_predictions import fetch_additional_data
import os

st.title("GRANT AI: A HR Task Assignment and Monitoring Tool")
st.image("images/team_image.png", caption="Our Amazing HR Tool")
st.write("Welcome to the GRANT HR Management System. Our vision is a flexible, ML-powered tool that predicts employee suitability for task assignment and provides actionable insights.")

# Sidebar for mode selection
mode = st.sidebar.selectbox("Select Prediction Mode", 
                             ["Task Suitability Prediction", 
                              "Burnout Risk Predictions",
                              "Additional Performance Forecasting",
                              "Leave Management",
                              "Task Assignment"])

# Utility function for visualization
def plot_predictions(df, prediction_col, title):
    plt.figure(figsize=(10, 5))
    plt.barh(df['name'], df[prediction_col], color='skyblue')
    plt.xlabel("Predicted Scores")
    plt.title(title)
    plt.gca().invert_yaxis()
    st.pyplot(plt)

if mode == "Task Suitability Prediction":
    st.header("Task Suitability Prediction")
    
    # Load employee data
    df = fetch_employee_data()
    
    # Filter by available hours
    min_hours = st.number_input("Minimum Available Hours", min_value=0, value=4)
    df_filtered = df[df['hours_available'] >= min_hours]
    
    st.write("Data Preview (first 10 rows):", df_filtered.head(10))
    
    # Features and labels
    feature_columns = ['hours_available', 'task_status', 'on_leave', 'logged_in']
    X = df_filtered[feature_columns]
    y = df_filtered['suitable']
    
    # Train a simple Linear Regression model
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    if st.button("Get Task Suitability"):
        # Generate predictions
        df_filtered['suitability_score'] = model.predict(X_scaled)
        df_filtered = df_filtered.sort_values(by="suitability_score", ascending=False)
        st.dataframe(df_filtered[['name', 'suitability_score']])
        
        # Plot predictions
        plot_predictions(df_filtered, "suitability_score", "Predicted Task Suitability Scores")

elif mode == "Burnout Risk Predictions":
    st.header("Burnout Risk Predictions")

    # Load additional HR data
    df_add = fetch_additional_data()

    st.write("Additional Data Preview (first 50 rows):")
    st.dataframe(df_add.head(50))

    # Burnout prediction logic based on threshold
    df_add['burnout_risk'] = df_add.apply(
        lambda x: "High Risk" if x['work_pattern'] < 5 and x['sentiment_score'] < 0.5 else "Low Risk", axis=1
    )

    # Separate High and Low Risk Employees
    high_risk = df_add[df_add['burnout_risk'] == "High Risk"]
    low_risk = df_add[df_add['burnout_risk'] == "Low Risk"]

    st.subheader("High Burnout Risk Employees")
    if high_risk.empty:
        st.write("No employees at high risk.")
    else:
        st.dataframe(high_risk[['name', 'burnout_risk']].head(50))

    st.subheader("Low Burnout Risk Employees")
    if low_risk.empty:
        st.write("No employees at low risk.")
    else:
        st.dataframe(low_risk[['name', 'burnout_risk']].head(50))

elif mode == "Additional Performance Forecasting":
    st.header("Additional Performance Forecasting")
    
    # Load additional HR data
    df_add = fetch_additional_data()
    
    st.write("Additional Data Preview (first 10 rows):", df_add.head(10))
    
    # Features and labels
    feature_columns = ['burnout_risk', 'sentiment_score', 'work_pattern']
    X = df_add[feature_columns]
    y = df_add['historical_performance']
    
    # Train Linear Regression for prediction
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    if st.button("View Performance Predictions"):
        # Generate predictions
        df_add['predicted_performance'] = model.predict(X_scaled)
        st.dataframe(df_add[['name', 'predicted_performance']])
        
        # Plot predictions
        plot_predictions(df_add, "predicted_performance", "Predicted Future Performance Scores")

elif mode == "Leave Management":
    st.header("Employee Leave Management")
    
    # Load employee data
    df = fetch_employee_data()
    
    # Show current leave status
    st.write("Employee Leave Status:", df[['name', 'on_leave']])
    
    # Mark an employee as on leave
    employee_name = st.selectbox("Select an employee to mark on leave", df['name'])
    if st.button("Mark on Leave"):
        df.loc[df['name'] == employee_name, 'on_leave'] = True
        st.success(f"{employee_name} has been marked on leave.")
    
    # Save updated DataFrame (to mimic persistent storage)
    st.write("Updated Leave Status:", df[['name', 'on_leave']])

elif mode == "Task Assignment":
    st.header("Task Assignment and Employee Availability")

    # Load employee data
    df = fetch_employee_data()

    # Filter by task expertise using skill_tags
    task_type = st.selectbox(
        "Select Task Type", 
        ["Data Analysis", "Machine Learning", "UI/UX Design", "Project Management"]
    )
    
    # Filter available employees based on skill tags
    available_employees = df[
        (df['on_leave'] == 0) &  # Ensure the employee is not on leave
        (df['skill_tags'].str.contains(task_type, case=False, na=False))  # Match task type in skill tags
    ]

    if available_employees.empty:
        st.warning(f"No available employees for {task_type}.")
    else:
        st.write(f"Available Employees for {task_type}:")
        st.dataframe(available_employees[['name', 'skill_tags']])


st.write("---")
st.write("As an HR Manager, you might want to use this tool not only for task assignment but also for identifying potential burnout risks, planning workforce allocation, and making data-driven decisions for promotions and training investments. Future updates will include richer data and more advanced prediction models.")
