import sqlite3
def monitor_tasks():
    conn = sqlite3.connect('hr_tool.db')
    cursor = conn.cursor()

    # Fetch all tasks and their status
    cursor.execute('''
    SELECT tasks.task_name, employees.name, tasks.task_status, tasks.deadline 
    FROM tasks
    JOIN employees ON tasks.assigned_to = employees.employee_id
    ''')
    
    results = cursor.fetchall()
    
    if results:
        print("Task Status Monitoring:")
        for task, employee, status, deadline in results:
            print(f"- Task: '{task}' assigned to {employee}, Status: {status}, Deadline: {deadline}")
    else:
        print("No tasks found.")
    
    conn.close()
