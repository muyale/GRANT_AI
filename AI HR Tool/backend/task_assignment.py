import sqlite3

def assign_task(employee_name, task_name, deadline):
    # Connect to the database
    conn = sqlite3.connect('hr_tool.db')
    cursor = conn.cursor()
    
    # Find the employee's ID
    cursor.execute("SELECT employee_id FROM employees WHERE name = ?", (employee_name,))
    result = cursor.fetchone()
    
    if not result:
        print(f"Employee '{employee_name}' not found.")
        conn.close()
        return
    
    employee_id = result[0]
    
    # Insert the task into the tasks table
    cursor.execute('''
    INSERT INTO tasks (task_name, assigned_to, deadline, task_status)
    VALUES (?, ?, ?, ?)
    ''', (task_name, employee_id, deadline, 'In Progress'))
    
    # Update the employee's task and task status
    cursor.execute('''
    UPDATE employees
    SET task = ?, task_status = 'In Progress'
    WHERE employee_id = ?
    ''', (task_name, employee_id))
    
    conn.commit()
    conn.close()
    print(f"Task '{task_name}' assigned to {employee_name} successfully!")
