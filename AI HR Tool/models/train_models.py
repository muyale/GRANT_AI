import sqlite3
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
import os
import sys
import os

# Set the root directory path for the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from backend.get_employees import fetch_employee_data

from config import HR_DB_PATH

# Dataset class for employee data
class EmployeeDataset(Dataset):
    def __init__(self, dataframe, feature_columns):
        self.data = dataframe[feature_columns].values.astype(float)
        self.labels = dataframe['suitable'].values.astype(float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32))

# Define the model
class EmployeeTaskModel(nn.Module):
    def __init__(self, input_dim):
        super(EmployeeTaskModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def fetch_employee_data():
    """
    Fetch employee data from the main HR database.
    """
    conn = sqlite3.connect(HR_DB_PATH)
    query = "SELECT * FROM employees"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def train_and_save_model():
    df = fetch_employee_data()
    
    # Example features used for training; adjust based on your dataset
    df['task_status'] = LabelEncoder().fit_transform(df['task_status'])

    feature_columns = ['hours_available', 'on_leave', 'logged_in','task_status']
    
    if any(col not in df.columns for col in feature_columns):
        raise ValueError("One or more required columns are missing from the dataset.")
    
    # Compute a simple 'suitable' target for demonstration purposes
    df['suitable'] = ((df['hours_available'] > 4) & (df['on_leave'] == 0) & (df['logged_in'] == 1)).astype(int)

    # Prepare dataset and dataloader
    dataset = EmployeeDataset(df, feature_columns)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Initialize the model
    model = EmployeeTaskModel(input_dim=len(feature_columns))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training loop
    epochs = 100
    for epoch in range(epochs):
        total_loss = 0.0
        for features, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

    # Save the trained model
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "hr_task_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
