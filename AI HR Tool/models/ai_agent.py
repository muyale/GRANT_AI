import sqlite3
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HR_DB_PATH

# Dataset class for employee data
class EmployeeDataset(Dataset):
    def __init__(self, dataframe, feature_columns):
        # We assume that the dataframe's feature_columns are numeric.
        self.data = dataframe[feature_columns].values.astype(float)
        self.labels = dataframe['suitable'].values.astype(float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32))

# Define a simple Feedforward Neural Network model
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
    Fetch employee data from the SQLite database.
    Returns a dataframe with:
      - name: employee name (string)
      - hours_available: numeric hours available
      - task_status: categorical status (encoded to numeric)
      - on_leave: 0/1 indicator
      - logged_in: 0/1 indicator
      - skill_tags: employee skills
      - suitable: a binary target computed from other factors
    """
    conn = sqlite3.connect(HR_DB_PATH)
    query = "SELECT name, hours_available, task_status, on_leave, logged_in, skill_tags FROM employees"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Encode categorical features using LabelEncoder
    le_task_status = LabelEncoder()
    df['task_status'] = le_task_status.fit_transform(df['task_status'])
    
    # Retain the skill_tags column for task assignment
    df['skill_tags'] = df['skill_tags'].fillna('')  # Handle missing skill tags
    
    # Compute 'suitable' target based on certain criteria
    df['suitable'] = ((df['hours_available'] > 4) & (df['on_leave'] == 0) & (df['logged_in'] == 1)).astype(int)
    
    return df

def train_pytorch_model(df, feature_columns):
    """
    Train a PyTorch model on the provided dataframe and features.
    The model predicts the 'suitable' binary target.
    """
    dataset = EmployeeDataset(df, feature_columns)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = EmployeeTaskModel(input_dim=len(feature_columns))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
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
    return model
