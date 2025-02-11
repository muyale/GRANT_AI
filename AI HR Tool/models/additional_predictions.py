import sqlite3
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import sys

# Ensure the project root is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ADDITIONAL_DB_PATH

# Dataset class for additional predictions (e.g., burnout risk)
class AdditionalPredictionsDataset(Dataset):
    def __init__(self, dataframe, feature_columns):
        # Convert features to float array; assumes preprocessing has been done.
        self.data = dataframe[feature_columns].values.astype(float)
        self.labels = dataframe['burnout_risk'].values.astype(float)  # Binary: 0 (low risk), 1 (high risk)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.tensor(self.data[idx], dtype=torch.float32),
                torch.tensor(self.labels[idx], dtype=torch.float32))

# Define a Neural Network for Additional Predictions
class AdditionalPredictionsModel(nn.Module):
    def __init__(self, input_dim):
        super(AdditionalPredictionsModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

def fetch_additional_data():
    """
    Fetch additional HR data from hr_additional.db.
    Expected columns: name, historical_performance, sentiment_score, work_pattern, burnout_risk.
    """
    conn = sqlite3.connect(ADDITIONAL_DB_PATH)
    query = "SELECT name, historical_performance, sentiment_score, work_pattern, burnout_risk FROM employee_predictions"
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Example: If sentiment_score is stored as string, convert it to numeric.
    # (Assumes data is clean; otherwise, add error handling.)
    df['historical_performance'] = pd.to_numeric(df['historical_performance'], errors='coerce')
    df['sentiment_score'] = pd.to_numeric(df['sentiment_score'], errors='coerce')
    df['work_pattern'] = pd.to_numeric(df['work_pattern'], errors='coerce')
    df['burnout_risk'] = pd.to_numeric(df['burnout_risk'], errors='coerce')
    
    # Fill missing values if needed
    df = df.fillna(0)
    return df

def train_additional_model(df, feature_columns):
    """
    Train a PyTorch model to predict burnout risk.
    """
    dataset = AdditionalPredictionsDataset(df, feature_columns)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    model = AdditionalPredictionsModel(input_dim=len(feature_columns))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 60
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
            print(f"[Additional Model] Epoch {epoch+1}, Loss: {total_loss:.4f}")
    return model
