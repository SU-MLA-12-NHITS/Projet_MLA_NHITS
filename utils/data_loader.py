import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

def split_data(data, train_ratio=0.7, valid_ratio=0.1):
    """
    Splits data into train, validation, and test sets.
    """
    total_rows = len(data)
    train_size = int(total_rows * train_ratio)
    valid_size = int(total_rows * valid_ratio)
    test_size = total_rows - train_size - valid_size

    train = data.iloc[:train_size].copy()
    valid = data.iloc[train_size:train_size + valid_size].copy()
    test = data.iloc[train_size + valid_size:].copy()

    return train, valid, test

def normalize_data(train, valid, test, feature_columns):
    """
    Normalizes the data using the training set statistics.
    """
    scaler = StandardScaler()
    train_scaled = train.copy()
    valid_scaled = valid.copy()
    test_scaled = test.copy()

    train_scaled[feature_columns] = scaler.fit_transform(train[feature_columns])
    valid_scaled[feature_columns] = scaler.transform(valid[feature_columns])
    test_scaled[feature_columns] = scaler.transform(test[feature_columns])

    return train_scaled, valid_scaled, test_scaled, scaler

def create_sequences(data, target_column, sequence_length=24):
    """
    Converts data into input-output sequences for time-series forecasting.
    """
    values = data[target_column].values
    X, y = [], []
    for i in range(len(values) - sequence_length):
        X.append(values[i:i + sequence_length])
        y.append(values[i + sequence_length])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def prepare_dataloaders(train, valid, test, feature_column, sequence_length=24, batch_size=32):
    """
    Prepares PyTorch DataLoaders for train, validation, and test sets.
    """
    train_X, train_y = create_sequences(train, feature_column, sequence_length)
    valid_X, valid_y = create_sequences(valid, feature_column, sequence_length)
    test_X, test_y = create_sequences(test, feature_column, sequence_length)

    train_loader = DataLoader(TensorDataset(train_X, train_y), batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(TensorDataset(valid_X, valid_y), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(test_X, test_y), batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
