"""
This code prepares and processes time series data for training, validation, and testing by normalizing, creating rolling windows, 
splitting the data, and converting it to PyTorch tensors for use with DataLoaders.
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from data.utils import create_rolling_windows, split_data

# --- Data preparation functions ---
def convert_date(df):
    """
    Converts the 'date' column of a DataFrame to datetime format.
    """
    if 'date' not in df.columns:
        raise KeyError("Column 'date' is missing.")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df['date'].isnull().any():
        raise ValueError("Some dates are invalid.")
    return df

def normalize_data(data):
    """
    Normalize the dataset by subtracting the mean and dividing by the standard deviation.
    """
    mean = data.mean()
    std = data.std()
    return (data - mean) / std

def ensure_numeric_type(*args):
    """
    Ensure all data inputs are of numeric type (float32).
    """
    return tuple(np.array(arg, dtype=np.float32) for arg in args)

def create_dataloaders(X_train, Y_train, X_val, Y_val, X_test, Y_test, batch_size):
    """
    Create PyTorch DataLoaders for training, validation, and testing data.
    """
    train_loader = DataLoader(list(zip(torch.tensor(X_train), torch.tensor(Y_train))),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(list(zip(torch.tensor(X_val), torch.tensor(Y_val))),
                            batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(list(zip(torch.tensor(X_test), torch.tensor(Y_test))),
                             batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def prepare_data(dataset, input_size, output_size, batch_size):
    """
    Prepare the dataset for training, validation, and testing by normalizing, creating rolling windows,
    and converting to PyTorch tensors.
    """
    # Normalize data
    normalized_data = normalize_data(dataset)
    
    # Split data into train, validation, and test
    train_data, val_data, test_data = split_data(normalized_data)
    
    # Create rolling windows
    X_train, Y_train = create_rolling_windows(train_data, input_size, output_size)
    X_val, Y_val = create_rolling_windows(val_data, input_size, output_size)
    X_test, Y_test = create_rolling_windows(test_data, input_size, output_size)
    
    # Ensure numeric type before tensor conversion
    X_train, Y_train, X_val, Y_val, X_test, Y_test = ensure_numeric_type(X_train, Y_train, X_val, Y_val, X_test, Y_test)
    
    # Create DataLoaders
    return create_dataloaders(X_train, Y_train, X_val, Y_val, X_test, Y_test, batch_size)