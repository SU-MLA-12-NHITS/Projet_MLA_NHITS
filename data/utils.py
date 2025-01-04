#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 12:59:37 2025

@author: lisadelplanque

Time series analysis from a dataset.
"""

# ----- Library imports ------
import numpy as np
import random
import torch
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def set_seed(seed):
    """
    Function to initialize random seeds for reproducibility.
    
    Parameters:
        seed (int): The seed value to use for random number generation.
    """
    random.seed(seed)  # For Python's built-in random module
    np.random.seed(seed)  # For NumPy random number generation
    torch.manual_seed(seed)  # For PyTorch on CPU
    torch.cuda.manual_seed_all(seed)  # For PyTorch on GPU
    torch.backends.cudnn.deterministic = True  # For CUDA (GPU) to ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # To avoid non-deterministic behavior in CUDA operations

def convert_date(df):
    """
    Converts the 'date' column of a DataFrame to datetime format.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame with a 'date' column.

    Returns:
        pd.DataFrame: DataFrame with the 'date' column converted to datetime.

    Raises:
        KeyError: If the 'date' column is missing.
        ValueError: If any dates are invalid and cannot be coerced.
    """
    if 'date' not in df.columns:
        raise KeyError("Column 'date' is missing.")
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    if df['date'].isnull().any():
        raise ValueError("Some dates are invalid.")
    return df


def plot_time_series(df, column=None):
    """
    Plots time series data from a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing time series data.
        column (str, optional): The column to plot. If None, all columns are plotted.

    Raises:
        ValueError: If the specified column does not exist in the DataFrame.
    """
    if column and column not in df.columns:
        raise ValueError(f"Column '{column}' doesn't exist.")
    data_to_plot = df if column is None else df[[column]]

    plt.figure(figsize=(12, 6))
    for col in data_to_plot.columns:
        plt.plot(data_to_plot.index, data_to_plot[col], label=col)
    plt.title("Time Series")
    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.legend(loc='upper left')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()


def split_data(data):
    """
    Splits data into training, validation, and test sets.

    Parameters:
        data (pd.DataFrame): The input data to split.

    Returns:
        tuple: Three DataFrames corresponding to training, validation, and test sets.
    """
    total_rows = len(data)
    test_size = int(total_rows * 0.2)
    valid_size = int(total_rows * 0.1)
    train_size = total_rows - test_size - valid_size

    train = data.iloc[:train_size]
    valid = data.iloc[train_size:train_size+valid_size]
    test = data.iloc[train_size+valid_size:]
    return train, valid, test


def plot_data_split(train, valid, test, column_name):
    """
    Visualizes the split of data into training, validation, and test sets.

    Parameters:
        train (pd.DataFrame): Training data.
        valid (pd.DataFrame): Validation data.
        test (pd.DataFrame): Test data.
        column_name (str): The name of the column to plot.
    """
    plt.figure(figsize=(12, 4))
    plt.plot(train.index, train[column_name], label="Train")
    plt.plot(valid.index, valid[column_name], label="Validation")
    plt.plot(test.index, test[column_name], label="Test")
    plt.title("Data Division")
    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.legend()
    plt.grid()
    plt.show()


def create_rolling_windows(data, input_size, output_size):
    """
    Creates rolling windows for time series forecasting.

    Parameters:
        data (pd.Series or pd.DataFrame): Input time series data.
        input_size (int): Number of time steps in the input window.
        output_size (int): Number of time steps in the output window.

    Returns:
        tuple: Two numpy arrays (X, Y) representing input and output windows.
    """
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    
    data = data.values if isinstance(data, pd.Series) else np.array(data, dtype=np.float32)
    
    if len(data) < input_size + output_size:
        raise ValueError(
            f"Data size ({len(data)}) is too small for input_size ({input_size}) and output_size ({output_size})."
        )
    
    X, Y = [], []
    for i in range(len(data) - input_size - output_size + 1):
        X.append(data[i:i + input_size])
        Y.append(data[i + input_size:i + input_size + output_size])
    
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    return X, Y


def plot_rolling_window(data, input_size, output_size, start_index=0):
    """
    Visualizes a single rolling window for time series forecasting.

    Parameters:
        data (pd.Series): Input time series data.
        input_size (int): Number of time steps in the input window.
        output_size (int): Number of time steps in the output window.
        start_index (int): Starting index of the rolling window.
    """
    input_window = data[start_index:start_index + input_size]
    output_window = data[start_index + input_size:start_index + input_size + output_size]

    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data.values, label="Complete Series", alpha=0.5)
    plt.plot(input_window.index, input_window.values, label="Input Window", color='blue')
    plt.plot(output_window.index, output_window.values, label="Output Window", color='orange')
    plt.title("Rolling Window Visualization")
    plt.xlabel("Date")
    plt.ylabel("Values")
    plt.legend()
    plt.grid()
    plt.show()