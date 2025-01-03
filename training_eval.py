#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 17:18:37 2025

@author: lisadelplanque

Script for training and evaluating an NHITS model on time-series data.

"""

import torch
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from NHITS_modele import *
from utils import create_rolling_windows, split_data
import torch.nn as nn
import torch.nn.functional as F

# --- Utility Functions ---
def set_seed(seed):
    """
    Fonction pour initialiser les graines aléatoires.
    
    Parameters:
        seed (int): La graine à utiliser pour la reproductibilité.
    """
    random.seed(seed)  # Pour Python
    np.random.seed(seed)  # Pour NumPy
    torch.manual_seed(seed)  # Pour PyTorch CPU
    torch.cuda.manual_seed_all(seed)  # Pour PyTorch GPU
    torch.backends.cudnn.deterministic = True  # Pour CUDA (GPU)
    torch.backends.cudnn.benchmark = False  # Pour éviter des comportements non déterministes


def train_model(model, train_loader, val_loader, training_steps, criterion, optimizer, device):
    """
    Train the NHITS model for a specified number of training steps.

    Parameters:
        model (nn.Module): NHITS model to train.
        train_loader (DataLoader): DataLoader for the training data.
        val_loader (DataLoader): DataLoader for the validation data.
        training_steps (int): Number of training steps.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to run training (CPU or GPU).
    """
    model.to(device)
    
    # Check if the validation DataLoader is empty
    if len(val_loader.dataset) == 0:
        raise ValueError("The validation DataLoader is empty!")
        
    step = 0
    while step < training_steps:
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            if step >= training_steps:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            step += 1

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        print(f"Step {step}/{training_steps}, Train Loss: {train_loss / len(train_loader):.4f}, "
              f"Validation Loss: {val_loss / len(val_loader):.4f}")


def evaluate_model(model, test_loader, horizon, device):
    """
    Evaluate the model on the test set and calculate MSE and MAE for a specific horizon.

    Parameters:
        model (nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for the test data.
        horizon (int): Horizon value to evaluate.
        device (torch.device): Device to run evaluation (CPU or GPU).
    """
    model.eval()
    
    total_mse = 0
    total_mae = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Predict only for the specified horizon
            targets = targets[:, :horizon]
            
            outputs = model(inputs)
            outputs = outputs[:, :horizon]

            # Compute errors for the specified horizon
            total_mse += F.mse_loss(outputs, targets, reduction='sum').item()
            total_mae += F.l1_loss(outputs, targets, reduction='sum').item()

            # Store true and predicted values for plotting
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    # Compute average errors for the horizon
    mse_avg = total_mse / len(test_loader.dataset)
    mae_avg = total_mae / len(test_loader.dataset)

    print(f"Horizon: {horizon} | Average MSE: {mse_avg:.4f} | Average MAE: {mae_avg:.4f}")

    # Plot true vs predicted values for the horizon
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label="True values", color='blue')
    plt.plot(y_pred, label="Predicted values", color='red')
    #plt.legend(loc='best')
    plt.title(f"Comparison of True vs Predicted Values for Horizon {horizon}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.show()


def main():
    # --- Dataset and Model Initialization ---
    
    # Set the random seed for reproducibility
    set_seed(10)  # Article: Discrete range (1,10)
    
    # Load the dataset
    dataset = pd.read_csv("all_six_dataset/exchange_rate.csv")
    column_name = '0'  # Column to be used
    data = dataset[column_name]

    # Split data into train, validation, and test sets
    train_data, val_data, test_data = split_data(data)

    # Convert to PyTorch tensors
    train_data = torch.tensor(train_data.values, dtype=torch.float32)
    val_data = torch.tensor(val_data.values, dtype=torch.float32)
    test_data = torch.tensor(test_data.values, dtype=torch.float32)

    # --- Hyperparameter Definition ---
    horizon = 96  # Forecast horizon
    m = 5
    input_size = m * horizon  # Input size based on horizon
    output_size = 24  # Output size
    batch_size = 32
    hidden_size = 512
    stacks = 3
    blocks_per_stack = 1
    pooling_kernel_sizes = [8, 4, 1]
    expressiveness_ratios = [168, 24, 1]
    epochs = 20
    learning_rate = 1e-3
    training_steps = 1000

    # --- Data Preparation ---
    # Generate rolling windows
    X_train, Y_train = create_rolling_windows(train_data, input_size, output_size)
    X_test, Y_test = create_rolling_windows(test_data, input_size, output_size)
    X_val, Y_val = create_rolling_windows(val_data, input_size, output_size)

    # Ensure numeric type before tensor conversion
    X_train = np.array(X_train, dtype=np.float32)
    Y_train = np.array(Y_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    Y_test = np.array(Y_test, dtype=np.float32)
    X_val = np.array(X_val, dtype=np.float32)
    Y_val = np.array(Y_val, dtype=np.float32)

    # Create DataLoaders
    train_loader = DataLoader(
        list(zip(torch.tensor(X_train), torch.tensor(Y_train))),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        list(zip(torch.tensor(X_val), torch.tensor(Y_val))),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        list(zip(torch.tensor(X_test), torch.tensor(Y_test))),
        batch_size=batch_size, shuffle=False
    )

    # --- Model Initialization ---
    model = NHITS(
        input_size=input_size,
        output_size=output_size,
        stacks=stacks,
        blocks_per_stack=blocks_per_stack,
        pooling_kernel_sizes=pooling_kernel_sizes,
        hidden_size=hidden_size,
        expressiveness_ratios=expressiveness_ratios
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # --- Training ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection for debugging

    # Train the model
    train_model(model, train_loader, val_loader, training_steps=training_steps, criterion=criterion, optimizer=optimizer, device=device)

    # --- Evaluation ---
    # Evaluate the model on the test set for the chosen horizon
    evaluate_model(model, test_loader, horizon, device)

# Entry point
if __name__ == "__main__":
    main()
