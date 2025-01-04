#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 17:44:55 2025

@author: lisadelplanque

This code evaluates the NHITS model on a test set, calculates MSE and MAE for a specific horizon, and visualizes the true vs predicted values.

"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, horizon, device):
    """
    Evaluate the model on the test set and calculate MSE and MAE for a specific horizon.
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
    plt.title(f"Comparison of true vs predicted values for horizon {horizon}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.show()

    # Return the MSE and MAE averages so they can be used later
    return mse_avg, mae_avg