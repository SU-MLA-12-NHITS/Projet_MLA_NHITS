"""
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
            outputs, _ = model(inputs)
            outputs = outputs[:, :horizon]

            # Ensure both are in normalized scale (avoid denormalization during loss calc)
            total_mse += F.mse_loss(outputs, targets, reduction='mean').item()
            total_mae += F.l1_loss(outputs, targets, reduction='mean').item()

            # Store true and predicted values for debugging/visualization
            y_true.extend(targets.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    # Compute average MSE and MAE
    mse_avg = total_mse / len(test_loader)
    mae_avg = total_mae / len(test_loader)

    print(f"Horizon: {horizon} | Average MSE: {mse_avg:.4f} | Average MAE: {mae_avg:.4f}")

    return mse_avg, mae_avg
