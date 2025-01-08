
""""
Main script to execute.
"""
import torch
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
from src.model.NHITS_model import *
from data.utils import plot_stack_outputs
import src.training.config as config
from src.training.training import train_model
from src.evaluation import evaluate_model
from data.data_preparation import *

# --- Main Function ---
def main():
    # --- Dataset and Model Initialization ---
    
    # Load the dataset
    dataset = pd.read_csv(config.DATASET_PATH )
    
    # Check if the first column is a date column and remove it if so
    if pd.to_datetime(dataset.iloc[:, 0], errors='coerce').notnull().all():
        dataset = dataset.drop(dataset.columns[0], axis=1)
    
    print(dataset.columns)
    
    mse_list = []
    mae_list = []

    for column_name in dataset.columns:
        print(f"Processing column: {column_name}")
        
        data = dataset[column_name]
        
        # Data processing and model setup using parameters from config.py
        horizon = config.HYPERPARAMETERS['horizon']
        input_size = config.HYPERPARAMETERS['input_size']
        output_size = config.HYPERPARAMETERS['output_size']
        batch_size = config.HYPERPARAMETERS['batch_size']
        hidden_size = config.HYPERPARAMETERS['hidden_size']
        stacks = config.HYPERPARAMETERS['stacks']
        blocks_per_stack = config.HYPERPARAMETERS['blocks_per_stack']
        pooling_kernel_sizes = config.HYPERPARAMETERS['pooling_kernel_sizes']
        expressiveness_ratios = config.HYPERPARAMETERS['expressiveness_ratios']
        learning_rate = config.HYPERPARAMETERS['learning_rate']
        learning_rate_decay = config.HYPERPARAMETERS['learning_rate_decay']
        training_steps = config.HYPERPARAMETERS['training_steps']
        
        # Prepare data
        train_loader, val_loader, test_loader = prepare_data(data, input_size, output_size, batch_size)
        
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
        # Sample input
        x = torch.randn(batch_size, input_size)  # Batch size, input size
        # Forward pass
        final_output, stack_outputs = model(x)

        # Access the final forecast
        print("Final Output Shape:", final_output.shape)

        # Access stack contributions
        # for i, stack_output in enumerate(stack_outputs):
        #    print(f"Stack {i+1} Output Shape:", stack_output.shape)

        # Plot stack contributions
        plot_stack_outputs(stack_outputs, horizon)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=training_steps // 3, gamma=learning_rate_decay)

        # Train the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection for debugging

        # Train the model
        train_model(model, train_loader, val_loader, training_steps=training_steps, criterion=criterion, optimizer=optimizer, scheduler=scheduler, device=device)

        # Evaluate the model
        mse, mae = evaluate_model(model, test_loader, horizon, device)
        
        # Store the results for MSE and MAE
        mse_list.append(mse)
        mae_list.append(mae)

    # Compute the average MSE and MAE across all columns
    avg_mse = np.mean(mse_list)
    avg_mae = np.mean(mae_list)

    print(f"Average MSE across all columns: {avg_mse}")
    print(f"Average MAE across all columns: {avg_mae}")

# Entry point
if __name__ == "__main__":
    main()