import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from nhits.nhits_model import NHITS  # Ensure NHITS and NHITSStack are properly implemented and imported.
from utils.data_loader import split_data, normalize_data, prepare_dataloaders

def train(model, train_loader, valid_loader, epochs, optimizer, criterion, device):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for X, y in valid_loader:
                X, y = X.to(device), y.to(device)
                output = model(X)
                loss = criterion(output, y)
                valid_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}")

def test(model, test_loader, device):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            output = model(X)
            predictions.append(output.cpu().numpy())
            actuals.append(y.cpu().numpy())
    return predictions, actuals

if __name__ == "__main__":
    # Load and preprocess the data
    data = pd.read_csv('traffic.csv')
    data['date'] = pd.to_datetime(data['date'])  # Ensure 'date' is in datetime format
    data = data.sort_values(by='date')

    # Select columns
    feature_column = "167"  # Example sensor column name
    train_data, val_data, test_data = split_data(data)
    train_data, val_data, test_data, scaler = normalize_data(train_data, val_data, test_data, [feature_column])

    # Prepare DataLoaders
    sequence_length = 24
    train_loader, valid_loader, test_loader = prepare_dataloaders(train_data, val_data, test_data, feature_column, sequence_length)

    # Initialize NHITS model
    input_size = sequence_length
    hidden_size = 128
    forecast_horizon = 1
    num_stacks = 3
    num_blocks_per_stack = [3, 3, 3]
    pooling_kernel_sizes = [[2, 2, 2], [4, 4, 4], [8, 8, 8]]
    expressiveness_ratios = [[0.5, 0.5, 0.5], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]]

    model = NHITS(
        input_size=input_size,
        hidden_size=hidden_size,
        forecast_horizon=forecast_horizon,
        num_stacks=num_stacks,
        num_blocks_per_stack=num_blocks_per_stack,
        pooling_kernel_sizes=pooling_kernel_sizes,
        expressiveness_ratios=expressiveness_ratios,
    )

    # Define training settings
    epochs = 20
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    device = torch.device('cpu')
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train and evaluate the model
    train(model, train_loader, valid_loader, epochs, optimizer, criterion, device)
    predictions, actuals = test(model, test_loader, device)

# Plot a subset of predictions vs actuals
plt.figure(figsize=(12, 6))
plt.plot(actuals[:100], label="Actuals", color="blue")
plt.plot(predictions[:100], label="Predictions", color="orange")
plt.title("Predictions vs Actuals")
plt.xlabel("Time Steps")
plt.ylabel("Values")
plt.legend()
plt.grid()
plt.show()
