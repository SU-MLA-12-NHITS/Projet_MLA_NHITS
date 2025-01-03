import torch
import torch.nn as nn
import torch.nn.functional as F

class NHITSBlock(nn.Module):
    def __init__(self, input_size, hidden_size, forecast_horizon, pooling_kernel_size, expressiveness_ratio=1.0):
        """
        Initializes an NHITS block.
        
        Args:
            input_size (int): Length of the input time series segment.
            hidden_size (int): Number of hidden units in the MLP.
            forecast_horizon (int): Number of time steps to forecast.
            pooling_kernel_size (int): Kernel size for max-pooling to reduce input size.
            expressiveness_ratio (float): Fraction of forecast horizon to use for coefficients.
        """
        super(NHITSBlock, self).__init__()
        
        # Pooling layer for multi-rate input sampling
        self.max_pool = nn.MaxPool1d(kernel_size=pooling_kernel_size, stride=pooling_kernel_size)
        
        # MLP for backcast and forecast generation
        reduced_size = input_size // pooling_kernel_size  # Adjusted size after pooling
        self.hidden_layer = nn.Linear(reduced_size, hidden_size)  # Hidden layer
        self.backcast_layer = nn.Linear(hidden_size, input_size)  # Backcast output
        self.forecast_layer = nn.Linear(hidden_size, int(forecast_horizon * expressiveness_ratio))  # Forecast coefficients
        
        # Store forecast horizon for interpolation
        self.forecast_horizon = forecast_horizon

    def forward(self, x):
        """
        Forward pass of the NHITS block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        
        Returns:
            backcast (torch.Tensor): Backcast output of shape (batch_size, input_size).
            forecast (torch.Tensor): Forecast output of shape (batch_size, forecast_horizon).
        """
        # Step 1: Multi-rate input sampling using MaxPool
        pooled_x = self.max_pool(x.unsqueeze(1)).squeeze(1)  # Reduce input size
        
        # Step 2: Non-linear feature extraction using MLP
        hidden = F.relu(self.hidden_layer(pooled_x))  # Hidden layer activation
        
        # Step 3: Generate backcast and forecast coefficients
        backcast = self.backcast_layer(hidden)  # Backcast reconstruction
        forecast_coeffs = self.forecast_layer(hidden)  # Forecast coefficients
        
        # Step 4: Hierarchical interpolation for forecast
        forecast = self.interpolate(forecast_coeffs, self.forecast_horizon)
        
        return backcast, forecast

    def interpolate(self, coeffs, horizon):
        """
        Interpolates forecast coefficients to match the forecast horizon.
        
        Args:
            coeffs (torch.Tensor): Forecast coefficients of shape (batch_size, reduced_horizon).
            horizon (int): Full forecast horizon.
        
        Returns:
            interpolated (torch.Tensor): Interpolated forecast of shape (batch_size, horizon).
        """
        # Step 4.1: Create time steps for interpolation
        batch_size, reduced_horizon = coeffs.size()
        scale = horizon / reduced_horizon
        t = torch.arange(0, horizon, device=coeffs.device) / scale
        
        # Step 4.2: Find indices for interpolation
        t_floor = torch.floor(t).long()
        t_ceil = torch.ceil(t).long()
        t_floor = torch.clamp(t_floor, 0, reduced_horizon - 1)
        t_ceil = torch.clamp(t_ceil, 0, reduced_horizon - 1)
        
        # Step 4.3: Linear interpolation
        weights = t - t_floor
        interpolated = (1 - weights) * coeffs[:, t_floor] + weights * coeffs[:, t_ceil]
        return interpolated


