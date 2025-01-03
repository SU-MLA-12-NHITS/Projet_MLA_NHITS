import torch
import torch.nn as nn
from .nhits_stack import NHITSStack

class NHITS(nn.Module):
    def __init__(self, input_size, hidden_size, forecast_horizon, num_stacks, num_blocks_per_stack, pooling_kernel_sizes, expressiveness_ratios):
        """
        Initializes the NHITS model.

        Args:
            input_size (int): Length of the input time series segment.
            hidden_size (int): Number of hidden units in each block's MLP.
            forecast_horizon (int): Number of time steps to forecast.
            num_stacks (int): Number of stacks in the model.
            num_blocks_per_stack (list[int]): Number of blocks in each stack.
            pooling_kernel_sizes (list[list[int]]): List of kernel sizes for pooling layers in each stack's blocks.
            expressiveness_ratios (list[list[float]]): List of expressiveness ratios for each block in each stack.
        """
        super(NHITS, self).__init__()
        
        assert len(num_blocks_per_stack) == num_stacks, "Number of blocks per stack must match the number of stacks."
        assert len(pooling_kernel_sizes) == num_stacks, "Pooling kernel sizes must match the number of stacks."
        assert len(expressiveness_ratios) == num_stacks, "Expressiveness ratios must match the number of stacks."

        # Create stacks of NHITS blocks
        self.stacks = nn.ModuleList([
            NHITSStack(
                input_size=input_size,
                hidden_size=hidden_size,
                forecast_horizon=forecast_horizon,
                num_blocks=num_blocks_per_stack[i],
                pooling_kernel_sizes=pooling_kernel_sizes[i],
                expressiveness_ratios=expressiveness_ratios[i]
            )
            for i in range(num_stacks)
        ])

    def forward(self, x):
        """
        Forward pass of the NHITS model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            global_forecast (torch.Tensor): Final forecast output of shape (batch_size, forecast_horizon).
        """
        residual = x  # Initialize residual as the input
        global_forecast = torch.zeros((x.size(0), self.stacks[0].blocks[0].forecast_horizon), device=x.device)

        for stack in self.stacks:
            _, forecast = stack(residual)
            residual = residual - forecast  # Update residual for the next stack
            global_forecast += forecast  # Aggregate forecasts from all stacks

        return global_forecast
