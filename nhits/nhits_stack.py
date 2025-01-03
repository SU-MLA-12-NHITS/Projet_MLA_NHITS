import torch
import torch.nn as nn
from .nhits_block import NHITSBlock

class NHITSStack(nn.Module):
    def __init__(self, input_size, hidden_size, forecast_horizon, num_blocks, pooling_kernel_sizes, expressiveness_ratios):
        """
        Initializes an NHITS stack.

        Args:
            input_size (int): Length of the input time series segment.
            hidden_size (int): Number of hidden units in the MLP of each block.
            forecast_horizon (int): Number of time steps to forecast.
            num_blocks (int): Number of blocks in the stack.
            pooling_kernel_sizes (list[int]): List of kernel sizes for each block's pooling layer.
            expressiveness_ratios (list[float]): List of expressiveness ratios for each block.
        """
        super(NHITSStack, self).__init__()

        assert len(pooling_kernel_sizes) == num_blocks, "Pooling kernel sizes must match the number of blocks."
        assert len(expressiveness_ratios) == num_blocks, "Expressiveness ratios must match the number of blocks."

        self.blocks = nn.ModuleList([
            NHITSBlock(
                input_size=input_size,
                hidden_size=hidden_size,
                forecast_horizon=forecast_horizon,
                pooling_kernel_size=pooling_kernel_sizes[i],
                expressiveness_ratio=expressiveness_ratios[i]
            )
            for i in range(num_blocks)
        ])

    def forward(self, x):
        """
        Forward pass of the NHITS stack.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            global_backcast (torch.Tensor): Final backcast output of shape (batch_size, input_size).
            global_forecast (torch.Tensor): Final forecast output of shape (batch_size, forecast_horizon).
        """
        residual = x  # Initialize residual as the input
        global_forecast = torch.zeros((x.size(0), self.blocks[0].forecast_horizon), device=x.device)

        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast  # Update residual for the next block
            global_forecast += forecast  # Aggregate forecasts from all blocks

        global_backcast = residual
        return global_backcast, global_forecast
