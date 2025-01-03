#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 15:19:18 2025

@author: lisadelplanque
"""
"""
Created on Fri Jan 3 15:19:18 2025

@author: lisadelplanque
"""

import torch
import torch.nn as nn

class NHITSBlock(nn.Module):
    """
    Represents a single block of the NHITS model.

    Each block consists of an MLP with activation layers and optional pooling,
    followed by a projection layer to align the output size with the input size.

    Parameters:
        input_size (int): The size of the input time series data.
        output_size (int): The size of the output predictions for this block.
        pooling_kernel_size (int): The size of the kernel used for pooling.
        hidden_size (int): The number of neurons in the hidden layers of the MLP.
        activation (nn.Module): Activation function used in the MLP layers.
    """
    def __init__(self, input_size, output_size, pooling_kernel_size, hidden_size, activation=nn.ReLU):
        super(NHITSBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.pooling_kernel_size = pooling_kernel_size

        # Multi-Layer Perceptron (MLP) for feature extraction
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, output_size)
        )

        # Optional pooling layer to reduce dimensionality
        self.pooling = nn.MaxPool1d(kernel_size=pooling_kernel_size, stride=1, padding=0)

        # Projection layer to align output size with input size
        self.projection = nn.Linear(output_size, input_size)

    def forward(self, x):
        """
        Forward pass for the NHITSBlock.

        Parameters:
            x (torch.Tensor): Input tensor with shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor after processing through the block.
        """
        #print(f"Input shape before flattening: {x.shape}")
        #x = x.view(x.size(0), -1)  # Flatten input 
        #print(f"Input shape after flattening: {x.shape}")
        x = self.mlp(x)
        #print(f"Shape after MLP: {x.shape}")
        x = self.projection(x)  # Project output to match input size
        return x


class NHITS(nn.Module):
    """
    Implements the NHITS model for time series forecasting.

    The model consists of multiple stacks, where each stack contains one or more blocks.
    Each block contributes to the final forecast by reducing the residuals iteratively.

    Parameters:
        input_size (int): The size of the input time series data.
        output_size (int): The size of the final output predictions.
        stacks (int): Number of stacks in the model.
        blocks_per_stack (int): Number of blocks within each stack.
        pooling_kernel_sizes (list of int): Pooling kernel sizes for each stack.
        hidden_size (int): The number of neurons in the hidden layers of the MLP.
        expressiveness_ratios (list of int): Ratios to scale residual contributions for each stack.
        activation (nn.Module): Activation function used in the MLP layers.
    """
    def __init__(self, input_size, output_size, stacks, blocks_per_stack, pooling_kernel_sizes, hidden_size, expressiveness_ratios, activation=nn.ReLU):
        super(NHITS, self).__init__()
        self.stacks = nn.ModuleList()
        self.output_size = output_size
        
        # Create stacks, each containing multiple blocks
        for stack_idx in range(stacks):
            stack_blocks = nn.ModuleList()
            for _ in range(blocks_per_stack):
                block = NHITSBlock(
                    input_size=input_size,
                    output_size=output_size,
                    pooling_kernel_size=pooling_kernel_sizes[stack_idx],
                    hidden_size=hidden_size,
                    activation=activation
                )
                stack_blocks.append(block)
            self.stacks.append(stack_blocks)
        
        self.expressiveness_ratios = expressiveness_ratios

    def forward(self, x):
        """
        Forward pass for the NHITS model.

        Parameters:
            x (torch.Tensor): Input tensor with shape (batch_size, input_size).

        Returns:
            torch.Tensor: Final output tensor with shape (batch_size, output_size).
        """
        residual = x.clone()  # Initialize residuals with the input
        outputs = []
        
        for stack_idx, stack_blocks in enumerate(self.stacks):
            for block in stack_blocks:
                block_output = block(residual)
                # Reduce residuals using expressiveness ratios
                residual = residual - (block_output / self.expressiveness_ratios[stack_idx])

                #residual -= block_output / self.expressiveness_ratios[stack_idx]
                outputs.append(block_output)

        # Combine outputs from all blocks
        final_output = torch.stack(outputs, dim=0).sum(dim=0)
        
        # Ensure output matches the desired size
        return final_output[:, :self.output_size]


# --- Testing the Model ---

# Hyperparameters for testing
input_size = 128
output_size = 24
stacks = 3
blocks_per_stack = 2
pooling_kernel_sizes = [2, 4, 8]
hidden_size = 512
expressiveness_ratios = [168, 24, 1]

"""
batch_size = 32
epochs = 20
learning_rate = 1e-3"""

# Instantiate the NHITS model
model = NHITS(
    input_size=input_size,
    output_size=output_size,
    stacks=stacks,
    blocks_per_stack=blocks_per_stack,
    pooling_kernel_sizes=pooling_kernel_sizes,
    hidden_size=hidden_size,
    expressiveness_ratios=expressiveness_ratios
)

# Generate dummy input data
x = torch.randn(32, input_size)  # Batch of 32 time series samples
output = model(x)

# Print output shape to verify correctness
print(output.shape)  # Expected shape: (32, output_size)
