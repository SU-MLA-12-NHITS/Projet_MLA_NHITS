"""
This code implements the NHITS model for time series forecasting.
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
        dropout_rate (float): Dropout rate to apply between MLP layers.
    """
    def __init__(self, input_size, output_size, pooling_kernel_size, hidden_size, activation=nn.ReLU, dropout_rate=0.5):
        super(NHITSBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.pooling_kernel_size = pooling_kernel_size
        
        # Pooling layer to reduce dimensionality
        self.pooling = nn.MaxPool1d(kernel_size=pooling_kernel_size, stride=1, padding=0)
        
        # Adjusted input size after pooling
        pooled_size = input_size - pooling_kernel_size + 1

        # Multi-Layer Perceptron (MLP) for feature extraction
        self.mlp = nn.Sequential(
            nn.Linear(pooled_size, hidden_size),  # First hidden layer
            activation(),                        # Activation function
            nn.Dropout(dropout_rate),            # Dropout layer for regularization
            nn.Linear(hidden_size, hidden_size), # Second hidden layer
            activation(),                        # Activation function
            nn.Dropout(dropout_rate),            # Dropout layer for regularization
            nn.Linear(hidden_size, output_size)  # Output layer
        )

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
        
        # Add a channel dimension for the pooling layer
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, input_size)

        # Apply pooling
        x = self.pooling(x).squeeze(1)  # Remove channel dimension after pooling
        # # Apply pooling if the kernel size is greater than 1
        # if self.pooling_kernel_size > 1:
        #     # Add a channel dimension for the pooling layer to operate
        #     x = self.pooling(x.unsqueeze(1)).squeeze(1)

        # Extract features using the MLP
        x = self.mlp(x)

        # Ensure dimensions match for the projection layer
        input_projection_size = x.shape[1]
        if input_projection_size != self.projection.in_features:
            # Redefine projection layer if input size to projection doesn't match
            self.projection = nn.Linear(input_projection_size, self.projection.out_features)

        # Apply the projection layer to align the output size with the input size
        x = self.projection(x)
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
        self.stacks = nn.ModuleList()  # List to hold stacks
        self.output_size = output_size

        # Create stacks, each containing multiple blocks
        for stack_idx in range(stacks):
            stack_blocks = nn.ModuleList()  # List for blocks in the current stack
            for _ in range(blocks_per_stack):
                # Create a block for each stack
                block = NHITSBlock(
                    input_size=input_size,
                    output_size=output_size,
                    pooling_kernel_size=pooling_kernel_sizes[stack_idx],
                    hidden_size=hidden_size,
                    activation=activation
                )
                stack_blocks.append(block)
            self.stacks.append(stack_blocks)
        
        # Define expressiveness ratios for each stack
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
        # outputs = []
        stack_outputs = []    # List to store outputs from each stack

        # Loop over each stack and each block within the stack
        for stack_idx, stack_blocks in enumerate(self.stacks):
            for block in stack_blocks:
                # Get output from the block
                block_output = block(residual)
                # Reduce residuals using expressiveness ratios
                residual = residual - (block_output / self.expressiveness_ratios[stack_idx])
                # Accumulate block output into the stack's output
                stack_outputs.append(block_output)

        # Combine outputs from all blocks
        # final_output = torch.stack(outputs, dim=0).sum(dim=0)
        final_output = torch.stack(stack_outputs, dim=0).sum(dim=0)
        
        # Return the final output, ensuring the output size is correct and the outputs of each stack
        return final_output[:, :self.output_size], stack_outputs


# --- Testing the Model ---

if __name__ == "__main__":
    # Hyperparameters for testing
    input_size = 128
    output_size = 24
    stacks = 3
    blocks_per_stack = 2
    pooling_kernel_sizes = [2, 4, 8]
    hidden_size = 512
    expressiveness_ratios = [168, 24, 1]

    # Instantiate the NHITS model with the specified hyperparameters
    model = NHITS(
        input_size=input_size,
        output_size=output_size,
        stacks=stacks,
        blocks_per_stack=blocks_per_stack,
        pooling_kernel_sizes=pooling_kernel_sizes,
        hidden_size=hidden_size,
        expressiveness_ratios=expressiveness_ratios
    )

    # Generate dummy input data (batch_size=32)
    x = torch.randn(32, input_size)  # Batch of 32 time series samples
    # Perform a forward pass through the model
    output, _ = model(x)

    # Print output shape to verify correctness
    print(output.shape)  # Expected shape: (32, output_size)