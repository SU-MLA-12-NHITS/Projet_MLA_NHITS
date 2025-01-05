"""
Configuration file. 
"""

# --- Hyperparameters for the training ---
HYPERPARAMETERS = {
    'horizon': 96,  # Forecast horizon
    # Datasets : ETTm2, exchange, ECL, traffic, weather, possible horizon's values = {96, 192, 336, 720}
    # Dataset : national_illness, possible horizon's values = {24, 36, 48, 60}
    'm': 5,  # Multiplicative factor for input size
    'input_size': 480,  # Input size based on horizon (m * horizon)
    'output_size': 96,  # Output size (horizon)
    'batch_size': 256,  # Batch size for the DataLoader
    'hidden_size': 512,  # Hidden size for the model
    'stacks': 3,  # Number of stacks in the model
    'blocks_per_stack': 1,  # Number of blocks per stack
    'pooling_kernel_sizes': [8, 4, 1],  # Pooling kernel sizes for each stack
    'expressiveness_ratios': [168, 24, 1],  # Expressiveness ratios for each stack
    'learning_rate': 1e-3,  # Learning rate
    'learning_rate_decay': 0.5,  # Learning rate decay factor
    'training_steps': 1000,  # Number of training steps
}

# --- Random Seed Configuration ---
SEED = 42  # Random seed for reproducibility

# --- File Paths ---
# DATASET_PATH = "data/all_six_dataset/ETTm2.csv"  # Path to the dataset
DATASET_PATH = "data/all_six_dataset/ETT-small/ETTm2.csv"
# DATASET_PATH = "data/all_six_dataset/exchange_rate/exchange_rate.csv"
# DATASET_PATH = "data/all_six_dataset/weather/weather.csv"
# DATASET_PATH = "data/all_six_dataset/electricity/electricity.csv"

# --- Model Configuration ---
MODEL_CONFIG = {
    'input_size': 480,  # Input size for the model
    'output_size': 24,  # Output size for the model
    'stacks': 3,  # Number of stacks in the model
    'blocks_per_stack': 1,  # Number of blocks per stack
    'pooling_kernel_sizes': [8, 4, 1],  # Pooling kernel sizes for each stack
    'hidden_size': 512,  # Hidden size for the model
    'expressiveness_ratios': [168, 24, 1],  # Expressiveness ratios for each stack
}

# --- Optimizer Configuration ---
OPTIMIZER_CONFIG = {
    'learning_rate': 1e-3,  # Learning rate
    'decay': 0.5,  # Learning rate decay factor
    'optimizer': 'adam',  # Optimizer 
}

# --- Training Configuration ---
TRAINING_CONFIG = {
    'training_steps': 1000,  # Number of training steps
}
