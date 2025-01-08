# Projet_MLA_NHITS
Advanced Machine Learning Project

This repository implements N-HiTS, the algorithm introduced in the following paper:

"N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting", by C. Challu, K. Olivares, B. Oreshkin, F. Garza, M. Mergenthaler-Canseco, A. Dubrawski (https://arxiv.org/abs/2201.12886v5).

This repository contains the implementation of the N-HITS model for time series forecasting, along with data preparation, training scripts, and exploratory analysis notebooks.

## Table of Contents

- [Folder Structure](#folder-structure)
- [Key Features](#key-features)
- [Setting Up the Environment](#setting-up-the-environment)
- [Usage](#usage)

## Folder Structure
```plaintext
├── data/
│   ├── data_preparation.py      # Core data preparation functions
│   └── utils.py                 # Utility functions
├── notebooks/exploratory_analysis/
│   ├── ECL_analysis.ipynb
│   ├── ETTm2_analysis.ipynb
│   ├── exchange_rate_analysis.ipynb
│   ├── national_illness_analysis.ipynb
│   ├── traffic_analysis.ipynb
│   └── weather_analysis.ipynb
├── src/
│   ├── model/
│   │   └── NHITS_model.py       # NHITS implementation
│   └── training/
│       ├── config.py            # Configuration and hyperparameters
│       ├── training.py          # Training loop
│   └── evaluation.py            # Evaluation logic
├── main.py                      # Main script to run
├── demo_model.ipynb             # Demonstrates the NHITS model in action
```

## Key Features

- **N-HITS Model**: Implements a neural network for time series forecasting.
- **Exploratory Analysis**: Jupyter notebooks for analyzing multiple datasets (e.g., ETTm2, weather, traffic).
- **Data Preparation**: Tools for preprocessing datasets, normalizing, and creating rolling windows.
- **Custom Configurations**: Customizable hyperparameters and dataset paths in `config.py`.


## Setting Up the Environment

To create and activate the Conda environment, follow these steps:

1. **Create the environment**:

   ```bash
   conda env create -f environment-cpu.yml

2. **Activate the environment**:
    ```bash
    conda activate nhits

### Prerequisites
- Python >= 3.12.4
- numpy >= 1.26.4
- matplotlib >= 3.8.4
- PyTorch >= 2.3.0
- pandas >= 2.2.2
- scikit-learn >= 1.4.2

### Dataset Download
Datasets for this project can be downloaded from [Google Drive](https://drive.google.com/file/d/1alE33S1GmP5wACMXaLu50rDIoVzBM4ik/view). Once downloaded, place the datasets in the `data/` folder.
```plaintexts
├── data/
│   ├── all_six_dataset/
│   │   ├── electricity
│   │   │   └──electricity.csv
│   │   ├── ETT-small
```


## Usage
### Notebooks
Analyze datasets using the Jupyter notebooks in `notebooks/`.

### Configuration

All model and training hyperparameters are stored in `src/training/config.py`.
Adjust hyperparameters and path to dataset in this file.

### Training
Run the `main.py` script to preprocess the data, train the N-HITS model and evaluate its performance:
```bash
python main.py
```

### Demo
Explore the N-HITS model in `demo_model.ipynb`.


