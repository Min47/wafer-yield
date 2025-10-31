"""
SECOM Dataset Downloader
========================

Download the SECOM dataset from UCI Machine Learning Repository
and prepares it for the wafer yield optimization project.

Dataset: SECOM - Semiconductor Manufacturing
Source: UCI ML Repository
URL: https://archive.ics.uci.edu/ml/datasets/SECOM
"""

import os
import pandas as pd
import numpy as np
from urllib.request import urlretrieve
import zipfile
import shutil

def download_secom_dataset():
    """Download and prepare the SECOM dataset"""
    
    # Create directories if they don't exist
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)

    # SECOM dataset URLs (UCI ML Repository)
    data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
    labels_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"
    
    try:
        # Download data file
        print("\nDownloading secom.data...")
        urlretrieve(data_url, 'data/raw/secom_data.csv')
        print(f"> Completed. Data shape: {pd.read_csv('data/raw/secom_data.csv', sep=' ', header=None).shape}")
        
        # Download labels file
        print("\nDownloading secom_labels.data...")
        urlretrieve(labels_url, 'data/raw/secom_labels.csv')
        print(f"> Completed. Data shape: {pd.read_csv('data/raw/secom_labels.csv', sep=' ', header=None).shape}")
        
        # Process and combine the files
        concat_secom_data()
        
    except Exception as e:
        print(f"\nError downloading dataset: {e}")
        print("Creating synthetic SECOM-like dataset for demo purposes...")
        create_synthetic_dataset()

def concat_secom_data():
    """Process the downloaded SECOM data"""
    
    # Load the data (space-separated values)
    data = pd.read_csv('data/raw/secom_data.csv', sep=' ', header=None)
    labels = pd.read_csv('data/raw/secom_labels.csv', sep=' ', header=None)
    
    # Add column names
    data.columns = [f'feature_{i}' for i in range(data.shape[1])]
    labels.columns = ['target', 'timestamp']
    
    # Combine data and labels
    print("\nCombining data and labels...")
    combined_data = pd.concat([data, labels], axis=1)
    print(f"> Combined. Data shape: {combined_data.shape}")
    
    # Handle missing values (SECOM has many missing values)
    print("\nHandling missing values...")
    print(f"> Missing values before cleaning (per features): \n{combined_data.isnull().sum()}")
    print(f"> Missing values before cleaning (total): {combined_data.isnull().sum().sum()}")
    
    # Save raw combined data
    print("\nSaving combined data to 'data/raw/secom_combined.csv'...")
    combined_data.to_csv('data/raw/secom_combined.csv', index=False)
    print("> SECOM data processed and saved!")

def create_synthetic_dataset():
    """Create a synthetic SECOM-like dataset for demo purposes"""
    
    print("Creating synthetic SECOM-like dataset...")
    
    np.random.seed(42)
    n_samples = 1567
    n_features = 591
    
    # Generate synthetic sensor data
    # Features represent various sensor measurements and process parameters
    data = np.random.normal(0, 1, (n_samples, n_features))
    
    # Add some realistic patterns
    # Temperature-like features (correlated)
    temp_features = np.random.choice(range(n_features), 50, replace=False)
    for i, feat in enumerate(temp_features):
        base_temp = 25 + i * 0.5  # Base temperature
        data[:, feat] = base_temp + np.random.normal(0, 2, n_samples)
    
    # Pressure-like features (correlated)
    pressure_features = np.random.choice(range(n_features), 30, replace=False)
    for i, feat in enumerate(pressure_features):
        base_pressure = 1.0 + i * 0.1
        data[:, feat] = base_pressure + np.random.normal(0, 0.1, n_samples)
    
    # Add missing values (SECOM characteristic)
    missing_mask = np.random.random((n_samples, n_features)) < 0.3
    data[missing_mask] = np.nan
    
    # Create target variable (yield: 1 = good, -1 = bad)
    # Good wafers: higher values in certain features, lower in others
    good_wafer_mask = (
        (data[:, temp_features[:10]].mean(axis=1) > 0) &  # Good temperature
        (data[:, pressure_features[:5]].mean(axis=1) > 0) &  # Good pressure
        (np.random.random(n_samples) > 0.15)  # 85% good yield
    )
    
    target = np.where(good_wafer_mask, 1, -1)
    
    # Create DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=feature_names)
    df['target'] = target
    
    # Add timestamp
    df['timestamp'] = pd.date_range('2023-01-01', periods=n_samples, freq='H')
    
    # Save synthetic dataset
    df.to_csv('data/raw/secom_combined.csv', index=False)
    
    print("Synthetic dataset created!")
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution: {df['target'].value_counts()}")
    print(f"Missing values: {df.isnull().sum().sum()}")

if __name__ == "__main__":
    download_secom_dataset()
