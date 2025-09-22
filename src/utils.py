# src/utils.py
"""
Utility functions for wine quality classification project.
Helper functions for data loading, preprocessing, and common operations.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os



def load_wine_data(file_path):
    """
    Load wine dataset from CSV file.
    
    Args:
        file_path (str): Path to wine CSV file
        
    Returns:
        pd.DataFrame: Loaded wine dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded wine data: {df.shape[0]} rows × {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"Wine data file not found at: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading wine data: {str(e)}")

def create_binary_target(df, config):
    """
    Create binary quality target from wine quality scores.
    
    Args:
        df (pd.DataFrame): Wine dataset
        config (WineConfig): Configuration object
        
    Returns:
        pd.DataFrame: Dataset with added binary target column
    """

    # Use a deep copy to avoid modifying orignal datafare
    df_copy = df.copy()
    
    def classify_quality(score):
        if score in config.regular_scores:
            return 'regular'
        elif score in config.premium_scores:
            return 'premium'
        else:
            return 'unknown'  # This shouldn't happen, but ensure it's checked for in case of config typo
    
    df_copy['quality_binary'] = df_copy[config.target_col].apply(classify_quality)
    
    # Check for any unknown classifications
    unknown_count = (df_copy['quality_binary'] == 'unknown').sum()
    if unknown_count > 0:
        print(f"Warning: {unknown_count} wines with quality scores outside expected ranges")

    
    return df_copy

def encode_wine_type(df):
    """
    Convert wine_type (red/white) to a binary encoding: red=0,white=1
    """

    # make a copy rather than change the passed in df
    df_encoded = df.copy()
    
    if 'wine_type' in df.columns:
        # Convert wine_type to binary (0 for red, 1 for white)
        df_encoded['wine_type_binary'] = (df['wine_type'] == 'white').astype(int)
        # Drop original text column
        df_encoded = df_encoded.drop('wine_type', axis=1)
        print(f"Encoded wine_type: red=0, white=1")
    
    return df_encoded

def prepare_features_and_target(df, config):
    """
    Prepare feature matrix (X) and target vector (y) for modeling.
    
    Args:
        df (pd.DataFrame): Wine dataset with binary target
        config (WineConfig): Configuration object
        
    Returns:
        tuple: (X, y) where X is features DataFrame and y is target Series
    """
    # quality_binary is the target
    # all others are features
    feature_cols = [col for col in df.columns 
                   if col not in [config.target_col, 'quality_binary']]
    
    if config.drop_features:
        print_section_header("DROPPING FEATURES")
        print(f"        {config.features_to_drop}")
        feature_cols = [ col for col in feature_cols 
                        if col not in config.features_to_drop ]
    
    X = df[feature_cols]
    y = df['quality_binary']
    
    print(f"Prepared {len(feature_cols)} features: {list(feature_cols)}")
    print(f"Target variable: {y.name}")
    
    return X, y

def split_and_scale_data(X, y, config):
    """
    Split data into train/test sets and apply scaling if set to True
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target vector
        config (WineConfig): Configuration metadata
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, 
                scaler, label_encoder)
    """
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y
    )
    
    print(f"Train/test split: {len(X_train)} / {len(X_test)} samples")
    
    # Encode target labels: regular -> 0, premium -> 1 (LabelEncoder encodes alphabetically, giving the wrong order)
    label_encoding = {"regular": 0, "premium": 1}
    y_train_encoded = y_train.map(label_encoding).values
    y_test_encoded  = y_test.map(label_encoding).values

    print(f"Target encoding: {label_encoding}")
    
    # Apply scaling if set to true in config file. This could be refactored later, but for now (since the only custom
    # scaling is log transform) it will work. 
    if config.scale_features:
        # Apply additional custom scaling first if set to True.
        # For now: the only one needed is log transform
        
         # Create copies to avoid modifying original data during transformation
        X_train = X_train.copy()  
        X_test = X_test.copy() 
        if config.apply_custom_transformations:
            for feature_name in config.features_to_scale:
                X_train[feature_name] = np.log1p(X_train[feature_name])
                X_test[feature_name] = np.log1p(X_test[feature_name])
                print(f"Applied log1p transformation to {feature_name} feature")

        scaler = StandardScaler()
        
        # Fit on training data, transform both train and test
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        print(f"Applied StandardScaler to {len(X_train.columns)} features")
        print(f"  Training data - mean: {X_train_scaled.mean().mean():.3f}, std: {X_train_scaled.std().mean():.3f}")
        
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
        scaler = None
        print("No scaling applied")
    
    return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, scaler, label_encoding



def print_data_summary(X_train, X_test, y_train, y_test, label_encoding):
    """
    Print summary of prepared training and test data.
    
    Args:
        X_train, X_test: Feature matrices
        y_train, y_test: Encoded target vectors  
        label_encoding: how class labels are encoded (regular -> 0, premium -> 1
        )
    """
    print_section_header("DATA PREP SUMMARY")
    print(f"Training features: {X_train.shape[0]} × {X_train.shape[1]}")
    print(f"Test features: {X_test.shape[0]} × {X_test.shape[1]}")
    
    # Class distribution in training set
    unique, counts = np.unique(y_train, return_counts=True)

    class_names = list(label_encoding.keys())

    # Print a little class distribution table
    print(f"\nTraining set class distribution:")
    for i, (class_idx, count) in enumerate(zip(unique, counts)):
        class_name = class_names[class_idx]
        pct = count / len(y_train) * 100
        print(f"  {class_name}: {count} ({pct:.1f}%)")
    
    print(f"\nFeatures: {list(X_train.columns)}")



def print_section_header(string):
    """
    consistent printing of banner headings 
    """
    print("=" * 50)
    print(f"{string}")
    print("=" * 50)