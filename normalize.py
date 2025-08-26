#!/usr/bin/env python3
"""
Data Normalization Module
This module handles normalization of feature columns using mean subtraction and standard deviation division.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import pickle
from logger import log


def load_data(input_file: str) -> pd.DataFrame:
    """
    Load data from parquet file.
    
    Args:
        input_file (str): Path to input parquet file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        log.info(f"📊 Loading data from {input_file}...")
        df = pd.read_parquet(input_file)
        log.info(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns")
        return df
    except FileNotFoundError:
        error_msg = f"Input file not found: {input_file}"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise FileNotFoundError(error_msg)
    except Exception as e:
        error_msg = f"Error loading data from {input_file}: {e}"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise Exception(error_msg)


def save_data(df: pd.DataFrame, output_file: str):
    """
    Save data to parquet file.
    
    Args:
        df (pd.DataFrame): Dataframe to save
        output_file (str): Path to output file
    """
    try:
        log.info(f"💾 Saving data to {output_file}...")
        df.to_parquet(output_file, index=False)
        log.info(f"✅ Saved {len(df):,} rows and {len(df.columns)} columns")
    except Exception as e:
        error_msg = f"Error saving data to {output_file}: {e}"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise Exception(error_msg)


def identify_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify feature columns for normalization.
    All columns with '[Feature]' prefix will be normalized.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        List[str]: List of feature column names
    """
    # Get all columns
    all_cols = df.columns.tolist()
    
    # Filter columns that contain the word "[Feature]"
    feature_cols = [col for col in all_cols if '[Feature]' in col]
    
    log.info(f"📊 Identified {len(feature_cols)} feature columns for normalization (with '[Feature]' prefix)")
    log.info(f"📋 Feature columns: {feature_cols}")
    
    return feature_cols


def compute_normalization_stats(train_df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Compute mean and standard deviation from training data.
    
    Args:
        train_df (pd.DataFrame): Training dataframe
        feature_cols (List[str]): List of feature columns
        
    Returns:
        Dict[str, Dict[str, float]]: Dictionary with mean and std for each feature
    """
    log.info("📊 Computing normalization statistics from training data...")
    
    stats = {}
    for col in feature_cols:
        # Compute mean and std, handling NaN values
        mean_val = train_df[col].mean()
        std_val = train_df[col].std()
        
        # Handle zero standard deviation
        if std_val == 0:
            log.warning(f"⚠️  Zero standard deviation for column {col}, using std=1")
            std_val = 1.0
        
        stats[col] = {
            'mean': mean_val,
            'std': std_val
        }
        
        log.info(f"  {col}: mean={mean_val:.6f}, std={std_val:.6f}")
    
    return stats


def normalize_dataframe(df: pd.DataFrame, feature_cols: List[str], stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Normalize dataframe using pre-computed statistics.
    
    Args:
        df (pd.DataFrame): Dataframe to normalize
        feature_cols (List[str]): List of feature columns
        stats (Dict[str, Dict[str, float]]): Normalization statistics
        
    Returns:
        pd.DataFrame: Normalized dataframe
    """
    log.info("📊 Normalizing dataframe...")
    
    # Create a copy to avoid modifying original
    normalized_df = df.copy()
    
    for col in feature_cols:
        if col in stats:
            mean_val = stats[col]['mean']
            std_val = stats[col]['std']
            
            # Apply normalization: (x - mean) / std
            normalized_df[col] = (normalized_df[col] - mean_val) / std_val
            
            log.info(f"  Normalized {col}: mean={normalized_df[col].mean():.6f}, std={normalized_df[col].std():.6f}")
    
    return normalized_df


def save_normalization_stats(stats: Dict[str, Dict[str, float]], output_directory: str, time_window: str):
    """
    Save normalization statistics to file.
    
    Args:
        stats (Dict[str, Dict[str, float]]): Normalization statistics
        output_directory (str): Output directory
        time_window (str): Time window used for aggregation
    """
    stats_file = os.path.join(output_directory, f"{time_window}_normalization_stats.pkl")
    
    try:
        with open(stats_file, 'wb') as f:
            pickle.dump(stats, f)
        log.info(f"✅ Normalization statistics saved to {stats_file}")
    except Exception as e:
        error_msg = f"Error saving normalization statistics: {e}"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise Exception(error_msg)


def run_normalize(config: Dict[str, Any]):
    """
    Run the normalization pipeline for train/val/test datasets.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    try:
        log.info("🎯 Starting normalization pipeline...")
        
        # Extract configuration
        input_files = config['input_files']
        output_directory = config['output_directory']
        time_window = config.get('time_window', None)
        
        # Find train file and load it
        train_file = None
        for input_file in input_files:
            if 'train.parquet' in input_file:
                train_file = input_file
                break
        
        if train_file is None:
            error_msg = "Train file not found. Cannot proceed with normalization."
            log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
            raise ValueError(error_msg)
        
        # Load training data to calculate parameters
        log.info(f"📊 Loading training data from: {os.path.basename(train_file)}")
        train_df = load_data(train_file)
        
        # Identify feature columns
        log.info("📊 Identifying feature columns for normalization...")
        feature_cols = identify_feature_columns(train_df)
        
        if not feature_cols:
            error_msg = "No feature columns found for normalization."
            log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
            raise ValueError(error_msg)
        
        # Compute normalization statistics from training data
        log.info("📊 Computing normalization statistics...")
        normalization_stats = compute_normalization_stats(train_df, feature_cols)
        
        # Save normalization statistics
        log.info("💾 Saving normalization statistics...")
        save_normalization_stats(normalization_stats, output_directory, time_window)
        
        # Process each input file
        log.info("📊 Normalizing datasets...")
        for input_file in input_files:
            log.info(f"📊 Processing: {os.path.basename(input_file)}")
            
            # Load data
            df = load_data(input_file)
            
            # Normalize data
            normalized_df = normalize_dataframe(df, feature_cols, normalization_stats)
            
            # Save with same filename
            save_data(normalized_df, input_file)
            
            log.info(f"✅ Normalized and saved: {os.path.basename(input_file)}")
        
        log.info("✅ Normalization pipeline completed successfully!")
        log.info(f"📊 Normalization statistics saved for future use")
        
    except Exception as e:
        error_msg = f"Normalization pipeline failed: {e}"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise e


if __name__ == "__main__":
    # Example usage
    config = {
        'input_files': [
            'data/5min_forex_data_train.parquet',
            'data/5min_forex_data_val.parquet', 
            'data/5min_forex_data_test.parquet'
        ],
        'output_directory': 'data',
        'time_window': '5min'
    }
    
    run_normalize(config) 