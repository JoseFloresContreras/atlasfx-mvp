#!/usr/bin/env python3
"""
Data Splitting Module
This module handles splitting data into train, validation, and test sets.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import train_test_split
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


def split_data(df: pd.DataFrame, split_config: Dict[str, float], output_directory: str, base_filename: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df (pd.DataFrame): Input dataframe
        split_config (Dict[str, float]): Split configuration with train, val, test ratios
        output_directory (str): Output directory for saving splits
        base_filename (str): Base filename without extension
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test dataframes
    """
    log.info("✂️  Splitting data into train/validation/test sets...")
    
    # Get split ratios
    train_ratio = split_config.get('train', 0.7)
    val_ratio = split_config.get('val', 0.15)
    test_ratio = split_config.get('test', 0.15)
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        log.warning(f"⚠️  Warning: Split ratios sum to {total_ratio:.3f}, normalizing to 1.0")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
    
    log.info(f"📊 Split ratios - Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}")
    
    # First split: train + temp, test
    train_temp, test_df = train_test_split(df, test_size=test_ratio, random_state=42, shuffle=False)
    
    # Second split: train, validation
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(train_temp, test_size=val_ratio_adjusted, random_state=42, shuffle=False)
    
    log.info(f"✅ Split complete:")
    log.info(f"   Train: {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%)")
    log.info(f"   Validation: {len(val_df):,} rows ({len(val_df)/len(df)*100:.1f}%)")
    log.info(f"   Test: {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%)")
    
    # Save splits with suffixes
    train_file = os.path.join(output_directory, f"{base_filename}_train.parquet")
    val_file = os.path.join(output_directory, f"{base_filename}_val.parquet")
    test_file = os.path.join(output_directory, f"{base_filename}_test.parquet")
    
    train_df.to_parquet(train_file, index=False)
    val_df.to_parquet(val_file, index=False)
    test_df.to_parquet(test_file, index=False)
    
    log.info(f"✅ Data splits saved:")
    log.info(f"   Train: {train_file}")
    log.info(f"   Validation: {val_file}")
    log.info(f"   Test: {test_file}")
    
    return train_df, val_df, test_df


def run_split(config: Dict[str, Any]):
    """
    Run the data splitting pipeline.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    try:
        log.info("✂️  Starting data splitting pipeline...")
        
        # Extract configuration
        input_file = config['input_file']
        output_directory = config['output_directory']
        split_config = config.get('split', {'train': 0.7, 'val': 0.15, 'test': 0.15})
        
        # Extract base filename from input file
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        
        # Load data
        df = load_data(input_file)
        
        # Split data
        train_df, val_df, test_df = split_data(df, split_config, output_directory, base_filename)
        
        log.info("✅ Data splitting pipeline completed successfully!")
        
    except Exception as e:
        error_msg = f"Data splitting pipeline failed: {e}"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise e


if __name__ == "__main__":
    # Example usage
    config = {
        'input_file': 'data/1H_forex_data_cleaned.parquet',
        'output_directory': 'data',
        'split': {
            'train': 0.7,
            'val': 0.15,
            'test': 0.15
        }
    }
    
    run_split(config) 