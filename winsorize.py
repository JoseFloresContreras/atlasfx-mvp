#!/usr/bin/env python3
"""
Data Winsorization Module
This module handles winsorization of specified columns with configurable percentiles.
Uses train set ranges to apply to val and test sets to prevent data leakage.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from scipy import stats
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


def calculate_winsorization_bounds(df: pd.DataFrame, aggregations: List[str], high_percentile: float, low_percentile: float) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    """
    Calculate winsorization bounds for columns matching aggregation keywords.
    
    Args:
        df (pd.DataFrame): Input dataframe
        aggregations (List[str]): List of aggregation keywords to match in column names
        high_percentile (float): High percentile for winsorization (e.g., 0.95)
        low_percentile (float): Low percentile for winsorization (e.g., 0.05)
        
    Returns:
        Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]: 
            - Dictionary mapping column names to (lower_bound, upper_bound)
            - Dictionary mapping column names to (low_percentile, high_percentile)
    """
    bounds = {}
    percentiles = {}
    
    for aggregation in aggregations:
        # Find all columns that contain the aggregation keyword
        matching_columns = [col for col in df.columns if aggregation in col]
        
        if matching_columns:
            log.info(f"🔍 Found {len(matching_columns)} columns matching '{aggregation}': {matching_columns}")
            
            for column in matching_columns:
                # Calculate percentiles
                lower_bound = df[column].quantile(low_percentile)
                upper_bound = df[column].quantile(high_percentile)
                bounds[column] = (lower_bound, upper_bound)
                percentiles[column] = (low_percentile, high_percentile)
                log.info(f"📊 {column}: bounds [{lower_bound:.4f}, {upper_bound:.4f}] (percentiles [{low_percentile:.1%}, {high_percentile:.1%}])")
        else:
            log.warning(f"⚠️  No columns found matching '{aggregation}', skipping...")
    
    return bounds, percentiles


def apply_winsorization(df: pd.DataFrame, bounds: Dict[str, Tuple[float, float]], percentiles: Dict[str, Tuple[float, float]], suffix: str = "") -> pd.DataFrame:
    """
    Apply winsorization to dataframe using pre-calculated bounds.
    
    Args:
        df (pd.DataFrame): Input dataframe
        bounds (Dict[str, Tuple[float, float]]): Dictionary mapping column names to (lower_bound, upper_bound)
        percentiles (Dict[str, Tuple[float, float]]): Dictionary mapping column names to (low_percentile, high_percentile)
        suffix (str): Suffix to add to column names after winsorization
        
    Returns:
        pd.DataFrame: Dataframe with winsorized columns
    """
    df_winsorized = df.copy()
    
    for column, (lower_bound, upper_bound) in bounds.items():
        if column in df_winsorized.columns:
            # Apply winsorization
            df_winsorized[column] = np.clip(df_winsorized[column], lower_bound, upper_bound)
            
            # Get percentiles for this column
            low_pct, high_pct = percentiles[column]
            
            # Rename column with winsorization percentile info (both lower and upper)
            new_column_name = f"{column} (Winsorized {low_pct:.2f}-{high_pct:.2f})"
            df_winsorized = df_winsorized.rename(columns={column: new_column_name})
            
            log.info(f"✅ Applied winsorization to {column} -> {new_column_name}")
    
    return df_winsorized


def process_winsorization(input_files: List[str], output_directory: str, winsorization_configs: List[Dict[str, Any]]) -> bool:
    """
    Process winsorization for train/val/test files.
    
    Args:
        input_files (List[str]): List of input file paths (train, val, test)
        output_directory (str): Output directory
        winsorization_configs (List[Dict[str, Any]]): List of winsorization configurations
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        log.info("🔧 Processing winsorization for train/val/test datasets...")
        
        # Separate train, val, test files
        train_file = None
        val_file = None
        test_file = None
        
        for file_path in input_files:
            filename = os.path.basename(file_path)
            if '_train' in filename:
                train_file = file_path
            elif '_val' in filename:
                val_file = file_path
            elif '_test' in filename:
                test_file = file_path
        
        if not all([train_file, val_file, test_file]):
            error_msg = "Could not identify train, val, and test files"
            log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
            return False
        
        # Load train data first to calculate bounds
        log.info("📊 Loading train data to calculate winsorization bounds...")
        train_df = load_data(train_file)
        
        # Calculate bounds for all configurations using train data
        all_bounds = {}
        all_percentiles = {}
        for config in winsorization_configs:
            aggregations = config['aggregations']
            high_percentile = config['high']
            low_percentile = config['low']
            
            log.info(f"🔧 Calculating bounds for aggregations: {aggregations}")
            bounds, percentiles = calculate_winsorization_bounds(train_df, aggregations, high_percentile, low_percentile)
            all_bounds.update(bounds)
            all_percentiles.update(percentiles)
        
        # Process each dataset
        datasets = [
            ('train', train_file),
            ('val', val_file),
            ('test', test_file)
        ]
        
        for dataset_name, file_path in datasets:
            log.info(f"\n📊 Processing {dataset_name} dataset...")
            
            # Load data
            df = load_data(file_path)
            
            # Apply winsorization using train-calculated bounds
            df_winsorized = apply_winsorization(df, all_bounds, all_percentiles)
            
            # Save winsorized data
            input_filename = os.path.splitext(os.path.basename(file_path))[0]
            output_filename = f"{input_filename}.parquet"
            output_path = os.path.join(output_directory, output_filename)
            
            df_winsorized.to_parquet(output_path, index=False)
            log.info(f"✅ Saved winsorized {dataset_name} data to: {output_path}")
        
        log.info("✅ Winsorization processing completed successfully!")
        return True
        
    except Exception as e:
        error_msg = f"Winsorization processing failed: {e}"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        return False


def run_winsorize(config: Dict[str, Any]):
    """
    Run the winsorization pipeline.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    try:
        log.info("🔧 Starting winsorization pipeline...")
        
        # Extract configuration
        input_files = config['input_files']
        output_directory = config['output_directory']
        winsorization_configs = config['winsorization_configs']
        
        log.info(f"📁 Input files: {len(input_files)} files to process")
        log.info(f"📁 Output directory: {output_directory}")
        log.info(f"🔧 Winsorization configurations: {len(winsorization_configs)} configs")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        
        # Process winsorization
        success = process_winsorization(input_files, output_directory, winsorization_configs)
        
        if success:
            log.info("✅ Winsorization pipeline completed successfully!")
        else:
            log.error("❌ Winsorization pipeline failed!")
            
    except Exception as e:
        error_msg = f"Winsorization pipeline failed: {e}"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise e


if __name__ == "__main__":
    # Example usage
    config = {
        'input_files': [
            'data/1H_forex_data_train.parquet',
            'data/1H_forex_data_val.parquet',
            'data/1H_forex_data_test.parquet'
        ],
        'output_directory': 'data',
        'winsorization_configs': [
            {
                'columns': ['tick_count', 'volatility'],
                'high': 0.95,
                'low': 0.05
            },
            {
                'columns': ['volume'],
                'high': 0.99,
                'low': 0.01
            }
        ]
    }
    
    run_winsorize(config) 