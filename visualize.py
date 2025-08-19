#!/usr/bin/env python3
"""
Data Visualization and Train/Validation/Test Split Module
This module handles data visualization and splitting data into train, validation, and test sets.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import yaml
import warnings
from logger import log
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def create_visualization_directory(output_directory: str) -> str:
    """
    Create visualization directory if it doesn't exist.
    
    Args:
        output_directory (str): Base output directory
        
    Returns:
        str: Path to visualization directory
    """
    viz_dir = os.path.join(output_directory, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    return viz_dir


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


def generate_basic_statistics(df: pd.DataFrame, viz_dir: str, time_window: str = None) -> Dict[str, Any]:
    """
    Generate basic statistics and save to file.
    
    Args:
        df (pd.DataFrame): Input dataframe
        viz_dir (str): Visualization directory
        time_window (str): Time window used for aggregation (e.g., '5min', '1H')
        
    Returns:
        Dict[str, Any]: Basic statistics
    """
    log.info("📊 Generating basic statistics...")
    
    # Basic statistics
    stats = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        'missing_values': df.isnull().sum().to_dict(),
        'data_types': df.dtypes.to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'datetime_columns': df.select_dtypes(include=['datetime64']).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    
    # Save statistics to file
    stats_file = os.path.join(viz_dir, "basic_statistics.txt")
    with open(stats_file, 'w') as f:
        f.write("BASIC DATA STATISTICS\n")
        f.write("=" * 50 + "\n\n")
        if time_window:
            f.write(f"Time Window: {time_window}\n\n")
        f.write(f"Total Rows: {stats['total_rows']:,}\n")
        f.write(f"Total Columns: {stats['total_columns']}\n")
        f.write(f"Memory Usage: {stats['memory_usage_mb']:.2f} MB\n\n")
        
        f.write("MISSING VALUES:\n")
        f.write("-" * 20 + "\n")
        for col, missing in stats['missing_values'].items():
            if missing > 0:
                f.write(f"{col}: {missing:,} ({missing/len(df)*100:.2f}%)\n")
        
        f.write("\nDATA TYPES:\n")
        f.write("-" * 20 + "\n")
        for col, dtype in stats['data_types'].items():
            f.write(f"{col}: {dtype}\n")
    
    log.info(f"✅ Basic statistics saved to {stats_file}")
    return stats


def create_time_series_plots(df: pd.DataFrame, viz_dir: str, time_column: str = 'start_time', time_window: str = None):
    """
    Create time series plots for key metrics.
    
    Args:
        df (pd.DataFrame): Input dataframe
        viz_dir (str): Visualization directory
        time_column (str): Name of time column
        time_window (str): Time window used for aggregation (e.g., '5min', '1H')
    """
    log.info("📈 Creating time series plots...")
    
    # Ensure time column is datetime
    if time_column in df.columns:
        df[time_column] = pd.to_datetime(df[time_column])
        df = df.sort_values(time_column)
    
    # Get numeric columns for plotting
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Create subplots for key metrics
    key_metrics = ['open_price', 'high', 'low', 'close_price', 'volume', 'spread']
    available_metrics = [col for col in key_metrics if col in numeric_cols]
    
    if not available_metrics:
        log.warning("⚠️  No key metrics found for time series plotting")
        return
    
    # Create time series plots
    fig, axes = plt.subplots(len(available_metrics), 1, figsize=(15, 3*len(available_metrics)))
    if len(available_metrics) == 1:
        axes = [axes]
    
    # Create main title with time window information
    if time_window:
        fig.suptitle(f'Time Series Analysis - {time_window} Aggregated Data', fontsize=16, y=0.98)
    
    for i, metric in enumerate(available_metrics):
        axes[i].plot(df[time_column], df[metric], linewidth=0.5, alpha=0.7)
        title = f'{metric.replace("_", " ").title()} Over Time'
        if time_window:
            title += f' ({time_window} intervals)'
        axes[i].set_title(title)
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    time_series_file = os.path.join(viz_dir, "time_series_plots.png")
    plt.savefig(time_series_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"✅ Time series plots saved to {time_series_file}")


def create_correlation_heatmap(df: pd.DataFrame, viz_dir: str, time_window: str = None):
    """
    Create correlation heatmap for numeric columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        viz_dir (str): Visualization directory
        time_window (str): Time window used for aggregation (e.g., '5min', '1H')
    """
    log.info("🔥 Creating correlation heatmap...")
    
    # Get numeric columns and limit to first 30 for correlation analysis
    numeric_df = df.select_dtypes(include=[np.number]).iloc[:, :30]
    
    if len(numeric_df.columns) < 2:
        log.warning("⚠️  Not enough numeric columns for correlation analysis")
        return
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    title = 'Correlation Heatmap'
    if time_window:
        title += f' - {time_window} Aggregated Data'
    plt.title(title)
    plt.tight_layout()
    
    correlation_file = os.path.join(viz_dir, "correlation_heatmap.png")
    plt.savefig(correlation_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"✅ Correlation heatmap saved to {correlation_file}")


def create_distribution_plots(df: pd.DataFrame, viz_dir: str, time_window: str = None):
    """
    Create distribution plots for numeric columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        viz_dir (str): Visualization directory
        time_window (str): Time window used for aggregation (e.g., '5min', '1H')
    """
    log.info("📊 Creating distribution plots...")
    
    # Get all numeric columns for visualization
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        log.warning("⚠️  No numeric columns found for distribution plots")
        return
    
    # Create distributions folder and clear it
    distributions_dir = os.path.join(viz_dir, "distributions")
    if os.path.exists(distributions_dir):
        # Clear the directory
        for file in os.listdir(distributions_dir):
            file_path = os.path.join(distributions_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        log.info(f"🗑️  Cleared existing files in {distributions_dir}")
    else:
        os.makedirs(distributions_dir, exist_ok=True)
        log.info(f"📁 Created distributions directory: {distributions_dir}")
    
    # Create individual distribution plots for each column
    for col in numeric_cols:
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        plt.hist(df[col].dropna(), bins=50, alpha=0.7, edgecolor='black', color='skyblue')
        plt.title(f'{col.replace("_", " ").title()} Distribution')
        if time_window:
            plt.title(f'{col.replace("_", " ").title()} Distribution - {time_window} Data')
        plt.xlabel(col.replace("_", " ").title())
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Save individual plot
        plot_filename = f"{col}_distribution.png"
        plot_path = os.path.join(distributions_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log.info(f"  ✅ Saved distribution plot: {plot_filename}")
    
    # Do not create combined subplot; only individual plots are saved
    log.info(f"✅ Individual distribution plots saved to {distributions_dir}")


def create_combined_distribution_plots(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                                     viz_dir: str, time_window: str = None):
    """
    Create distribution plots with train/val/test subplots for each numeric column.
    
    Args:
        train_df (pd.DataFrame): Training dataframe
        val_df (pd.DataFrame): Validation dataframe  
        test_df (pd.DataFrame): Test dataframe
        viz_dir (str): Visualization directory
        time_window (str): Time window used for aggregation (e.g., '5min', '1H')
    """
    log.info("📊 Creating combined distribution plots with train/val/test splits...")
    
    # Get all numeric columns from all dataframes
    train_cols = set(train_df.select_dtypes(include=[np.number]).columns.tolist())
    val_cols = set(val_df.select_dtypes(include=[np.number]).columns.tolist())
    test_cols = set(test_df.select_dtypes(include=[np.number]).columns.tolist())
    
    # Get common columns across all splits
    all_cols = train_cols.intersection(val_cols).intersection(test_cols)
    numeric_cols = list(all_cols)
    
    if not numeric_cols:
        log.warning("⚠️  No common numeric columns found across all splits for distribution plots")
        return
    
    # Create distributions folder and clear it
    distributions_dir = os.path.join(viz_dir, "distributions")
    if os.path.exists(distributions_dir):
        # Clear the directory
        for file in os.listdir(distributions_dir):
            file_path = os.path.join(distributions_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
        log.info(f"🗑️  Cleared existing files in {distributions_dir}")
    else:
        os.makedirs(distributions_dir, exist_ok=True)
        log.info(f"📁 Created distributions directory: {distributions_dir}")
    
    # Create individual distribution plots with train/val/test subplots for each column
    for col in numeric_cols:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Train subplot
        axes[0].hist(train_df[col].dropna(), bins=50, alpha=0.7, edgecolor='black', color='blue', label='Train')
        axes[0].set_title(f'{col.replace("_", " ").title()} - Train')
        axes[0].set_xlabel(col.replace("_", " ").title())
        axes[0].set_ylabel('Frequency')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Validation subplot
        axes[1].hist(val_df[col].dropna(), bins=50, alpha=0.7, edgecolor='black', color='orange', label='Validation')
        axes[1].set_title(f'{col.replace("_", " ").title()} - Validation')
        axes[1].set_xlabel(col.replace("_", " ").title())
        axes[1].set_ylabel('Frequency')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Test subplot
        axes[2].hist(test_df[col].dropna(), bins=50, alpha=0.7, edgecolor='black', color='red', label='Test')
        axes[2].set_title(f'{col.replace("_", " ").title()} - Test')
        axes[2].set_xlabel(col.replace("_", " ").title())
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        # Add overall title
        if time_window:
            fig.suptitle(f'{col.replace("_", " ").title()} Distribution - {time_window} Data', fontsize=16, y=0.98)
        else:
            fig.suptitle(f'{col.replace("_", " ").title()} Distribution', fontsize=16, y=0.98)
        
        plt.tight_layout()
        
        # Save individual plot
        plot_filename = f"{col}_distribution.png"
        plot_path = os.path.join(distributions_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        log.info(f"  ✅ Saved combined distribution plot: {plot_filename}")
    
    log.info(f"✅ Combined distribution plots saved to {distributions_dir}")


def create_box_plots(df: pd.DataFrame, viz_dir: str, time_window: str = None):
    """
    Create box plots for numeric columns to show outliers.
    
    Args:
        df (pd.DataFrame): Input dataframe
        viz_dir (str): Visualization directory
        time_window (str): Time window used for aggregation (e.g., '5min', '1H')
    """
    log.info("📦 Creating box plots...")
    
    # Get numeric columns and limit to first 20 for visualization
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:20]
    
    if not numeric_cols:
        log.warning("⚠️  No numeric columns found for box plots")
        return
    
    # Create subplots
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Add main title with time window information
    if time_window:
        fig.suptitle(f'Box Plot Analysis - {time_window} Aggregated Data', fontsize=16, y=0.98)
    
    for i, col in enumerate(numeric_cols):
        row = i // n_cols
        col_idx = i % n_cols
        
        # Create box plot
        axes[row, col_idx].boxplot(df[col].dropna())
        axes[row, col_idx].set_title(f'{col.replace("_", " ").title()} Box Plot')
        axes[row, col_idx].set_ylabel(col.replace("_", " ").title())
        axes[row, col_idx].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(len(numeric_cols), n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    boxplot_file = os.path.join(viz_dir, "box_plots.png")
    plt.savefig(boxplot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    log.info(f"✅ Box plots saved to {boxplot_file}")








def run_visualize(config: Dict[str, Any]):
    """
    Run the visualization pipeline for train/val/test datasets.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    try:
        log.info("🎨 Starting visualization pipeline...")
        
        # Extract configuration
        input_files = config['input_files']
        output_directory = config['output_directory']
        time_column = config.get('time_column', 'start_time')
        time_window = config.get('time_window', None)
        
        # Create visualization directory
        viz_dir = create_visualization_directory(output_directory)
        
        # Initialize dataframes for each split
        train_df = None
        val_df = None
        test_df = None
        
        # Process each split (train/val/test) separately
        for input_file in input_files:
            log.info(f"\n📊 Processing visualization for: {os.path.basename(input_file)}")
            
            # Load data
            df = load_data(input_file)
            
            # Extract split name from filename
            filename = os.path.basename(input_file)
            # Look for train, val, or test in the filename
            if '_train.parquet' in filename:
                split_name = 'train'
                train_df = df
            elif '_val.parquet' in filename:
                split_name = 'val'
                val_df = df
            elif '_test.parquet' in filename:
                split_name = 'test'
                test_df = df
            else:
                split_name = 'unknown'
            
            # Create split-specific visualization directory
            split_viz_dir = os.path.join(viz_dir, split_name)
            os.makedirs(split_viz_dir, exist_ok=True)
            
            # Generate basic statistics
            stats = generate_basic_statistics(df, split_viz_dir, time_window)
            
            # Create visualizations (except distribution plots - will be done combined)
            create_time_series_plots(df, split_viz_dir, time_column, time_window)
            create_correlation_heatmap(df, split_viz_dir, time_window)
            create_box_plots(df, split_viz_dir, time_window)
            
            log.info(f"✅ Visualization completed for {split_name} dataset")
        
        # Create combined distribution plots with train/val/test splits
        if train_df is not None and val_df is not None and test_df is not None:
            log.info("\n📊 Creating combined distribution plots with train/val/test splits...")
            create_combined_distribution_plots(train_df, val_df, test_df, viz_dir, time_window)
        else:
            log.warning("⚠️  Not all splits (train/val/test) found, skipping combined distribution plots")
        
        log.info("✅ Visualization pipeline completed successfully!")
        log.info(f"📁 Visualizations saved to: {viz_dir}")
        
    except Exception as e:
        error_msg = f"Visualization pipeline failed: {e}"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise e


if __name__ == "__main__":
    # Example usage
    config = {
        'input_file': 'data/5min_forex_data_cleaned_forex_data_featurized_cleaned.parquet',
        'output_directory': 'data',
        'time_column': 'start_time',
        'time_window': '5min',
        'split': {
            'train': 0.7,
            'val': 0.15,
            'test': 0.15
        }
    }
    
    run_visualize(config) 