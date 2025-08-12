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
from typing import Dict, List, Any, Tuple
import yaml
from sklearn.model_selection import train_test_split
import warnings
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
    print(f"📊 Loading data from {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"✅ Loaded {len(df):,} rows and {len(df.columns)} columns")
    return df


def generate_basic_statistics(df: pd.DataFrame, viz_dir: str) -> Dict[str, Any]:
    """
    Generate basic statistics and save to file.
    
    Args:
        df (pd.DataFrame): Input dataframe
        viz_dir (str): Visualization directory
        
    Returns:
        Dict[str, Any]: Basic statistics
    """
    print("📊 Generating basic statistics...")
    
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
    
    print(f"✅ Basic statistics saved to {stats_file}")
    return stats


def create_time_series_plots(df: pd.DataFrame, viz_dir: str, time_column: str = 'start_time'):
    """
    Create time series plots for key metrics.
    
    Args:
        df (pd.DataFrame): Input dataframe
        viz_dir (str): Visualization directory
        time_column (str): Name of time column
    """
    print("📈 Creating time series plots...")
    
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
        print("⚠️  No key metrics found for time series plotting")
        return
    
    # Create time series plots
    fig, axes = plt.subplots(len(available_metrics), 1, figsize=(15, 3*len(available_metrics)))
    if len(available_metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(available_metrics):
        axes[i].plot(df[time_column], df[metric], linewidth=0.5, alpha=0.7)
        axes[i].set_title(f'{metric.replace("_", " ").title()} Over Time')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    time_series_file = os.path.join(viz_dir, "time_series_plots.png")
    plt.savefig(time_series_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Time series plots saved to {time_series_file}")


def create_correlation_heatmap(df: pd.DataFrame, viz_dir: str):
    """
    Create correlation heatmap for numeric columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        viz_dir (str): Visualization directory
    """
    print("🔥 Creating correlation heatmap...")
    
    # Get numeric columns and limit to first 30 for correlation analysis
    numeric_df = df.select_dtypes(include=[np.number]).iloc[:, :30]
    
    if len(numeric_df.columns) < 2:
        print("⚠️  Not enough numeric columns for correlation analysis")
        return
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    
    correlation_file = os.path.join(viz_dir, "correlation_heatmap.png")
    plt.savefig(correlation_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Correlation heatmap saved to {correlation_file}")


def create_distribution_plots(df: pd.DataFrame, viz_dir: str):
    """
    Create distribution plots for numeric columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        viz_dir (str): Visualization directory
    """
    print("📊 Creating distribution plots...")
    
    # Get numeric columns and limit to first 20 for visualization
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:20]
    
    if not numeric_cols:
        print("⚠️  No numeric columns found for distribution plots")
        return
    
    # Create subplots
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, col in enumerate(numeric_cols):
        row = i // n_cols
        col_idx = i % n_cols
        
        # Create histogram
        axes[row, col_idx].hist(df[col].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[row, col_idx].set_title(f'{col.replace("_", " ").title()} Distribution')
        axes[row, col_idx].set_xlabel(col.replace("_", " ").title())
        axes[row, col_idx].set_ylabel('Frequency')
        axes[row, col_idx].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(len(numeric_cols), n_rows * n_cols):
        row = i // n_cols
        col_idx = i % n_cols
        axes[row, col_idx].set_visible(False)
    
    plt.tight_layout()
    distribution_file = os.path.join(viz_dir, "distribution_plots.png")
    plt.savefig(distribution_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Distribution plots saved to {distribution_file}")


def create_box_plots(df: pd.DataFrame, viz_dir: str):
    """
    Create box plots for numeric columns to show outliers.
    
    Args:
        df (pd.DataFrame): Input dataframe
        viz_dir (str): Visualization directory
    """
    print("📦 Creating box plots...")
    
    # Get numeric columns and limit to first 20 for visualization
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()[:20]
    
    if not numeric_cols:
        print("⚠️  No numeric columns found for box plots")
        return
    
    # Create subplots
    n_cols = min(3, len(numeric_cols))
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)
    
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
    
    print(f"✅ Box plots saved to {boxplot_file}")


def split_data(df: pd.DataFrame, split_config: Dict[str, float], output_directory: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets.
    
    Args:
        df (pd.DataFrame): Input dataframe
        split_config (Dict[str, float]): Split configuration with train, val, test ratios
        output_directory (str): Output directory for saving splits
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Train, validation, and test dataframes
    """
    print("✂️  Splitting data into train/validation/test sets...")
    
    # Get split ratios
    train_ratio = split_config.get('train', 0.7)
    val_ratio = split_config.get('val', 0.15)
    test_ratio = split_config.get('test', 0.15)
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        print(f"⚠️  Warning: Split ratios sum to {total_ratio:.3f}, normalizing to 1.0")
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio
    
    print(f"📊 Split ratios - Train: {train_ratio:.1%}, Val: {val_ratio:.1%}, Test: {test_ratio:.1%}")
    
    # First split: train + temp, test
    train_temp, test_df = train_test_split(df, test_size=test_ratio, random_state=42, shuffle=False)
    
    # Second split: train, validation
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_df, val_df = train_test_split(train_temp, test_size=val_ratio_adjusted, random_state=42, shuffle=False)
    
    print(f"✅ Split complete:")
    print(f"   Train: {len(train_df):,} rows ({len(train_df)/len(df)*100:.1f}%)")
    print(f"   Validation: {len(val_df):,} rows ({len(val_df)/len(df)*100:.1f}%)")
    print(f"   Test: {len(test_df):,} rows ({len(test_df)/len(df)*100:.1f}%)")
    
    # Save splits
    splits_dir = os.path.join(output_directory, "splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    train_file = os.path.join(splits_dir, "train.parquet")
    val_file = os.path.join(splits_dir, "validation.parquet")
    test_file = os.path.join(splits_dir, "test.parquet")
    
    train_df.to_parquet(train_file, index=False)
    val_df.to_parquet(val_file, index=False)
    test_df.to_parquet(test_file, index=False)
    
    print(f"✅ Data splits saved to {splits_dir}/")
    
    return train_df, val_df, test_df


def create_split_summary(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, viz_dir: str):
    """
    Create summary statistics for each split.
    
    Args:
        train_df (pd.DataFrame): Training data
        val_df (pd.DataFrame): Validation data
        test_df (pd.DataFrame): Test data
        viz_dir (str): Visualization directory
    """
    print("📋 Creating split summary...")
    
    # Calculate basic statistics for each split
    splits = {
        'Train': train_df,
        'Validation': val_df,
        'Test': test_df
    }
    
    summary_file = os.path.join(viz_dir, "split_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("DATA SPLIT SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        for split_name, split_df in splits.items():
            f.write(f"{split_name.upper()} SET:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Rows: {len(split_df):,}\n")
            f.write(f"Columns: {len(split_df.columns)}\n")
            f.write(f"Memory Usage: {split_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB\n")
            f.write(f"Missing Values: {split_df.isnull().sum().sum()}\n\n")
            
            # Time range if time column exists
            time_cols = split_df.select_dtypes(include=['datetime64']).columns.tolist()
            if time_cols:
                time_col = time_cols[0]
                f.write(f"Time Range: {split_df[time_col].min()} to {split_df[time_col].max()}\n\n")
    
    print(f"✅ Split summary saved to {summary_file}")


def run_visualize(config: Dict[str, Any]):
    """
    Run the visualization and data splitting pipeline.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
    """
    print("🎨 Starting visualization and data splitting pipeline...")
    
    # Extract configuration
    input_file = config['input_file']
    output_directory = config['output_directory']
    time_column = config.get('time_column', 'start_time')
    split_config = config.get('split', {'train': 0.7, 'val': 0.15, 'test': 0.15})
    
    # Create visualization directory
    viz_dir = create_visualization_directory(output_directory)
    
    # Load data
    df = load_data(input_file)
    
    # Generate basic statistics
    stats = generate_basic_statistics(df, viz_dir)
    
    # Create visualizations
    create_time_series_plots(df, viz_dir, time_column)
    create_correlation_heatmap(df, viz_dir)
    create_distribution_plots(df, viz_dir)
    create_box_plots(df, viz_dir)
    
    # Split data
    train_df, val_df, test_df = split_data(df, split_config, output_directory)
    
    # Create split summary
    create_split_summary(train_df, val_df, test_df, viz_dir)
    
    print("✅ Visualization and data splitting pipeline completed successfully!")
    print(f"📁 Visualizations saved to: {viz_dir}")
    print(f"📁 Data splits saved to: {os.path.join(output_directory, 'splits')}")


if __name__ == "__main__":
    # Example usage
    config = {
        'input_file': 'data/5min_forex_data_cleaned_forex_data_augmented_cleaned.parquet',
        'output_directory': 'data',
        'time_column': 'start_time',
        'split': {
            'train': 0.7,
            'val': 0.15,
            'test': 0.15
        }
    }
    
    run_visualize(config) 