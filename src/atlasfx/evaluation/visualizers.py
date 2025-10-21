#!/usr/bin/env python3
"""
Data Visualization and Train/Validation/Test Split Module
This module handles data visualization and splitting data into train, validation, and test sets.
"""

import os
import warnings
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from atlasfx.utils.logging import log

warnings.filterwarnings("ignore")

# Set style for better plots
plt.style.use("seaborn-v0_8")
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
    except FileNotFoundError as e:
        error_msg = f"Input file not found: {input_file}"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise FileNotFoundError(error_msg) from e
    except Exception as e:
        error_msg = f"Error loading data from {input_file}: {e}"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise Exception(error_msg) from e


def generate_basic_statistics(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    viz_dir: str,
    time_window: str = None,
) -> dict[str, Any]:
    """
    Generate basic statistics for all splits and save to file in a comparison-friendly format.

    Args:
        train_df (pd.DataFrame): Training dataframe
        val_df (pd.DataFrame): Validation dataframe
        test_df (pd.DataFrame): Test dataframe
        viz_dir (str): Visualization directory
        time_window (str): Time window used for aggregation (e.g., '5min', '1H')

    Returns:
        dict[str, Any]: Basic statistics
    """
    log.info("📊 Generating basic statistics...")

    # Basic statistics for each split
    splits = {"train": train_df, "val": val_df, "test": test_df}

    stats = {}
    for split_name, df in splits.items():
        # Get numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Calculate mean and std for numeric columns
        numeric_stats = {}
        for col in numeric_cols:
            numeric_stats[col] = {
                "mean": df[col].mean(),
                "std": df[col].std(),
                "min": df[col].min(),
                "max": df[col].max(),
            }

        stats[split_name] = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "missing_values": df.isnull().sum().to_dict(),
            "data_types": df.dtypes.to_dict(),
            "numeric_columns": numeric_cols,
            "numeric_stats": numeric_stats,
            "categorical_columns": df.select_dtypes(include=["object"]).columns.tolist(),
        }

    # Save statistics to file in comparison format
    stats_file = os.path.join(viz_dir, "basic_statistics.txt")
    with open(stats_file, "w") as f:
        f.write("BASIC DATA STATISTICS - COMPARISON FORMAT\n")
        f.write("=" * 60 + "\n\n")
        if time_window:
            f.write(f"Time Window: {time_window}\n\n")

        # Overall dataset comparison
        f.write("DATASET OVERVIEW:\n")
        f.write("-" * 20 + "\n")
        f.write(f"{'Metric':<20} {'Train':<15} {'Val':<15} {'Test':<15}\n")
        f.write("-" * 65 + "\n")
        f.write(
            f"{'Total Rows':<20} {stats['train']['total_rows']:<15,} {stats['val']['total_rows']:<15,} {stats['test']['total_rows']:<15,}\n"
        )
        f.write(
            f"{'Total Columns':<20} {stats['train']['total_columns']:<15} {stats['val']['total_columns']:<15} {stats['test']['total_columns']:<15}\n"
        )
        f.write(
            f"{'Memory (MB)':<20} {stats['train']['memory_usage_mb']:<15.2f} {stats['val']['memory_usage_mb']:<15.2f} {stats['test']['memory_usage_mb']:<15.2f}\n"
        )
        f.write(
            f"{'Numeric Cols':<20} {len(stats['train']['numeric_columns']):<15} {len(stats['val']['numeric_columns']):<15} {len(stats['test']['numeric_columns']):<15}\n"
        )
        f.write(
            f"{'Categorical Cols':<20} {len(stats['train']['categorical_columns']):<15} {len(stats['val']['categorical_columns']):<15} {len(stats['test']['categorical_columns']):<15}\n"
        )
        f.write("\n" + "=" * 60 + "\n\n")

        # Numeric columns statistics comparison
        f.write("NUMERIC COLUMNS STATISTICS:\n")
        f.write("-" * 30 + "\n")

        # Get all numeric columns from all splits
        all_numeric_cols = set()
        for split_name, split_stats in stats.items():
            all_numeric_cols.update(split_stats["numeric_columns"])

        for col in sorted(all_numeric_cols):
            f.write(f"\n{col}:\n")
            f.write(f"{'Split':<10} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}\n")
            f.write("-" * 75 + "\n")

            for split_name in ["train", "val", "test"]:
                if col in stats[split_name]["numeric_stats"]:
                    col_stats = stats[split_name]["numeric_stats"][col]
                    f.write(
                        f"{split_name:<10} {col_stats['mean']:<15.6f} {col_stats['std']:<15.6f} {col_stats['min']:<15.6f} {col_stats['max']:<15.6f}\n"
                    )
                else:
                    f.write(f"{split_name:<10} {'N/A':<15} {'N/A':<15} {'N/A':<15} {'N/A':<15}\n")

        f.write("\n" + "=" * 60 + "\n\n")

        # Missing values comparison
        f.write("MISSING VALUES COMPARISON:\n")
        f.write("-" * 25 + "\n")

        # Get all columns that have missing values in any split
        all_missing_cols = set()
        for split_name, split_stats in stats.items():
            for col, missing in split_stats["missing_values"].items():
                if missing > 0:
                    all_missing_cols.add(col)

        if all_missing_cols:
            f.write(f"{'Column':<30} {'Train':<15} {'Val':<15} {'Test':<15}\n")
            f.write("-" * 75 + "\n")
            for col in sorted(all_missing_cols):
                train_missing = stats["train"]["missing_values"].get(col, 0)
                val_missing = stats["val"]["missing_values"].get(col, 0)
                test_missing = stats["test"]["missing_values"].get(col, 0)

                train_pct = (
                    (train_missing / stats["train"]["total_rows"] * 100) if train_missing > 0 else 0
                )
                val_pct = (val_missing / stats["val"]["total_rows"] * 100) if val_missing > 0 else 0
                test_pct = (
                    (test_missing / stats["test"]["total_rows"] * 100) if test_missing > 0 else 0
                )

                f.write(f"{col:<30} {train_missing:<15,} {val_missing:<15,} {test_missing:<15,}\n")
                f.write(
                    f"{'  (% of total)':<30} {train_pct:<15.2f}% {val_pct:<15.2f}% {test_pct:<15.2f}%\n"
                )
        else:
            f.write("No missing values found in any dataset.\n")

        f.write("\n" + "=" * 60 + "\n\n")

        # Data types comparison
        f.write("DATA TYPES COMPARISON:\n")
        f.write("-" * 20 + "\n")
        f.write(f"{'Column':<30} {'Train':<15} {'Val':<15} {'Test':<15}\n")
        f.write("-" * 75 + "\n")

        # Get all columns from all splits
        all_cols = set()
        for split_name, split_stats in stats.items():
            all_cols.update(split_stats["data_types"].keys())

        for col in sorted(all_cols):
            train_dtype = stats["train"]["data_types"].get(col, "N/A")
            val_dtype = stats["val"]["data_types"].get(col, "N/A")
            test_dtype = stats["test"]["data_types"].get(col, "N/A")

            f.write(
                f"{col:<30} {str(train_dtype):<15} {str(val_dtype):<15} {str(test_dtype):<15}\n"
            )

        f.write("\n" + "=" * 60 + "\n\n")

        # Detailed split information
        f.write("DETAILED SPLIT INFORMATION:\n")
        f.write("=" * 60 + "\n\n")

        for split_name, split_stats in stats.items():
            f.write(f"{split_name.upper()} SPLIT DETAILS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Rows: {split_stats['total_rows']:,}\n")
            f.write(f"Total Columns: {split_stats['total_columns']}\n")
            f.write(f"Memory Usage: {split_stats['memory_usage_mb']:.2f} MB\n")
            f.write(f"Numeric Columns: {len(split_stats['numeric_columns'])}\n")
            f.write(f"Categorical Columns: {len(split_stats['categorical_columns'])}\n\n")

            if split_stats["numeric_columns"]:
                f.write("Numeric Columns:\n")
                for col in split_stats["numeric_columns"]:
                    f.write(f"  - {col}\n")
                f.write("\n")

            if split_stats["categorical_columns"]:
                f.write("Categorical Columns:\n")
                for col in split_stats["categorical_columns"]:
                    f.write(f"  - {col}\n")
                f.write("\n")

            f.write("-" * 60 + "\n\n")

    log.info(f"✅ Basic statistics saved to {stats_file}")
    return stats


def create_distribution_plots(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    viz_dir: str,
    time_window: str = None,
):
    """
    Create distribution plots with train/val/test subplots for each numeric column.

    Args:
        train_df (pd.DataFrame): Training dataframe
        val_df (pd.DataFrame): Validation dataframe
        test_df (pd.DataFrame): Test dataframe
        viz_dir (str): Visualization directory
        time_window (str): Time window used for aggregation (e.g., '5min', '1H')
    """
    log.info("📊 Creating distribution plots with train/val/test splits...")

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
    distributions_dir = os.path.join(viz_dir, "distributions_plots")
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
        axes[0].hist(
            train_df[col].dropna(),
            bins=50,
            alpha=0.7,
            edgecolor="black",
            color="blue",
            label="Train",
        )
        axes[0].set_title(f'{col.replace("_", " ").title()} - Train')
        axes[0].set_xlabel(col.replace("_", " ").title())
        axes[0].set_ylabel("Frequency")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # Validation subplot
        axes[1].hist(
            val_df[col].dropna(),
            bins=50,
            alpha=0.7,
            edgecolor="black",
            color="orange",
            label="Validation",
        )
        axes[1].set_title(f'{col.replace("_", " ").title()} - Validation')
        axes[1].set_xlabel(col.replace("_", " ").title())
        axes[1].set_ylabel("Frequency")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # Test subplot
        axes[2].hist(
            test_df[col].dropna(), bins=50, alpha=0.7, edgecolor="black", color="red", label="Test"
        )
        axes[2].set_title(f'{col.replace("_", " ").title()} - Test')
        axes[2].set_xlabel(col.replace("_", " ").title())
        axes[2].set_ylabel("Frequency")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        # Add overall title
        if time_window:
            fig.suptitle(
                f'{col.replace("_", " ").title()} Distribution - {time_window} Data',
                fontsize=16,
                y=0.98,
            )
        else:
            fig.suptitle(f'{col.replace("_", " ").title()} Distribution', fontsize=16, y=0.98)

        plt.tight_layout()

        # Save individual plot
        plot_filename = f"{col} | distribution.png"
        plot_path = os.path.join(distributions_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        log.info(f"  ✅ Saved distribution plot: {plot_filename}")

    log.info(f"✅ Distribution plots saved to {distributions_dir}")


def run_visualize(config: dict[str, Any]):
    """
    Run the visualization pipeline for train/val/test datasets.
    Generates only basic statistics and distribution plots.

    Args:
        config (dict[str, Any]): Configuration dictionary
    """
    try:
        log.info("🎨 Starting visualization pipeline...")

        # Extract configuration
        input_files = config["input_files"]
        output_directory = config["output_directory"]
        time_window = config.get("time_window", None)

        # Create visualization directory
        viz_dir = create_visualization_directory(output_directory)

        # Initialize dataframes for each split
        train_df = None
        val_df = None
        test_df = None

        # Process each split (train/val/test) separately
        for input_file in input_files:
            log.info(f"\n📊 Loading data from: {os.path.basename(input_file)}")

            # Load data
            df = load_data(input_file)

            # Extract split name from filename
            filename = os.path.basename(input_file)
            # Look for train, val, or test in the filename
            if "_train.parquet" in filename:
                split_name = "train"
                train_df = df
            elif "_val.parquet" in filename:
                split_name = "val"
                val_df = df
            elif "_test.parquet" in filename:
                split_name = "test"
                test_df = df
            else:
                split_name = "unknown"

            log.info(f"✅ Loaded {split_name} dataset with {len(df):,} rows")

        # Check if all splits are available
        if train_df is None or val_df is None or test_df is None:
            error_msg = "Not all splits (train/val/test) found. Cannot proceed with visualization."
            log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
            raise ValueError(error_msg)

        # Generate basic statistics for all splits
        log.info("\n📊 Generating basic statistics...")
        stats = generate_basic_statistics(train_df, val_df, test_df, viz_dir, time_window)

        # Create distribution plots with train/val/test splits
        log.info("\n📊 Creating distribution plots...")
        create_distribution_plots(train_df, val_df, test_df, viz_dir, time_window)

        log.info("✅ Visualization pipeline completed successfully!")
        log.info(f"📁 Visualizations saved to: {viz_dir}")
        log.info(f"📄 Statistics file: {os.path.join(viz_dir, 'basic_statistics.txt')}")
        log.info(f"📊 Distribution plots: {os.path.join(viz_dir, 'distributions_plots')}")

    except Exception as e:
        error_msg = f"Visualization pipeline failed: {e}"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise e


if __name__ == "__main__":
    # Example usage
    config = {
        "input_files": [
            "data/5min_forex_data_train.parquet",
            "data/5min_forex_data_val.parquet",
            "data/5min_forex_data_test.parquet",
        ],
        "output_directory": "data",
        "time_window": "5min",
    }

    run_visualize(config)
