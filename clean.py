#!/usr/bin/env python3
"""
Data cleaning script for time series data processing.
This script analyzes gaps in time series data and provides cleaning functionality.
"""

import os
import sys
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime


def load_config(config_file="clean.yaml"):
    """
    Load configuration from YAML file.
    
    Args:
        config_file (str): Path to the YAML configuration file
        
    Returns:
        dict: Configuration dictionary
    """
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config_file}' not found")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}")


def load_data(input_file: str) -> pd.DataFrame:
    """
    Load data from parquet file.
    
    Args:
        input_file (str): Path to the input parquet file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        print(f"📁 Loading data from {input_file}")
        data = pd.read_parquet(input_file)
        print(f"✅ Loaded {len(data)} rows with {len(data.columns)} columns")
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_file}")
    except Exception as e:
        raise Exception(f"Error loading data: {e}")


def convert_timestamp_to_datetime(data: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """
    Convert timestamp column to datetime for easier analysis.
    
    Args:
        data (pd.DataFrame): Input data with time column
        time_column (str): Name of the time column
        
    Returns:
        pd.DataFrame: Data with converted datetime column
    """
    if time_column not in data.columns:
        raise ValueError(f"Time column '{time_column}' not found in data")
    
    # Create a copy to avoid modifying original data
    data_copy = data.copy()
    
    # Convert timestamp to datetime (assuming milliseconds)
    data_copy['datetime'] = pd.to_datetime(data_copy[time_column], unit='ms', utc=True)
    
    # Sort by datetime
    data_copy = data_copy.sort_values('datetime').reset_index(drop=True)
    
    print(f"✅ Converted {time_column} to datetime and sorted data")
    print(f"📅 Time range: {data_copy['datetime'].min()} to {data_copy['datetime'].max()}")
    
    return data_copy


def calculate_time_interval(data: pd.DataFrame) -> pd.Timedelta:
    """
    Calculate the time interval between consecutive rows.
    
    Args:
        data (pd.DataFrame): Data with datetime column
        
    Returns:
        pd.Timedelta: Time interval between rows
    """
    if len(data) < 2:
        raise ValueError("Need at least 2 rows to calculate time interval")
    
    # Calculate time differences between consecutive rows
    time_diffs = data['datetime'].diff().dropna()
    
    # Get the most common interval (mode)
    most_common_interval = time_diffs.mode().iloc[0]
    
    print(f"⏰ Detected time interval: {most_common_interval}")
    return most_common_interval


def analyze_gaps(data: pd.DataFrame, time_column: str) -> Dict:
    """
    Analyze gaps in the time series data for all features.
    
    Args:
        data (pd.DataFrame): Input data with datetime column
        time_column (str): Name of the time column
        
    Returns:
        Dict: Gap analysis results
    """
    print("\n🔍 Analyzing gaps in time series data...")
    
    # Get all numeric columns except time columns
    numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
    time_related_columns = [time_column, 'datetime']
    features_to_analyze = [col for col in numeric_columns if col not in time_related_columns]
    
    print(f"📊 Analyzing {len(features_to_analyze)} features: {features_to_analyze[:5]}{'...' if len(features_to_analyze) > 5 else ''}")
    
    gap_analysis = {}
    
    # Analyze gaps for each feature
    for feature in features_to_analyze:
        print(f"  Analyzing gaps for {feature}...")
        
        # Find gaps (NaN values) in the feature
        gap_values = data[feature].isna()
        gap_indices = gap_values[gap_values].index.tolist()
        
        # Calculate gap statistics
        total_gap_length = len(gap_indices)
        gap_percentage = (total_gap_length / len(data)) * 100
        
        # Find consecutive gaps
        gaps = []
        if gap_indices:
            current_gap = [gap_indices[0]]
            for i in range(1, len(gap_indices)):
                if gap_indices[i] == gap_indices[i-1] + 1:
                    current_gap.append(gap_indices[i])
                else:
                    gaps.append(current_gap)
                    current_gap = [gap_indices[i]]
            gaps.append(current_gap)
        
        # Calculate gap statistics
        gap_lengths = [len(gap) for gap in gaps]
        max_gap_length = max(gap_lengths) if gap_lengths else 0
        avg_gap_length = np.mean(gap_lengths) if gap_lengths else 0
        
        # Store results
        gap_analysis[feature] = {
            'total_gap_length': total_gap_length,
            'gap_percentage': gap_percentage,
            'gaps': len(gaps),
            'max_gap_length': max_gap_length,
            'avg_gap_length': avg_gap_length,
            'gap_indices': gap_indices,
            'gaps_detail': gaps
        }
    
    return gap_analysis


def print_gap_summary(gap_analysis: Dict, time_interval: pd.Timedelta = None, is_regular_interval: bool = True) -> None:
    """
    Print a comprehensive summary of gap analysis results.
    
    Args:
        gap_analysis (Dict): Gap analysis results
        time_interval (pd.Timedelta): Time interval between rows (None for irregular data)
        is_regular_interval (bool): Whether the data has regular time intervals
    """
    print("\n" + "="*80)
    print("📊 GAP ANALYSIS SUMMARY")
    print("="*80)
    
    if not gap_analysis:
        print("No gaps found in any features.")
        return
    
    # Create summary DataFrame
    summary_data = []
    for feature, stats in gap_analysis.items():
        if is_regular_interval and time_interval:
            # Convert gap lengths to days for regular intervals
            interval_days = time_interval.total_seconds() / (24 * 3600)
            max_gap_days = stats['max_gap_length'] * interval_days
            avg_gap_days = stats['avg_gap_length'] * interval_days
            
            summary_data.append({
                'Feature': feature,
                'Total Gap Length': stats['total_gap_length'],
                'Gap %': f"{stats['gap_percentage']:.2f}%",
                'Number of Gaps': stats['gaps'],
                'Max Gap (days)': f"{max_gap_days:.1f}",
                'Avg Gap (days)': f"{avg_gap_days:.1f}"
            })
        else:
            # Show gap lengths in number of rows for irregular intervals
            summary_data.append({
                'Feature': feature,
                'Total Gap Length': stats['total_gap_length'],
                'Gap %': f"{stats['gap_percentage']:.2f}%",
                'Number of Gaps': stats['gaps'],
                'Max Gap (# Rows)': stats['max_gap_length'],
                'Avg Gap (# Rows)': f"{stats['avg_gap_length']:.1f}"
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Print summary table
    print("\nFeature-wise Gap Summary:")
    print("-" * 80)
    print(summary_df.to_string(index=False))
    
    # Print overall statistics
    print("\n" + "-" * 80)
    print("OVERALL STATISTICS:")
    print("-" * 80)
    
    total_features = len(gap_analysis)
    features_with_gaps = sum(1 for stats in gap_analysis.values() if stats['total_gap_length'] > 0)
    
    print(f"Total features analyzed: {total_features}")
    print(f"Features with gaps: {features_with_gaps}")
    print(f"Features without gaps: {total_features - features_with_gaps}")
    
    if features_with_gaps > 0:
        # Find features with largest gap length
        max_gap_length_feature = max(gap_analysis.items(), key=lambda x: x[1]['total_gap_length'])
        print(f"\nFeature with largest gap length: {max_gap_length_feature[0]} ({max_gap_length_feature[1]['total_gap_length']} total gap length, {max_gap_length_feature[1]['gap_percentage']:.2f}%)")
        
        # Find features with longest gaps
        max_gap_feature = max(gap_analysis.items(), key=lambda x: x[1]['max_gap_length'])
        if is_regular_interval and time_interval:
            max_gap_days = max_gap_feature[1]['max_gap_length'] * interval_days
            print(f"Feature with longest gap: {max_gap_feature[0]} ({max_gap_days:.1f} days)")
        else:
            print(f"Feature with longest gap: {max_gap_feature[0]} ({max_gap_feature[1]['max_gap_length']} rows)")
        
        # Calculate average gap percentage across all features
        avg_gap_percentage = np.mean([stats['gap_percentage'] for stats in gap_analysis.values()])
        print(f"Average gap percentage across all features: {avg_gap_percentage:.2f}%")
    
    print("\n" + "="*80)


def save_gap_analysis_report(gap_analysis: Dict, output_directory: str, time_interval: pd.Timedelta, input_file: str, is_regular_interval: bool = True, report_suffix: str = "") -> None:
    """
    Save detailed gap analysis report to file.
    
    Args:
        gap_analysis (Dict): Gap analysis results
        output_directory (str): Output directory path
        time_interval (pd.Timedelta): Time interval between rows (None for irregular data)
        input_file (str): Input file path for generating report name
        is_regular_interval (bool): Whether the data has regular time intervals
        report_suffix (str): Suffix to add to report filename (e.g., "_before_cleaning", "_after_cleaning")
    """
    # Generate report filename based on input file name
    input_filename = os.path.splitext(os.path.basename(input_file))[0]
    report_filename = f"{input_filename}_report{report_suffix}.txt"
    output_path = os.path.join(output_directory, report_filename)
    
    try:
        with open(output_path, 'w') as f:
            f.write("GAP ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Time interval: {time_interval}\n")
            if report_suffix:
                f.write(f"Report type: {report_suffix.replace('_', ' ').title()}\n")
            f.write("\n")
            
            for feature, stats in gap_analysis.items():
                f.write(f"Feature: {feature}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Total gap length: {stats['total_gap_length']}\n")
                f.write(f"Gap percentage: {stats['gap_percentage']:.2f}%\n")
                f.write(f"Number of gaps: {stats['gaps']}\n")
                
                if is_regular_interval and time_interval:
                    # Convert to days for regular intervals
                    interval_days = time_interval.total_seconds() / (24 * 3600)
                    max_gap_days = stats['max_gap_length'] * interval_days
                    avg_gap_days = stats['avg_gap_length'] * interval_days
                    
                    f.write(f"Maximum gap length: {stats['max_gap_length']} intervals ({max_gap_days:.1f} days)\n")
                    f.write(f"Average gap length: {stats['avg_gap_length']:.2f} intervals ({avg_gap_days:.1f} days)\n")
                else:
                    # Show in rows for irregular intervals
                    f.write(f"Maximum gap length: {stats['max_gap_length']} rows\n")
                    f.write(f"Average gap length: {stats['avg_gap_length']:.2f} rows\n")
                
                if stats['gaps'] > 0:
                    f.write("Gaps:\n")
                    for i, gap in enumerate(stats['gaps_detail'][:10]):  # Show first 10 gaps
                        if is_regular_interval and time_interval:
                            gap_days = len(gap) * interval_days
                            f.write(f"  Gap {i+1}: {len(gap)} consecutive gaps ({gap_days:.1f} days) starting at index {gap[0]}\n")
                        else:
                            f.write(f"  Gap {i+1}: {len(gap)} consecutive gaps ({len(gap)} rows) starting at index {gap[0]}\n")
                    if len(stats['gaps_detail']) > 10:
                        f.write(f"  ... and {len(stats['gaps_detail']) - 10} more gaps\n")
                f.write("\n")
        
        print(f"✅ Gap analysis report saved to {output_path}")
    except Exception as e:
        print(f"❌ Error saving gap analysis report: {e}")


def clean_data(data: pd.DataFrame, time_column: str) -> pd.DataFrame:
    """
    Clean the data using linear interpolation, backward fill for start gaps, and forward fill for end gaps.
    
    Args:
        data (pd.DataFrame): Input data with datetime column
        time_column (str): Name of the time column
        
    Returns:
        pd.DataFrame: Cleaned data
    """
    print("\n🧹 Starting data cleaning process...")
    
    # Create a copy to avoid modifying original data
    cleaned_data = data.copy()
    
    # Get all numeric columns except time columns
    numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns.tolist()
    time_related_columns = [time_column, 'datetime']
    features_to_clean = [col for col in numeric_columns if col not in time_related_columns]
    
    print(f"📊 Cleaning {len(features_to_clean)} features: {features_to_clean[:5]}{'...' if len(features_to_clean) > 5 else ''}")
    
    total_cleaned_values = 0
    
    for feature in features_to_clean:
        print(f"  Cleaning {feature}...")
        
        # Get the feature column
        feature_data = cleaned_data[feature]
        
        # Count missing values before cleaning
        missing_before = feature_data.isna().sum()
        
        if missing_before == 0:
            print(f"    ✅ No missing values found in {feature}")
            continue
        
        # Apply cleaning strategy:
        # 1. Linear interpolation for gaps in the middle
        # 2. Backward fill for gaps at the start
        # 3. Forward fill for gaps at the end
        
        # First, apply linear interpolation
        feature_data_interpolated = feature_data.interpolate(method='linear')
        
        # Find remaining gaps (at the start or end)
        remaining_gaps = feature_data_interpolated.isna()
        
        if remaining_gaps.any():
            # Apply backward fill for gaps at the start
            feature_data_bfill = feature_data_interpolated.bfill()
            
            # Apply forward fill for any remaining gaps at the end
            feature_data_ffill = feature_data_bfill.ffill()
            
            # Update the cleaned data
            cleaned_data[feature] = feature_data_ffill
        else:
            # Update the cleaned data with interpolated values
            cleaned_data[feature] = feature_data_interpolated
        
        # Count missing values after cleaning
        missing_after = cleaned_data[feature].isna().sum()
        cleaned_count = missing_before - missing_after
        
        print(f"    ✅ Cleaned {cleaned_count} missing values in {feature}")
        total_cleaned_values += cleaned_count
    
    print(f"\n🎉 Data cleaning completed! Total values cleaned: {total_cleaned_values}")
    
    return cleaned_data


def save_cleaned_data(data: pd.DataFrame, output_file: str) -> None:
    """
    Save cleaned data to parquet file.
    
    Args:
        data (pd.DataFrame): Cleaned data to save
        output_file (str): Output file path
    """
    try:
        # Remove the datetime column if it exists (it was only used for analysis)
        if 'datetime' in data.columns:
            data_to_save = data.drop(columns=['datetime'])
        else:
            data_to_save = data
        
        # Save to parquet
        data_to_save.to_parquet(output_file, index=False)
        print(f"✅ Cleaned data saved to {output_file}")
    except Exception as e:
        print(f"❌ Error saving cleaned data: {e}")


def process_cleaning(input_file: str, output_directory: str, time_column: str, pipeline_stage: str) -> Tuple[bool, str]:
    """
    Process the cleaning analysis and cleaning for a single file.
    
    Args:
        input_file (str): Path to the input parquet file
        output_directory (str): Output directory for reports and cleaned data
        time_column (str): Name of the time column
        
    Returns:
        Tuple[bool, str]: (True if successful, output file path) or (False, None)
    """
    try:
        print(f"\n🔄 Processing file: {input_file}")
        
        # Load data
        data = load_data(input_file)
        
        # Convert timestamp to datetime
        data = convert_timestamp_to_datetime(data, time_column)
        
        # Determine if data has regular intervals based on pipeline stage
        # Tick data has irregular intervals, aggregated data has regular intervals
        is_regular_interval = (pipeline_stage != 'ticks')
        
        # Calculate time interval only for regular intervals
        time_interval = None
        if is_regular_interval:
            time_interval = calculate_time_interval(data)
        
        # STEP 1: Generate report before cleaning
        print("\n📊 STEP 1: Generating report before cleaning...")
        gap_analysis_before = analyze_gaps(data, time_column)
        print_gap_summary(gap_analysis_before, time_interval, is_regular_interval)
        
        # Create output directory if it doesn't exist
        os.makedirs(output_directory, exist_ok=True)
        
        # Save detailed report before cleaning
        save_gap_analysis_report(gap_analysis_before, output_directory, time_interval, input_file, is_regular_interval, "_before_cleaning")
        
        # STEP 2: Clean the data
        print("\n🧹 STEP 2: Cleaning the data...")
        cleaned_data = clean_data(data, time_column)
        
        # STEP 3: Generate report after cleaning
        print("\n📊 STEP 3: Generating report after cleaning...")
        gap_analysis_after = analyze_gaps(cleaned_data, time_column)
        print_gap_summary(gap_analysis_after, time_interval, is_regular_interval)
        
        # Save detailed report after cleaning
        save_gap_analysis_report(gap_analysis_after, output_directory, time_interval, input_file, is_regular_interval, "_after_cleaning")
        
        # STEP 4: Save cleaned data
        print("\n💾 STEP 4: Saving cleaned data...")
        input_filename = os.path.splitext(os.path.basename(input_file))[0]
        output_filename = f"{input_filename}_cleaned.parquet"
        output_file = os.path.join(output_directory, output_filename)
        save_cleaned_data(cleaned_data, output_file)
        
        print(f"✅ Successfully completed cleaning process for {input_file}")
        return True, output_file
        
    except Exception as e:
        print(f"❌ Error processing {input_file}: {e}")
        return False, None


def run_clean(config):
    """
    Run the cleaning process with the specified configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing pipeline stage and stage-specific settings
    """
    try:
        print("🧹 Data Cleaning Tool")
        print("=" * 50)
        
        # Extract configuration values
        pipeline_stage = config['pipeline_stage']
        stage_config = config['stages'][pipeline_stage]
        
        input_files = stage_config['input_files']
        output_directory = stage_config['output_directory']
        time_column = stage_config['time_column']
        
        print(f"🔧 Pipeline stage: {pipeline_stage}")
        print(f"📁 Input files: {len(input_files)} files to process")
        print(f"📁 Output directory: {output_directory}")
        print(f"⏰ Time column: {time_column}")
        
        # Process each input file independently
        successful_files = 0
        total_files = len(input_files)
        
        for input_file in input_files:
            print(f"\n{'='*60}")
            success, output_file = process_cleaning(input_file, output_directory, time_column, pipeline_stage)
            if success:
                successful_files += 1
        
        # Display final summary
        print(f"\n🎯 Cleaning Process Summary:")
        print("=" * 50)
        print(f"📊 Total files: {total_files}")
        print(f"✅ Successful: {successful_files}")
        print(f"❌ Failed: {total_files - successful_files}")
        print(f"📁 Output directory: {output_directory}")
        
        if successful_files == total_files:
            print("🎉 All files processed successfully!")
        elif successful_files > 0:
            print(f"⚠️  {total_files - successful_files} files failed to process")
        else:
            print("❌ All files failed to process")
        print("=" * 50)
        
    except Exception as e:
        print(f"❌ Fatal Error: {e}")
        raise 