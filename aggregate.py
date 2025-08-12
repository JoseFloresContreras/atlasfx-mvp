import os
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Any
import importlib.util

def load_config(config_file="aggregate.yaml"):
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

def load_aggregators(aggregator_names: List[str]) -> Dict[str, Callable]:
    """
    Dynamically load aggregator functions from aggregators.py.
    
    Args:
        aggregator_names (List[str]): List of aggregator function names to load
        
    Returns:
        Dict[str, Callable]: Dictionary mapping aggregator names to functions
    """
    aggregators = {}
    
    # Import aggregators module
    spec = importlib.util.spec_from_file_location("aggregators", "aggregators.py")
    aggregators_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(aggregators_module)
    
    # Load requested aggregators
    for name in aggregator_names:
        if hasattr(aggregators_module, name):
            aggregators[name] = getattr(aggregators_module, name)
            print(f"✅ Loaded aggregator: {name}")
        else:
            print(f"⚠️  Warning: Aggregator '{name}' not found in aggregators.py")
    
    return aggregators

def aggregate_data(data: pd.DataFrame, time_window: str, aggregators: Dict[str, Callable]) -> pd.DataFrame:
    """
    Aggregate tick data into time-based windows using pandas resampling and specified aggregators.
    
    Args:
        data (pd.DataFrame): Input tick data with Unix timestamp column
        time_window (str): Time window size (e.g., "5m")
        aggregators (Dict[str, Callable]: Dictionary of aggregator functions
        
    Returns:
        pd.DataFrame: Aggregated data with one row per time window
    """
    if data.empty:
        print("⚠️  No data to aggregate")
        return pd.DataFrame()
    
    # Parse time window for resampling
    resample_freq = time_window
    print(f"📊 Aggregating data into {time_window} windows using resampling...")
    
    # Convert Unix timestamps to datetime for resampling
    print("🕐 Converting Unix timestamps to datetime for resampling...")
    data = data.copy()
    data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
    
    # Set datetime as index for resampling
    data.set_index('datetime', inplace=True)
    
    # Sort by timestamp
    data = data.sort_index()
    
    print(f"📅 Data time range: {data.index.min()} to {data.index.max()}")
    print(f"📊 Total ticks: {len(data)}")
    
    # Use pandas resampling to create time windows
    print(f"⏰ Creating {resample_freq} time windows...")
    
    # Resample the data
    resampled = data.resample(resample_freq)
    
    # Create a function to apply all aggregators to a group
    def apply_aggregators(group):
        """Apply all aggregator functions to a group and return results as a dict."""
        # Get the window start time from the group's name (which is the window start)
        window_start = group.name
        window_duration = pd.Timedelta(resample_freq)
        
        result = {
            'start_time': int(window_start.timestamp() * 1000)
        }
        
        # Apply each aggregator function
        for agg_name, agg_func in aggregators.items():
            # Get aggregator results (now returns a dictionary)
            agg_results = agg_func(window_start, window_duration, group)
            
            # Handle multi-output results
            for output_name, value in agg_results.items():
                result[output_name] = value

        return result
    
    # Apply the aggregators to each group
    results = resampled.apply(apply_aggregators)
    
    # Convert results to DataFrame
    aggregated_df = pd.DataFrame(results.tolist())
    
    if not aggregated_df.empty:
        # Reorder columns for better readability
        time_cols = ['start_time']
        other_cols = [col for col in aggregated_df.columns if col not in time_cols]
        aggregated_df = aggregated_df[time_cols + other_cols]
    
    return aggregated_df

def process_single_file(input_file: str, time_window: str, aggregators: Dict[str, Callable]) -> tuple:
    """
    Process a single input file and return the aggregated result.
    
    Args:
        input_file (str): Path to the input parquet file
        time_window (str): Time window size
        aggregators (Dict[str, Callable]): Dictionary of aggregator functions
        
    Returns:
        tuple: (success: bool, symbol: str, aggregated_data: pd.DataFrame or None)
    """
    try:
        print(f"\n📁 Processing file: {input_file}")
        
        # Check if file exists
        if not os.path.exists(input_file):
            print(f"❌ File not found: {input_file}")
            return False, None, None
        
        # Load the data
        data = pd.read_parquet(input_file)
        print(f"📊 Loaded {len(data)} rows with columns: {list(data.columns)}")
        
        # Aggregate the data
        aggregated_data = aggregate_data(data, time_window, aggregators)
        
        if aggregated_data.empty:
            print(f"⚠️  No data was aggregated for {input_file}")
            return False, None, None
        
        # Extract symbol from input filename (everything before first underscore)
        filename = os.path.basename(input_file)  # Get just the filename without path
        symbol = filename.split('_')[0]  # Extract symbol from filename
        
        print(f"✅ Successfully aggregated {len(aggregated_data)} windows for {symbol}")
        return True, symbol, aggregated_data
        
    except Exception as e:
        print(f"❌ Error processing {input_file}: {e}")
        return False, None, None

def combine_aggregated_dataframes(dataframes_dict: Dict[str, pd.DataFrame], time_window: str) -> pd.DataFrame:
    """
    Combine multiple aggregated dataframes into a single dataframe.
    
    Args:
        dataframes_dict (Dict[str, pd.DataFrame]): Dictionary mapping symbols to their aggregated dataframes
        time_window (str): Time window used for aggregation
        
    Returns:
        pd.DataFrame: Combined dataframe with symbol prefixes in column names
    """
    if not dataframes_dict:
        print("❌ No dataframes to combine")
        return pd.DataFrame()
    
    print(f"\n🔄 Combining {len(dataframes_dict)} dataframes...")
    
    # Start with the first dataframe as base
    symbols = list(dataframes_dict.keys())
    base_symbol = symbols[0]
    combined_df = dataframes_dict[base_symbol].copy()
    
    # Set window_start_unix as index for merging
    combined_df.set_index('start_time', inplace=True)
    
    # Add symbol prefix to base dataframe columns (except time-related columns)
    for col in combined_df.columns:
        combined_df[f'{base_symbol}_{col}'] = combined_df[col]
        combined_df.drop(columns=[col], inplace=True)
    
    # Merge other dataframes
    for symbol in symbols[1:]:
        df = dataframes_dict[symbol].copy()
        df.set_index('start_time', inplace=True)
        
        # Add symbol prefix to columns (except time-related columns)
        for col in df.columns:
            df[f'{symbol}_{col}'] = df[col]
            df.drop(columns=[col], inplace=True)
                
        combined_df = combined_df.merge(df, left_index=True, right_index=True, how='outer')
    
    # Reset index to make window_start_unix a column again
    combined_df.reset_index(inplace=True)
    
    print(f"✅ Combined dataframe shape: {combined_df.shape}")
    print(f"📊 Columns: {list(combined_df.columns)}")
    
    return combined_df



def run_aggregate(config):
    """
    Run the aggregation process with the specified configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing time window, aggregators, input files, and output settings
    """
    try:
        # Extract configuration values
        time_window = config['time_window']
        aggregator_names = config['aggregators']
        input_files = config['input_files']
        output_directory = config['output_directory']
        # Get custom output filename if specified
        custom_filename = config.get('output_filename', None)
        
        print(f"⏰ Time window: {time_window}")
        print(f"🔧 Aggregators: {', '.join(aggregator_names)}")
        print(f"📁 Input files: {len(input_files)} files to process")
        print(f"📁 Output directory: {output_directory}")
        
        # Load aggregator functions
        aggregators = load_aggregators(aggregator_names)
        
        if not aggregators:
            print("❌ No valid aggregators loaded. Exiting.")
            return
        
        # Process each input file and collect dataframes
        successful_files = 0
        total_files = len(input_files)
        dataframes_dict = {}
        
        for input_file in input_files:
            success, symbol, aggregated_data = process_single_file(input_file, time_window, aggregators)
            if success and symbol and aggregated_data is not None:
                dataframes_dict[symbol] = aggregated_data
                successful_files += 1
        
        # Combine all dataframes if we have any
        if dataframes_dict:
            combined_df = combine_aggregated_dataframes(dataframes_dict, time_window)
            
            if not combined_df.empty:
                # Create output directory if it doesn't exist
                os.makedirs(output_directory, exist_ok=True)
                
                # Generate output filename
                output_filename = f"{time_window}_{custom_filename}.parquet"
                output_path = os.path.join(output_directory, output_filename)
                
                # Save combined data
                print(f"💾 Saving combined data to: {output_path}")
                combined_df.to_parquet(output_path, engine='pyarrow')
                
                print(f"✅ Successfully saved combined data with {len(combined_df)} rows to {output_path}")
            else:
                print("❌ Failed to combine dataframes")
        else:
            print("❌ No dataframes to combine")
        
        # Display final summary
        print(f"\n📈 Aggregation Summary:")
        print(f"   - Time window: {time_window}")
        print(f"   - Aggregators applied: {', '.join(aggregator_names)}")
        print(f"   - Files processed: {successful_files}/{total_files}")
        print(f"   - Symbols combined: {list(dataframes_dict.keys())}")
        
        if successful_files == total_files:
            print("🎉 All files processed successfully!")
        else:
            print(f"⚠️  {total_files - successful_files} files failed to process")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise 