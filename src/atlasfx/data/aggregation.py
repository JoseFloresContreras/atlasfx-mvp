import os
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Any
import importlib.util
from atlasfx.utils.logging import log

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
        error_msg = f"Configuration file '{config_file}' not found"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise FileNotFoundError(error_msg)
    except yaml.YAMLError as e:
        error_msg = f"Error parsing YAML file: {e}"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise yaml.YAMLError(error_msg)

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
    from atlasfx.data import aggregators as aggregators_module
    
    # Load requested aggregators
    for name in aggregator_names:
        if hasattr(aggregators_module, name):
            aggregators[name] = getattr(aggregators_module, name)
            log.info(f"✅ Loaded aggregator: {name}")
        else:
            log.warning(f"⚠️  Warning: Aggregator '{name}' not found in aggregators module")
    
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
        log.warning("⚠️  No data to aggregate")
        return pd.DataFrame()
    
    # Parse time window for resampling
    resample_freq = time_window
    log.info(f"📊 Aggregating data into {time_window} windows using resampling...")
    
    # Convert Unix timestamps to datetime for resampling
    log.info("🕐 Converting Unix timestamps to datetime for resampling...")
    data = data.copy()
    data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms', utc=True)
    
    # Set datetime as index for resampling
    data.set_index('datetime', inplace=True)
    
    # Sort by timestamp
    data = data.sort_index()
    
    log.info(f"📅 Data time range: {data.index.min()} to {data.index.max()}")
    log.info(f"📊 Total ticks: {len(data)}")
    
    # Use pandas resampling to create time windows
    log.info(f"⏰ Creating {resample_freq} time windows...")
    
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
        aggregated_df[other_cols] = aggregated_df[other_cols].astype(np.float32)
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
        log.info(f"\n📁 Processing file: {input_file}")
        
        # Check if file exists
        if not os.path.exists(input_file):
            log.error(f"❌ File not found: {input_file}")
            raise Exception(f"File not found: {input_file}")
        
        # Load the data
        data = pd.read_parquet(input_file)
        log.info(f"📊 Loaded {len(data)} rows with columns: {list(data.columns)}")
        log.info(f"Data types: {data.dtypes}")
        log.info(f"Data memory usage: {data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        # Aggregate the data
        aggregated_data = aggregate_data(data, time_window, aggregators)

        log.info(f"Aggregated data shape: {aggregated_data.shape}")
        log.info(f"Aggregated data types: {aggregated_data.dtypes}")
        log.info(f"Aggregated data memory usage: {aggregated_data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        if aggregated_data.empty:
            log.warning(f"⚠️  No data was aggregated for {input_file}")
            return False, None, None
        
        # Extract symbol from input filename (everything before first underscore)
        filename = os.path.basename(input_file)  # Get just the filename without path
        symbol = filename.split('_')[0]  # Extract symbol from filename
        
        log.info(f"✅ Successfully aggregated {len(aggregated_data)} windows for {symbol}")
        return True, symbol, aggregated_data
        
    except Exception as e:
        log.error(f"❌ Error processing {input_file}: {e}")
        raise e

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
        log.error("❌ No dataframes to combine")
        return pd.DataFrame()
    
    log.info(f"\n🔄 Combining {len(dataframes_dict)} dataframes...")
    
    dfs = []
    for symbol, df in dataframes_dict.items():
        df = df.copy()
        df = df.set_index("start_time")
        df = df.add_prefix(f"{symbol} | ")
        dfs.append(df)

    combined_df = pd.concat(dfs, axis=1, join="outer").reset_index()
    
    log.info(f"Combined dataframe memory usage: {combined_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    log.info(f"✅ Combined dataframe shape: {combined_df.shape}")
    log.info(f"📊 Columns: {list(combined_df.columns)}")

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
        
        log.info(f"⏰ Time window: {time_window}")
        log.info(f"🔧 Aggregators: {', '.join(aggregator_names)}")
        log.info(f"📁 Input files: {len(input_files)} files to process")
        log.info(f"📁 Output directory: {output_directory}")
        
        # Load aggregator functions
        aggregators = load_aggregators(aggregator_names)
        
        if not aggregators:
            error_msg = "No valid aggregators loaded. Exiting."
            log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
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
                log.info(f"💾 Saving combined data to: {output_path}")
                combined_df.to_parquet(output_path, engine='pyarrow')
                
                log.info(f"✅ Successfully saved combined data with {len(combined_df)} rows to {output_path}")
            else:
                log.error("❌ Failed to combine dataframes")
        else:
            log.error("❌ No dataframes to combine")
        
        # Display final summary
        log.info(f"\n📈 Aggregation Summary:")
        log.info(f"   - Time window: {time_window}")
        log.info(f"   - Aggregators applied: {', '.join(aggregator_names)}")
        log.info(f"   - Files processed: {successful_files}/{total_files}")
        log.info(f"   - Symbols combined: {list(dataframes_dict.keys())}")
        
        if successful_files == total_files:
            log.info("🎉 All files processed successfully!")
        else:
            log.warning(f"⚠️  {total_files - successful_files} files failed to process")
        
    except Exception as e:
        error_msg = f"Error: {e}"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise e