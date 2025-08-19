import os
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Any
import importlib.util
import traceback
from logger import log

def load_config(config_file="featurize.yaml"):
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

def load_featurizers(featurizer_names: List[str]) -> Dict[str, Callable]:
    """
    Dynamically load featurizer functions from featurizers.py.
    
    Args:
        featurizer_names (List[str]): List of featurizer function names to load
        
    Returns:
        Dict[str, Callable]: Dictionary mapping featurizer names to functions
    """
    featurizers = {}
    
    # Import featurizers module
    spec = importlib.util.spec_from_file_location("featurizers", "featurizers.py")
    featurizers_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(featurizers_module)
    
    # Load requested featurizers
    for name in featurizer_names:
        if hasattr(featurizers_module, name):
            featurizers[name] = getattr(featurizers_module, name)
            log.info(f"✅ Loaded featurizer: {name}")
        else:
            log.warning(f"⚠️  Warning: Featurizer '{name}' not found in featurizers.py")
    
    return featurizers

def featurize_data(data: pd.DataFrame, featurizers_config: Dict[str, Dict[str, Any]], featurizers: Dict[str, Callable]) -> pd.DataFrame:
    """
    Apply featurizers to the input dataframe with their specific configurations.
    
    Args:
        data (pd.DataFrame): Input dataframe with time index
        featurizers_config (Dict[str, Dict[str, Any]]): Dictionary mapping featurizer names to their configurations
        featurizers (Dict[str, Callable]): Dictionary of featurizer functions
        
    Returns:
        pd.DataFrame: Featurized data with original columns plus new features
    """
    if data.empty:
        log.warning("⚠️  No data to featurize")
        return pd.DataFrame()
    
    log.info(f"📊 Featurizing data with specific configurations...")
    log.info(f"📊 Original data shape: {data.shape}")
    log.info(f"📊 Original columns: {list(data.columns)}")
    
    # Make a copy of the original data
    featurized_df = data.copy()
    
    # Collect all featurizer results first
    all_featurizer_results = []
    
    # Apply featurizers with their specific configurations
    for feat_name, config in featurizers_config.items():
        if feat_name not in featurizers:
            log.warning(f"⚠️  Warning: Featurizer '{feat_name}' not found, skipping...")
            continue
            
        log.info(f"🔧 Applying featurizer: {feat_name} with config: {config}")
        
        try:
            # Apply the featurizer function to the original data (not the featurized data)
            feat_func = featurizers[feat_name]
            feat_result = feat_func(data, config)
            
            if not feat_result.empty:
                # Collect the featurizer result for later merging
                all_featurizer_results.append(feat_result)
                log.info(f"  ✅ Added {len(feat_result.columns)} columns from {feat_name}")
            else:
                log.warning(f"  ⚠️  No results from featurizer: {feat_name}")
                
        except Exception as e:
            log.error(f"  ❌ Error applying featurizer {feat_name}: {e}")
            continue
    
    # Merge all featurizer results with the original data at once
    if all_featurizer_results:
        log.info(f"🔄 Merging {len(all_featurizer_results)} featurizer results...")
        
        # Merge all featurizer results together first
        merged_featurizers = all_featurizer_results[0]
        for feat_result in all_featurizer_results[1:]:
            merged_featurizers = merged_featurizers.merge(
                feat_result, 
                left_index=True, 
                right_index=True, 
                how='outer'
            )
        
        # Now merge with the original data
        featurized_df = featurized_df.merge(
            merged_featurizers, 
            left_index=True, 
            right_index=True, 
            how='outer'
        )
        featurized_df = featurized_df.astype(np.float32)
        
        log.info(f"✅ Successfully merged all featurizer results")
    
    log.info(f"📊 Final featurized data shape: {featurized_df.shape}")
    log.info(f"📊 Final columns: {list(featurized_df.columns)}")
    
    return featurized_df

def process_single_file(input_file: str, featurizers_config: Dict[str, Dict[str, Any]], featurizers: Dict[str, Callable]) -> tuple:
    """
    Process a single input file and return the featurized result.
    
    Args:
        input_file (str): Path to the input parquet file
        featurizers_config (Dict[str, Dict[str, Any]]): Dictionary mapping featurizer names to their configurations
        featurizers (Dict[str, Callable]): Dictionary of featurizer functions
        
    Returns:
        tuple: (success: bool, featurized_data: pd.DataFrame or None)
    """
    try:
        log.info(f"\n📁 Processing file: {input_file}")
        
        # Check if file exists
        if not os.path.exists(input_file):
            log.error(f"❌ File not found: {input_file}")
            return False, None
        
        # Load the data
        data = pd.read_parquet(input_file)
        log.info(f"📊 Loaded {len(data)} rows with columns: {list(data.columns)}")
        
        # Check if data has a time column to set as index
        time_column = 'start_time'
        
        # Convert to datetime if it's Unix timestamp
        data[time_column] = pd.to_datetime(data[time_column], unit='ms', utc=True)
            
        # Set as index
        data.set_index(time_column, inplace=True)
        log.info(f"🕐 Set {time_column} as time index")
        
        
        # Sort by index
        data = data.sort_index()
        
        # Featurize the data
        featurized_data = featurize_data(data, featurizers_config, featurizers)
        
        if featurized_data.empty:
            log.warning(f"⚠️  No data was featurized for {input_file}")
            return False, None
        
        # Reset index to make start_time a regular column (same as aggregate)
        featurized_data = featurized_data.reset_index()
        featurized_data[time_column] = (featurized_data[time_column].view('int64') // 10**6)

        log.info(f"✅ Successfully featurized data to {len(featurized_data)} rows")
        return True, featurized_data
        
    except Exception as e:
        log.error(f"❌ Error processing {input_file}: {e}")
        raise
    
def run_featurize(config):
    """
    Run the featurization process with the specified configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing featurizers config, input files, and output settings
    """
    try:
        # Extract configuration values
        featurizers_config = config['featurizers']
        input_files = config['input_files']
        output_directory = config['output_directory']
        # Get custom output filename if specified
        custom_filename = config.get('output_filename', None)
        
        # Collect all unique featurizer names
        all_featurizer_names = list(featurizers_config.keys())
        
        log.info(f"🔧 Featurizers: {', '.join(all_featurizer_names)}")
        log.info(f"📁 Input files: {len(input_files)} files to process")
        log.info(f"📁 Output directory: {output_directory}")
        
        # Load featurizer functions
        featurizers = load_featurizers(all_featurizer_names)
        
        if not featurizers:
            error_msg = "No valid featurizers loaded. Exiting."
            log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
            return
        
        # Process each input file
        successful_files = 0
        total_files = len(input_files)
        
        for input_file in input_files:
            success, featurized_data = process_single_file(input_file, featurizers_config, featurizers)
            if success and featurized_data is not None:
                # Save featurized data (replace original file)
                log.info(f"💾 Saving featurized data to: {input_file}")
                featurized_data.to_parquet(input_file, engine='pyarrow')
                
                log.info(f"✅ Successfully updated data with featurization: {input_file}")
                successful_files += 1
            else:
                log.error(f"❌ Failed to process {input_file}")
        
        # Display final summary
        log.info(f"\n📈 Featurization Summary:")
        log.info(f"   - Featurizers applied: {', '.join(all_featurizer_names)}")
        log.info(f"   - Files processed: {successful_files}/{total_files}")
        
        if successful_files == total_files:
            log.info("🎉 All files processed successfully!")
        else:
            log.warning(f"⚠️  {total_files - successful_files} files failed to process")
        
    except Exception as e:
        error_msg = f"Error: {e}"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise 