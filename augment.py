import os
import pandas as pd
import numpy as np
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Any
import importlib.util
import traceback

def load_config(config_file="augment.yaml"):
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

def load_augmentations(augmentation_names: List[str]) -> Dict[str, Callable]:
    """
    Dynamically load augmentation functions from augmentations.py.
    
    Args:
        augmentation_names (List[str]): List of augmentation function names to load
        
    Returns:
        Dict[str, Callable]: Dictionary mapping augmentation names to functions
    """
    augmentations = {}
    
    # Import augmentations module
    spec = importlib.util.spec_from_file_location("augmentations", "augmentations.py")
    augmentations_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(augmentations_module)
    
    # Load requested augmentations
    for name in augmentation_names:
        if hasattr(augmentations_module, name):
            augmentations[name] = getattr(augmentations_module, name)
            print(f"✅ Loaded augmentation: {name}")
        else:
            print(f"⚠️  Warning: Augmentation '{name}' not found in augmentations.py")
    
    return augmentations

def augment_data(data: pd.DataFrame, augmentations_config: Dict[int, List[str]], augmentations: Dict[str, Callable]) -> pd.DataFrame:
    """
    Apply augmentations to the input dataframe with multiple window sizes.
    
    Args:
        data (pd.DataFrame): Input dataframe with time index
        augmentations_config (Dict[int, List[str]]): Dictionary mapping window sizes to list of augmentation names
        augmentations (Dict[str, Callable]): Dictionary of augmentation functions
        
    Returns:
        pd.DataFrame: Augmented data with original columns plus new features
    """
    if data.empty:
        print("⚠️  No data to augment")
        return pd.DataFrame()
    
    print(f"📊 Augmenting data with multiple window sizes...")
    print(f"📊 Original data shape: {data.shape}")
    print(f"📊 Original columns: {list(data.columns)}")
    
    # Make a copy of the original data
    augmented_df = data.copy()
    
    # Collect all augmentation results first
    all_augmentation_results = []
    
    # Apply augmentations for each window size
    for window_size, aug_names in augmentations_config.items():
        print(f"🔧 Processing window size: {window_size}")
        
        for aug_name in aug_names:
            if aug_name not in augmentations:
                print(f"⚠️  Warning: Augmentation '{aug_name}' not found, skipping...")
                continue
                
            print(f"  🔧 Applying augmentation: {aug_name} (window={window_size})")
            
            try:
                # Apply the augmentation function to the original data (not the augmented data)
                aug_func = augmentations[aug_name]
                aug_result = aug_func(data, window_size)
                
                if not aug_result.empty:
                    # Add window size suffix to column names to distinguish between different window sizes
                    aug_result_renamed = aug_result.copy()
                    new_columns = {}
                    for col in aug_result.columns:
                        new_columns[col] = f"{col}_w{window_size}"
                    aug_result_renamed = aug_result_renamed.rename(columns=new_columns)
                    
                    # Collect the augmentation result for later merging
                    all_augmentation_results.append(aug_result_renamed)
                    print(f"  ✅ Added {len(aug_result.columns)} columns from {aug_name} (window={window_size})")
                else:
                    print(f"  ⚠️  No results from augmentation: {aug_name} (window={window_size})")
                    
            except Exception as e:
                print(f"  ❌ Error applying augmentation {aug_name} (window={window_size}): {e}")
                continue
    
    # Merge all augmentation results with the original data at once
    if all_augmentation_results:
        print(f"🔄 Merging {len(all_augmentation_results)} augmentation results...")
        
        # Merge all augmentation results together first
        merged_augmentations = all_augmentation_results[0]
        for aug_result in all_augmentation_results[1:]:
            merged_augmentations = merged_augmentations.merge(
                aug_result, 
                left_index=True, 
                right_index=True, 
                how='outer'
            )
        
        # Now merge with the original data
        augmented_df = augmented_df.merge(
            merged_augmentations, 
            left_index=True, 
            right_index=True, 
            how='outer'
        )
        
        print(f"✅ Successfully merged all augmentation results")
    
    print(f"📊 Final augmented data shape: {augmented_df.shape}")
    print(f"📊 Final columns: {list(augmented_df.columns)}")
    
    return augmented_df

def process_single_file(input_file: str, augmentations_config: Dict[int, List[str]], augmentations: Dict[str, Callable]) -> tuple:
    """
    Process a single input file and return the augmented result.
    
    Args:
        input_file (str): Path to the input parquet file
        augmentations_config (Dict[int, List[str]]): Dictionary mapping window sizes to list of augmentation names
        augmentations (Dict[str, Callable]): Dictionary of augmentation functions
        
    Returns:
        tuple: (success: bool, augmented_data: pd.DataFrame or None)
    """
    try:
        print(f"\n📁 Processing file: {input_file}")
        
        # Check if file exists
        if not os.path.exists(input_file):
            print(f"❌ File not found: {input_file}")
            return False, None
        
        # Load the data
        data = pd.read_parquet(input_file)
        print(f"📊 Loaded {len(data)} rows with columns: {list(data.columns)}")
        
        # Check if data has a time column to set as index
        time_column = 'start_time'
        
        # Convert to datetime if it's Unix timestamp
        data[time_column] = pd.to_datetime(data[time_column], unit='ms', utc=True)
            
        # Set as index
        data.set_index(time_column, inplace=True)
        print(f"🕐 Set {time_column} as time index")
        
        
        # Sort by index
        data = data.sort_index()
        
        # Augment the data
        augmented_data = augment_data(data, augmentations_config, augmentations)
        
        if augmented_data.empty:
            print(f"⚠️  No data was augmented for {input_file}")
            return False, None
        
        # Reset index to make start_time a regular column (same as aggregate)
        augmented_data = augmented_data.reset_index()
        augmented_data[time_column] = (augmented_data[time_column].view('int64') // 10**6)

        print(f"✅ Successfully augmented data to {len(augmented_data)} rows")
        return True, augmented_data
        
    except Exception as e:
        print(f"❌ Error processing {input_file}: {e}")
        raise
    
def run_augment(config):
    """
    Run the augmentation process with the specified configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary containing augmentations config, input files, and output settings
    """
    try:
        # Extract configuration values
        augmentations_config = config['augmentations']
        input_files = config['input_files']
        output_directory = config['output_directory']
        # Get custom output filename if specified
        custom_filename = config.get('output_filename', None)
        
        # Collect all unique augmentation names from all window sizes
        all_augmentation_names = []
        for window_size, aug_names in augmentations_config.items():
            all_augmentation_names.extend(aug_names)
        all_augmentation_names = list(set(all_augmentation_names))  # Remove duplicates
        
        print(f"⏰ Window sizes: {list(augmentations_config.keys())}")
        print(f"🔧 Augmentations: {', '.join(all_augmentation_names)}")
        print(f"📁 Input files: {len(input_files)} files to process")
        print(f"📁 Output directory: {output_directory}")
        
        # Load augmentation functions
        augmentations = load_augmentations(all_augmentation_names)
        
        if not augmentations:
            print("❌ No valid augmentations loaded. Exiting.")
            return
        
        # Process each input file
        successful_files = 0
        total_files = len(input_files)
        
        for input_file in input_files:
            success, augmented_data = process_single_file(input_file, augmentations_config, augmentations)
            if success and augmented_data is not None:
                # Create output directory if it doesn't exist
                os.makedirs(output_directory, exist_ok=True)
                
                # Generate output filename
                input_filename = os.path.basename(input_file)
                input_name = os.path.splitext(input_filename)[0]  # Remove .parquet extension
                
                if custom_filename:
                    output_filename = f"{input_name}_{custom_filename}_augmented.parquet"
                else:
                    output_filename = f"{input_name}_augmented.parquet"
                
                output_path = os.path.join(output_directory, output_filename)
                
                # Save augmented data
                print(f"💾 Saving augmented data to: {output_path}")
                augmented_data.to_parquet(output_path, engine='pyarrow')
                
                print(f"✅ Successfully saved augmented data with {len(augmented_data)} rows to {output_path}")
                successful_files += 1
            else:
                print(f"❌ Failed to process {input_file}")
        
        # Display final summary
        print(f"\n📈 Augmentation Summary:")
        print(f"   - Window sizes: {list(augmentations_config.keys())}")
        print(f"   - Augmentations applied: {', '.join(all_augmentation_names)}")
        print(f"   - Files processed: {successful_files}/{total_files}")
        
        if successful_files == total_files:
            print("🎉 All files processed successfully!")
        else:
            print(f"⚠️  {total_files - successful_files} files failed to process")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise 