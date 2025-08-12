#!/usr/bin/env python3
"""
Dynamic Data Processing Pipeline
This script orchestrates data processing steps based on configuration.
Supports flexible step execution order and dependency management.
"""

import os
import yaml
from typing import Dict, List, Any

# Import the run functions from each module
from merge import run_merge
from clean import run_clean
from aggregate import run_aggregate
from augment import run_augment
from visualize import run_visualize


def load_pipeline_config(config_file="pipeline.yaml"):
    """
    Load the main pipeline configuration.
    
    Args:
        config_file (str): Path to the pipeline configuration file
        
    Returns:
        dict: Pipeline configuration dictionary
    """
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Pipeline configuration file '{config_file}' not found")
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing pipeline YAML file: {e}")

def get_expected_input_files(step_name: str, pipeline_config: Dict[str, Any] = None):
    """
    Get the expected input files for a step based on the step name.
    
    Args:
        step_name (str): Name of the step
        pipeline_config (Dict[str, Any]): Pipeline configuration (needed for clean_ticks step)
        
    Returns:
        List[str]: List of expected input file paths
    """
    if step_name == "merge":
        # Merge doesn't have input files from previous steps
        return None
    elif step_name == "clean_ticks":
        # Clean ticks expects <symbol>_ticks.parquet files where symbols come from merge section
        if pipeline_config is None:
            raise ValueError("Pipeline config is required for clean_ticks step to determine symbols")
        
        symbols = [symbol_config['symbol'] for symbol_config in pipeline_config['merge']['symbols']]
        files = []
        missing_files = []
        for symbol in symbols:
            file_path = os.path.join(pipeline_config['output_directory'], f"{symbol}_ticks.parquet")
            if os.path.exists(file_path):
                files.append(file_path)
            else:
                missing_files.append(file_path)
        
        if missing_files:
            raise FileNotFoundError(f"Required input files for clean_ticks step not found: {missing_files}")
        
        files.sort()
        return files
    elif step_name == "aggregate":
        # Aggregate expects <symbol>_ticks_cleaned.parquet files where symbols come from merge section
        if pipeline_config is None:
            raise ValueError("Pipeline config is required for aggregate step to determine symbols")
        
        symbols = [symbol_config['symbol'] for symbol_config in pipeline_config['merge']['symbols']]
        files = []
        missing_files = []
        for symbol in symbols:
            file_path = os.path.join(pipeline_config['output_directory'], f"{symbol}_ticks_cleaned.parquet")
            if os.path.exists(file_path):
                files.append(file_path)
            else:
                missing_files.append(file_path)
        
        if missing_files:
            raise FileNotFoundError(f"Required input files for aggregate step not found: {missing_files}")
        
        files.sort()
        return files
    elif step_name == "clean_aggregated":
        # Clean aggregated expects <time_freq>_<agg_output_filename>.parquet files
        if pipeline_config is None:
            raise ValueError("Pipeline config is required for clean_aggregated step to determine output filename")
        
        time_window = pipeline_config['aggregate']['time_window']
        output_filename = pipeline_config['aggregate']['output_filename']
        file_path = os.path.join(pipeline_config['output_directory'], f"{time_window}_{output_filename}.parquet")
        
        if os.path.exists(file_path):
            return file_path
        else:
            raise FileNotFoundError(f"Required input file for clean_aggregated step not found: {file_path}")
    elif step_name == "augment":
        # Augment expects <time_freq>_<agg_output_filename>_cleaned.parquet files
        if pipeline_config is None:
            raise ValueError("Pipeline config is required for augment step to determine output filename")
        
        time_window = pipeline_config['aggregate']['time_window']
        output_filename = pipeline_config['aggregate']['output_filename']
        file_path = os.path.join(pipeline_config['output_directory'], f"{time_window}_{output_filename}_cleaned.parquet")
        
        if os.path.exists(file_path):
            return [file_path]  # Return as list since augment expects list of files
        else:
            raise FileNotFoundError(f"Required input file for augment step not found: {file_path}")
    elif step_name == "clean_augmented":
        # Clean augmented expects <time_freq>_<agg_output_filename>_cleaned_<output_filename>_augmented.parquet files
        if pipeline_config is None:
            raise ValueError("Pipeline config is required for clean_augmented step to determine output filename")
        
        time_window = pipeline_config['aggregate']['time_window']
        output_filename = pipeline_config['aggregate']['output_filename']
        augment_output_filename = pipeline_config['augment'].get('output_filename', None)
        
        if augment_output_filename:
            file_path = os.path.join(pipeline_config['output_directory'], f"{time_window}_{output_filename}_cleaned_{augment_output_filename}_augmented.parquet")
        else:
            file_path = os.path.join(pipeline_config['output_directory'], f"{time_window}_{output_filename}_cleaned_augmented.parquet")
        
        if os.path.exists(file_path):
            return [file_path]  # Return as list since clean expects list of files
        else:
            raise FileNotFoundError(f"Required input file for clean_augmented step not found: {file_path}")
    elif step_name == "visualize":
        # Visualize expects the final cleaned augmented file
        if pipeline_config is None:
            raise ValueError("Pipeline config is required for visualize step to determine output filename")
        
        time_window = pipeline_config['aggregate']['time_window']
        output_filename = pipeline_config['aggregate']['output_filename']
        augment_output_filename = pipeline_config['augment'].get('output_filename', None)
        
        if augment_output_filename:
            file_path = os.path.join(pipeline_config['output_directory'], f"{time_window}_{output_filename}_cleaned_{augment_output_filename}_augmented_cleaned.parquet")
        else:
            file_path = os.path.join(pipeline_config['output_directory'], f"{time_window}_{output_filename}_cleaned_augmented_cleaned.parquet")
        
        if os.path.exists(file_path):
            return file_path  # Return as single file path
        else:
            raise FileNotFoundError(f"Required input file for visualize step not found: {file_path}")
    else:
        raise ValueError(f"Unknown step: {step_name}")

def generate_merge_config(pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate merge configuration dictionary.
    
    Args:
        pipeline_config (Dict[str, Any]): Main pipeline configuration
        
    Returns:
        Dict[str, Any]: Generated merge configuration dictionary
    """
    print("\n🔧 Generating merge configuration...")
    
    merge_config = {
        'symbols': pipeline_config['merge']['symbols'],
        'output_directory': pipeline_config['output_directory']
    }
    
    return merge_config


def generate_clean_ticks_config(pipeline_config: Dict[str, Any], input_files: List[str]) -> Dict[str, Any]:
    """
    Generate clean configuration dictionary for tick data.
    
    Args:
        pipeline_config (Dict[str, Any]): Main pipeline configuration
        input_files (List[str]): List of input file paths
        
    Returns:
        Dict[str, Any]: Generated clean configuration dictionary
    """
    print("\n🔧 Generating clean ticks configuration...")
    
    clean_config = {
        'pipeline_stage': 'ticks',
        'stages': {
            'ticks': {
                'input_files': input_files,
                'output_directory': pipeline_config['output_directory'],
                'time_column': pipeline_config['clean_ticks']['time_column']
            }
        }
    }
        
    return clean_config


def generate_aggregate_config(pipeline_config: Dict[str, Any], input_files: List[str]) -> Dict[str, Any]:
    """
    Generate aggregate configuration dictionary.
    
    Args:
        pipeline_config (Dict[str, Any]): Main pipeline configuration
        input_files (List[str]): List of input file paths
        
    Returns:
        Dict[str, Any]: Generated aggregate configuration dictionary
    """
    print("\n🔧 Generating aggregate configuration...")
    
    aggregate_config = {
        'time_window': pipeline_config['aggregate']['time_window'],
        'input_files': input_files,
        'output_directory': pipeline_config['output_directory'],
        'aggregators': pipeline_config['aggregate']['aggregators'],
        'output_filename': pipeline_config['aggregate']['output_filename']
    }
    
    return aggregate_config


def generate_clean_aggregated_config(pipeline_config: Dict[str, Any], input_file: str) -> Dict[str, Any]:
    """
    Generate clean configuration dictionary for aggregated data.
    
    Args:
        pipeline_config (Dict[str, Any]): Main pipeline configuration
        input_file (str): Path to the input file
        
    Returns:
        Dict[str, Any]: Generated clean configuration dictionary
    """
    print("\n🔧 Generating clean aggregated configuration...")
    
    clean_config = {
        'pipeline_stage': 'aggregated',
        'stages': {
            'aggregated': {
                'input_files': [input_file],
                'output_directory': pipeline_config['output_directory'],
                'time_column': 'start_time'
            }
        }
    }
        
    return clean_config


def generate_augment_config(pipeline_config: Dict[str, Any], input_files: List[str]) -> Dict[str, Any]:
    """
    Generate augment configuration dictionary.
    
    Args:
        pipeline_config (Dict[str, Any]): Main pipeline configuration
        input_files (List[str]): List of input file paths
        
    Returns:
        Dict[str, Any]: Generated augment configuration dictionary
    """
    print("\n🔧 Generating augment configuration...")
    
    augment_config = {
        'input_files': input_files,
        'output_directory': pipeline_config['output_directory'],
        'augmentations': pipeline_config['augment']['augmentations'],
        'output_filename': pipeline_config['augment'].get('output_filename', None)
    }
    
    return augment_config


def generate_clean_augmented_config(pipeline_config: Dict[str, Any], input_files: List[str]) -> Dict[str, Any]:
    """
    Generate clean configuration dictionary for augmented data.
    
    Args:
        pipeline_config (Dict[str, Any]): Main pipeline configuration
        input_files (List[str]): List of input file paths
        
    Returns:
        Dict[str, Any]: Generated clean configuration dictionary
    """
    print("\n🔧 Generating clean augmented configuration...")
    
    clean_config = {
        'pipeline_stage': 'augmented',
        'stages': {
            'augmented': {
                'input_files': input_files,
                'output_directory': pipeline_config['output_directory'],
                'time_column': 'start_time'
            }
        }
    }
        
    return clean_config


def generate_visualize_config(pipeline_config: Dict[str, Any], input_file: str) -> Dict[str, Any]:
    """
    Generate visualize configuration dictionary.
    
    Args:
        pipeline_config (Dict[str, Any]): Main pipeline configuration
        input_file (str): Path to the input file
        
    Returns:
        Dict[str, Any]: Generated visualize configuration dictionary
    """
    print("\n🔧 Generating visualize configuration...")
    
    visualize_config = {
        'input_file': input_file,
        'output_directory': pipeline_config['output_directory'],
        'time_column': 'start_time',
        'split': pipeline_config.get('visualize', {}).get('split', {'train': 0.7, 'val': 0.15, 'test': 0.15})
    }
    
    return visualize_config


def validate_and_order_steps(steps_to_execute: List[str]) -> List[str]:
    """
    Validate and reorder steps to execute in correct sequence with dependency warnings.
    
    Args:
        steps_to_execute (List[str]): List of steps to execute
        
    Returns:
        List[str]: Ordered list of steps to execute
    """
    # Define the correct order - each step depends on the previous one
    correct_order = ["merge", "clean_ticks", "aggregate", "clean_aggregated", "augment", "clean_augmented", "visualize"]
    
    # Reorder steps to execute in correct sequence
    ordered_steps = []
    for step in correct_order:
        if step in steps_to_execute:
            ordered_steps.append(step)
    
    # Check for missing dependencies and warn
    for i, step in enumerate(ordered_steps):
        if i > 0:  # Skip first step (merge) as it has no dependencies
            previous_step = correct_order[correct_order.index(step) - 1]
            if previous_step not in ordered_steps:
                print(f"⚠️  Warning: {step} step depends on {previous_step} but it's not in the execution list.")
                print(f"   The pipeline will run using previously saved data from {previous_step} step.")
    
    if ordered_steps != steps_to_execute:
        print(f"🔄 Reordering steps to correct sequence: {', '.join(ordered_steps)}")
    
    return ordered_steps


def run_pipeline_step(step_name: str, step_function, config):
    """
    Run a pipeline step with error handling.
    
    Args:
        step_name (str): Name of the step for logging
        step_function: Function to run the step
        config (Dict[str, Any]): Configuration dictionary for the step
    """
    try:
        print(f"\n{'='*60}")
        print(f"🚀 Starting {step_name} step...")
        print(f"{'='*60}")
        
        step_function(config)
        
        print(f"✅ {step_name} step completed successfully!")
        
    except Exception as e:
        print(f"❌ {step_name} step failed: {e}")
        raise


def run_pipeline(config_file="pipeline.yaml"):
    """
    Run the data processing pipeline with dynamic step execution.
    
    Args:
        config_file (str): Path to the pipeline configuration file
    """
    try:
        print("🎯 DYNAMIC FOREX DATA PROCESSING PIPELINE")
        print("=" * 60)
        print(f"📋 Loading pipeline configuration from {config_file}...")
        
        # Load pipeline configuration
        pipeline_config = load_pipeline_config(config_file)
        
        # Get steps to execute
        steps_to_execute = pipeline_config.get('steps', [])
        if not steps_to_execute:
            print("❌ No steps specified in pipeline configuration")
            return
        
        print(f"📋 Steps to execute: {', '.join(steps_to_execute)}")
        
        # Validate and reorder steps with dependency warnings
        steps_to_execute = validate_and_order_steps(steps_to_execute)
    
        # Track pipeline progress
        pipeline_success = True
        executed_steps = []
        
        # Execute each step in order
        for i, step_name in enumerate(steps_to_execute, 1):
            print(f"\n📊 Pipeline Step {i}/{len(steps_to_execute)}: {step_name.upper()}")
            
            # Get expected input files for this step
            expected_input_files = get_expected_input_files(step_name, pipeline_config)
            
            # Run the step
            try:
                if step_name == "merge":
                    run_pipeline_step(step_name, run_merge, generate_merge_config(pipeline_config))
                elif step_name == "clean_ticks":
                    run_pipeline_step(step_name, run_clean, generate_clean_ticks_config(pipeline_config, expected_input_files))
                elif step_name == "aggregate":
                    run_pipeline_step(step_name, run_aggregate, generate_aggregate_config(pipeline_config, expected_input_files))
                elif step_name == "clean_aggregated":
                    run_pipeline_step(step_name, run_clean, generate_clean_aggregated_config(pipeline_config, expected_input_files))
                elif step_name == "augment":
                    run_pipeline_step(step_name, run_augment, generate_augment_config(pipeline_config, expected_input_files))
                elif step_name == "clean_augmented":
                    run_pipeline_step(step_name, run_clean, generate_clean_augmented_config(pipeline_config, expected_input_files))
                elif step_name == "visualize":
                    run_pipeline_step(step_name, run_visualize, generate_visualize_config(pipeline_config, expected_input_files))
                
                executed_steps.append(step_name)
                
            except Exception as e:
                print(f"❌ Pipeline failed at {step_name} step: {e}")
                pipeline_success = False
                break
        
        # PIPELINE SUMMARY
        print(f"\n{'='*60}")
        print("🎯 PIPELINE COMPLETION SUMMARY")
        print(f"{'='*60}")
        
        if pipeline_success:
            print("✅ All specified steps completed successfully!")
            print(f"📊 Steps executed: {len(executed_steps)}/{len(steps_to_execute)}")
            print(f"📋 Executed steps: {', '.join(executed_steps)}")
            
        else:
            print("❌ Pipeline failed - some steps did not complete successfully")
            print(f"📊 Steps completed: {len(executed_steps)}/{len(steps_to_execute)}")
            if executed_steps:
                print(f"📋 Completed steps: {', '.join(executed_steps)}")
        
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"❌ Pipeline failed with error: {e}")


def main():
    """Main function to run the pipeline."""
    run_pipeline()
    print(f"\n📋 Pipeline execution completed!")


if __name__ == "__main__":
    main() 