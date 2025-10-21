#!/usr/bin/env python3
"""
Dynamic Data Processing Pipeline
This script orchestrates data processing steps based on configuration.
Supports flexible step execution order and dependency management.
"""

import os
from typing import Any, Dict, List

import yaml

from atlasfx.data.aggregation import run_aggregate
from atlasfx.data.cleaning import run_clean
from atlasfx.data.featurization import run_featurize

# Import the run functions from each module
from atlasfx.data.loaders import run_merge
from atlasfx.data.normalization import run_normalize
from atlasfx.data.splitters import run_split
from atlasfx.data.winsorization import run_winsorize
from atlasfx.evaluation.visualizers import run_visualize
from atlasfx.utils.logging import log


def load_pipeline_config(config_file="configs/data_pipeline.yaml"):
    """
    Load the main pipeline configuration.

    Args:
        config_file (str): Path to the pipeline configuration file

    Returns:
        dict: Pipeline configuration dictionary
    """
    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError as e:
        error_msg = f"Pipeline configuration file '{config_file}' not found"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise FileNotFoundError(error_msg) from e
    except yaml.YAMLError as e:
        error_msg = f"Error parsing pipeline YAML file: {e}"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise yaml.YAMLError(error_msg) from e


def get_expected_input_files(step_name: str, pipeline_config: Dict[str, Any] = None):
    """
    Get the expected input files for a step based on the step name.

    Args:
        step_name (str): Name of the step
        pipeline_config (Dict[str, Any]): Pipeline configuration (needed for clean_ticks step)

    Returns:
        List[str]: List of expected input file paths
    """
    # Check if pipeline_config is required and provided
    steps_requiring_config = [
        "clean_ticks",
        "aggregate",
        "split",
        "winsorize",
        "clean_aggregated",
        "featurize",
        "clean_featurized",
        "normalize",
        "visualize",
    ]
    if step_name in steps_requiring_config and pipeline_config is None:
        error_msg = f"Pipeline config is required for {step_name} step"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise ValueError(error_msg)

    def get_split_files(time_window: str, output_filename: str, output_directory: str):
        """
        Helper function to get split files.

        Args:
            time_window (str): Time window from aggregate config
            output_filename (str): Output filename from aggregate config
            output_directory (str): Output directory

        Returns:
            List[str]: List of file paths
        """
        suffixes = ["_train", "_val", "_test"]
        files = []
        missing_files = []

        for suffix in suffixes:
            file_path = os.path.join(
                output_directory, f"{time_window}_{output_filename}{suffix}.parquet"
            )
            if os.path.exists(file_path):
                files.append(file_path)
            else:
                missing_files.append(file_path)

        if missing_files:
            error_msg = f"Required input files not found: {missing_files}"
            log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
            raise FileNotFoundError(error_msg)

        files.sort()
        return files

    if step_name == "merge":
        # Merge doesn't have input files from previous steps
        return None
    elif step_name == "clean_ticks":
        # Clean ticks expects <symbol>-pair_ticks.parquet and <symbol>-instrument_ticks.parquet files
        files = []
        missing_files = []

        # Get pairs and instruments from merge config
        pairs = [
            symbol_config["symbol"] for symbol_config in pipeline_config["merge"].get("pairs", [])
        ]
        instruments = [
            symbol_config["symbol"]
            for symbol_config in pipeline_config["merge"].get("instruments", [])
        ]

        # Check for pair files
        for symbol in pairs:
            file_path = os.path.join(
                pipeline_config["output_directory"], f"{symbol}-pair_ticks.parquet"
            )
            if os.path.exists(file_path):
                files.append(file_path)
            else:
                missing_files.append(file_path)

        # Check for instrument files
        for symbol in instruments:
            file_path = os.path.join(
                pipeline_config["output_directory"], f"{symbol}-instrument_ticks.parquet"
            )
            if os.path.exists(file_path):
                files.append(file_path)
            else:
                missing_files.append(file_path)

        if missing_files:
            error_msg = f"Required input files for clean_ticks step not found: {missing_files}"
            log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
            raise FileNotFoundError(error_msg)

        files.sort()
        return files
    elif step_name == "aggregate":
        # Aggregate expects <symbol>-pair_ticks.parquet and <symbol>-instrument_ticks.parquet files
        files = []
        missing_files = []

        # Get pairs and instruments from merge config
        pairs = [
            symbol_config["symbol"] for symbol_config in pipeline_config["merge"].get("pairs", [])
        ]
        instruments = [
            symbol_config["symbol"]
            for symbol_config in pipeline_config["merge"].get("instruments", [])
        ]

        # Check for pair files
        for symbol in pairs:
            file_path = os.path.join(
                pipeline_config["output_directory"], f"{symbol}-pair_ticks.parquet"
            )
            if os.path.exists(file_path):
                files.append(file_path)
            else:
                missing_files.append(file_path)

        # Check for instrument files
        for symbol in instruments:
            file_path = os.path.join(
                pipeline_config["output_directory"], f"{symbol}-instrument_ticks.parquet"
            )
            if os.path.exists(file_path):
                files.append(file_path)
            else:
                missing_files.append(file_path)

        if missing_files:
            error_msg = f"Required input files for aggregate step not found: {missing_files}"
            log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
            raise FileNotFoundError(error_msg)

        files.sort()
        return files
    elif step_name == "split":
        # Split expects <time_freq>_<agg_output_filename>.parquet files
        time_window = pipeline_config["aggregate"]["time_window"]
        output_filename = pipeline_config["aggregate"]["output_filename"]
        file_path = os.path.join(
            pipeline_config["output_directory"], f"{time_window}_{output_filename}.parquet"
        )

        if os.path.exists(file_path):
            return file_path
        else:
            error_msg = f"Required input file for split step not found: {file_path}"
            log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
            raise FileNotFoundError(error_msg)
    elif step_name in [
        "winsorize",
        "clean_aggregated",
        "featurize",
        "clean_featurized",
        "normalize",
        "visualize",
    ]:
        # These steps expect <time_freq>_<output_filename>_train.parquet, _val.parquet, _test.parquet files
        time_window = pipeline_config["aggregate"]["time_window"]
        output_filename = pipeline_config["aggregate"]["output_filename"]
        return get_split_files(time_window, output_filename, pipeline_config["output_directory"])
    else:
        error_msg = f"Unknown step: {step_name}"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)
        raise ValueError(error_msg)


def generate_merge_config(pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate merge configuration dictionary.

    Args:
        pipeline_config (Dict[str, Any]): Main pipeline configuration

    Returns:
        Dict[str, Any]: Generated merge configuration dictionary
    """
    log.info("\n🔧 Generating merge configuration...")

    merge_config = {
        "pairs": pipeline_config["merge"].get("pairs", []),
        "instruments": pipeline_config["merge"].get("instruments", []),
        "output_directory": pipeline_config["output_directory"],
        "time_column": pipeline_config["merge"]["time_column"],
    }

    return merge_config


def generate_clean_ticks_config(
    pipeline_config: Dict[str, Any], input_files: List[str]
) -> Dict[str, Any]:
    """
    Generate clean configuration dictionary for tick data.

    Args:
        pipeline_config (Dict[str, Any]): Main pipeline configuration
        input_files (List[str]): List of input file paths

    Returns:
        Dict[str, Any]: Generated clean configuration dictionary
    """
    log.info("\n🔧 Generating clean ticks configuration...")

    clean_config = {
        "pipeline_stage": "ticks",
        "stages": {
            "ticks": {
                "input_files": input_files,
                "output_directory": pipeline_config["output_directory"],
                "time_column": pipeline_config["merge"]["time_column"],
            }
        },
    }

    return clean_config


def generate_aggregate_config(
    pipeline_config: Dict[str, Any], input_files: List[str]
) -> Dict[str, Any]:
    """
    Generate aggregate configuration dictionary.

    Args:
        pipeline_config (Dict[str, Any]): Main pipeline configuration
        input_files (List[str]): List of input file paths

    Returns:
        Dict[str, Any]: Generated aggregate configuration dictionary
    """
    log.info("\n🔧 Generating aggregate configuration...")

    aggregate_config = {
        "time_window": pipeline_config["aggregate"]["time_window"],
        "input_files": input_files,
        "output_directory": pipeline_config["output_directory"],
        "aggregators": pipeline_config["aggregate"]["aggregators"],
        "output_filename": pipeline_config["aggregate"]["output_filename"],
    }

    return aggregate_config


def generate_split_config(pipeline_config: Dict[str, Any], input_file: str) -> Dict[str, Any]:
    """
    Generate split configuration dictionary.

    Args:
        pipeline_config (Dict[str, Any]): Main pipeline configuration
        input_file (str): Path to the input file

    Returns:
        Dict[str, Any]: Generated split configuration dictionary
    """
    log.info("\n🔧 Generating split configuration...")

    split_config = {
        "input_file": input_file,
        "output_directory": pipeline_config["output_directory"],
        "split": pipeline_config.get("split", {"train": 0.7, "val": 0.15, "test": 0.15}),
    }

    return split_config


def generate_winsorize_config(
    pipeline_config: Dict[str, Any], input_files: List[str]
) -> Dict[str, Any]:
    """
    Generate winsorize configuration dictionary.

    Args:
        pipeline_config (Dict[str, Any]): Main pipeline configuration
        input_files (List[str]): List of input file paths

    Returns:
        Dict[str, Any]: Generated winsorize configuration dictionary
    """
    log.info("\n🔧 Generating winsorize configuration...")

    winsorize_config = {
        "input_files": input_files,
        "output_directory": pipeline_config["output_directory"],
        "winsorization_configs": pipeline_config.get("winsorize", []),
        "time_window": pipeline_config["aggregate"]["time_window"],
    }

    return winsorize_config


def generate_clean_aggregated_config(
    pipeline_config: Dict[str, Any], input_files: List[str]
) -> Dict[str, Any]:
    """
    Generate clean configuration dictionary for aggregated data.

    Args:
        pipeline_config (Dict[str, Any]): Main pipeline configuration
        input_files (List[str]): List of input file paths

    Returns:
        Dict[str, Any]: Generated clean configuration dictionary
    """
    log.info("\n🔧 Generating clean aggregated configuration...")

    clean_config = {
        "pipeline_stage": "aggregated",
        "stages": {
            "aggregated": {
                "input_files": input_files,
                "output_directory": pipeline_config["output_directory"],
                "time_column": "start_time",
            }
        },
    }

    return clean_config


def generate_featurize_config(
    pipeline_config: Dict[str, Any], input_files: List[str]
) -> Dict[str, Any]:
    """
    Generate featurize configuration dictionary.

    Args:
        pipeline_config (Dict[str, Any]): Main pipeline configuration
        input_files (List[str]): List of input file paths

    Returns:
        Dict[str, Any]: Generated featurize configuration dictionary
    """
    log.info("\n🔧 Generating featurize configuration...")

    featurize_config = {
        "input_files": input_files,
        "output_directory": pipeline_config["output_directory"],
        "featurizers": pipeline_config["featurize"]["featurizers"],
    }

    return featurize_config


def generate_clean_featurized_config(
    pipeline_config: Dict[str, Any], input_files: List[str]
) -> Dict[str, Any]:
    """
    Generate clean configuration dictionary for featurized data.

    Args:
        pipeline_config (Dict[str, Any]): Main pipeline configuration
        input_files (List[str]): List of input file paths

    Returns:
        Dict[str, Any]: Generated clean configuration dictionary
    """
    log.info("\n🔧 Generating clean featurized configuration...")

    clean_config = {
        "pipeline_stage": "featurized",
        "stages": {
            "featurized": {
                "input_files": input_files,
                "output_directory": pipeline_config["output_directory"],
                "time_column": "start_time",
            }
        },
    }

    return clean_config


def generate_normalize_config(
    pipeline_config: Dict[str, Any], input_files: List[str]
) -> Dict[str, Any]:
    """
    Generate normalize configuration dictionary.

    Args:
        pipeline_config (Dict[str, Any]): Main pipeline configuration
        input_files (List[str]): List of input file paths

    Returns:
        Dict[str, Any]: Generated normalize configuration dictionary
    """
    log.info("\n🔧 Generating normalize configuration...")

    normalize_config = {
        "input_files": input_files,
        "output_directory": pipeline_config["output_directory"],
        "time_window": pipeline_config["aggregate"]["time_window"],
        "clip_threshold": pipeline_config["normalize"][
            "clip_threshold"
        ],  # Keep for pickle file naming
    }

    return normalize_config


def generate_visualize_config(
    pipeline_config: Dict[str, Any], input_files: List[str]
) -> Dict[str, Any]:
    """
    Generate visualize configuration dictionary.

    Args:
        pipeline_config (Dict[str, Any]): Main pipeline configuration
        input_files (List[str]): List of input file paths

    Returns:
        Dict[str, Any]: Generated visualize configuration dictionary
    """
    log.info("\n🔧 Generating visualize configuration...")

    visualize_config = {
        "input_files": input_files,
        "output_directory": pipeline_config["output_directory"],
        "time_column": "start_time",
        "time_window": pipeline_config["aggregate"]["time_window"],
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
    correct_order = [
        "merge",
        "clean_ticks",
        "aggregate",
        "split",
        "winsorize",
        "clean_aggregated",
        "featurize",
        "clean_featurized",
        "normalize",
        "visualize",
    ]

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
                log.warning(
                    f"⚠️  Warning: {step} step depends on {previous_step} but it's not in the execution list."
                )
                log.warning(
                    f"   The pipeline will run using previously saved data from {previous_step} step."
                )

    if ordered_steps != steps_to_execute:
        log.info(f"🔄 Reordering steps to correct sequence: {', '.join(ordered_steps)}")

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
        log.info(f"\n{'='*60}", also_print=True)
        log.info(f"🚀 Starting {step_name} step...", also_print=True)
        log.info(f"{'='*60}", also_print=True)

        step_function(config)

        log.info(f"✅ {step_name} step completed successfully!", also_print=True)

    except Exception as e:
        log.error(f"❌ {step_name} step failed: {e}")
        raise


def run_pipeline(config_file="pipeline.yaml"):
    """
    Run the data processing pipeline with dynamic step execution.

    Args:
        config_file (str): Path to the pipeline configuration file
    """
    try:
        log.info("🎯 DYNAMIC FOREX DATA PROCESSING PIPELINE", also_print=True)
        log.info("=" * 60, also_print=True)
        log.info(f"📋 Loading pipeline configuration from {config_file}...", also_print=True)

        # Load pipeline configuration
        pipeline_config = load_pipeline_config(config_file)

        # Get steps to execute
        steps_to_execute = pipeline_config.get("steps", [])
        if not steps_to_execute:
            log.error("❌ No steps specified in pipeline configuration")
            return

        log.info(f"📋 Steps to execute: {', '.join(steps_to_execute)}", also_print=True)

        # Validate and reorder steps with dependency warnings
        steps_to_execute = validate_and_order_steps(steps_to_execute)

        # Track pipeline progress
        pipeline_success = True
        executed_steps = []

        # Execute each step in order
        for i, step_name in enumerate(steps_to_execute, 1):
            log.info(
                f"\n📊 Pipeline Step {i}/{len(steps_to_execute)}: {step_name.upper()}",
                also_print=True,
            )

            # Get expected input files for this step
            expected_input_files = get_expected_input_files(step_name, pipeline_config)

            # Run the step
            try:
                if step_name == "merge":
                    run_pipeline_step(step_name, run_merge, generate_merge_config(pipeline_config))
                elif step_name == "clean_ticks":
                    run_pipeline_step(
                        step_name,
                        run_clean,
                        generate_clean_ticks_config(pipeline_config, expected_input_files),
                    )
                elif step_name == "aggregate":
                    run_pipeline_step(
                        step_name,
                        run_aggregate,
                        generate_aggregate_config(pipeline_config, expected_input_files),
                    )
                elif step_name == "split":
                    run_pipeline_step(
                        step_name,
                        run_split,
                        generate_split_config(pipeline_config, expected_input_files),
                    )
                elif step_name == "winsorize":
                    run_pipeline_step(
                        step_name,
                        run_winsorize,
                        generate_winsorize_config(pipeline_config, expected_input_files),
                    )
                elif step_name == "clean_aggregated":
                    run_pipeline_step(
                        step_name,
                        run_clean,
                        generate_clean_aggregated_config(pipeline_config, expected_input_files),
                    )
                elif step_name == "featurize":
                    run_pipeline_step(
                        step_name,
                        run_featurize,
                        generate_featurize_config(pipeline_config, expected_input_files),
                    )
                elif step_name == "clean_featurized":
                    run_pipeline_step(
                        step_name,
                        run_clean,
                        generate_clean_featurized_config(pipeline_config, expected_input_files),
                    )
                elif step_name == "normalize":
                    run_pipeline_step(
                        step_name,
                        run_normalize,
                        generate_normalize_config(pipeline_config, expected_input_files),
                    )
                elif step_name == "visualize":
                    run_pipeline_step(
                        step_name,
                        run_visualize,
                        generate_visualize_config(pipeline_config, expected_input_files),
                    )

                executed_steps.append(step_name)

            except Exception as e:
                log.error(f"❌ Pipeline failed at {step_name} step: {e}")
                pipeline_success = False
                break

        # PIPELINE SUMMARY
        log.info(f"\n{'='*60}", also_print=True)
        log.info("🎯 PIPELINE COMPLETION SUMMARY", also_print=True)
        log.info(f"{'='*60}", also_print=True)

        if pipeline_success:
            log.info("✅ All specified steps completed successfully!", also_print=True)
            log.info(
                f"📊 Steps executed: {len(executed_steps)}/{len(steps_to_execute)}", also_print=True
            )
            log.info(f"📋 Executed steps: {', '.join(executed_steps)}", also_print=True)

        else:
            log.error(
                "❌ Pipeline failed - some steps did not complete successfully", also_print=True
            )
            log.info(
                f"📊 Steps completed: {len(executed_steps)}/{len(steps_to_execute)}",
                also_print=True,
            )
            if executed_steps:
                log.info(f"📋 Completed steps: {', '.join(executed_steps)}", also_print=True)

        log.info(f"{'='*60}", also_print=True)

    except Exception as e:
        error_msg = f"Pipeline failed with error: {e}"
        log.critical(f"❌ CRITICAL ERROR: {error_msg}", also_print=True)


def main():
    """Main function to run the pipeline."""
    run_pipeline()
    log.info(f"\n📋 Pipeline execution completed!")


if __name__ == "__main__":
    main()
