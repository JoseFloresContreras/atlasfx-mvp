"""
End-to-end pipeline test.

This test validates the complete data pipeline execution, ensuring:
- The pipeline runs successfully twice consecutively
- Output files are generated correctly
- The pipeline is idempotent (same output on repeated runs)
- Generated parquet files are valid and non-empty
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from atlasfx.data.validators import DataValidator

import pytest
@pytest.mark.integration

def test_pipeline_e2e():
    """
    Test end-to-end pipeline execution.

    This test:
    1. Cleans and recreates the output directory
    2. Runs the pipeline twice consecutively
    3. Verifies both runs complete successfully (returncode == 0)
    4. Confirms idempotency by comparing file lists and sizes
    5. Validates that parquet files are readable and non-empty
    """
    # Define paths
    pipeline_script = Path("scripts/run_data_pipeline.py")
    config_file = Path("tests/fixtures/e2e_pipeline_config.yaml")
    output_dir = Path("tests/fixtures/e2e_test_output")

    # Step 1: Clean up output directory if it exists
    if output_dir.exists():
        shutil.rmtree(output_dir)

    # Step 2: Recreate output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 3: First pipeline run
    result1 = subprocess.run(
        [sys.executable, str(pipeline_script), "--config", str(config_file)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    # Verify first run succeeded
    assert result1.returncode == 0, (
        f"First pipeline run failed with returncode {result1.returncode}.\n"
        f"STDOUT:\n{result1.stdout}\n"
        f"STDERR:\n{result1.stderr}"
    )

    # Collect file information after first run
    first_run_files = {}
    for parquet_file in output_dir.glob("*.parquet"):
        first_run_files[parquet_file.name] = parquet_file.stat().st_size

    # Verify at least one parquet file was created
    assert len(first_run_files) > 0, (
        f"No parquet files found after first pipeline run.\n"
        f"STDOUT:\n{result1.stdout}\n"
        f"STDERR:\n{result1.stderr}"
    )

    # Step 4: Second pipeline run (test idempotency)
    result2 = subprocess.run(
        [sys.executable, str(pipeline_script), "--config", str(config_file)],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    # Verify second run succeeded
    assert result2.returncode == 0, (
        f"Second pipeline run failed with returncode {result2.returncode}.\n"
        f"STDOUT:\n{result2.stdout}\n"
        f"STDERR:\n{result2.stderr}"
    )

    # Collect file information after second run
    second_run_files = {}
    for parquet_file in output_dir.glob("*.parquet"):
        second_run_files[parquet_file.name] = parquet_file.stat().st_size

    # Step 5: Verify idempotency - same files and sizes
    assert first_run_files.keys() == second_run_files.keys(), (
        f"File lists differ between runs.\n"
        f"First run: {sorted(first_run_files.keys())}\n"
        f"Second run: {sorted(second_run_files.keys())}"
    )

    for filename in first_run_files:
        assert first_run_files[filename] == second_run_files[filename], (
            f"File size changed for {filename}.\n"
            f"First run: {first_run_files[filename]} bytes\n"
            f"Second run: {second_run_files[filename]} bytes"
        )

    # Step 6: Verify parquet files are readable and non-empty
    for parquet_file in output_dir.glob("*.parquet"):
        # Read with pandas to verify file is valid
        df = pd.read_parquet(parquet_file)

        # Verify the dataframe is not empty
        assert not df.empty, f"Parquet file {parquet_file.name} is empty"
        assert len(df) > 0, f"Parquet file {parquet_file.name} has no rows"

    # Step 7: Extended validation - structure, data types, and schema compliance
    validator = DataValidator(schema_path="configs/schema.yaml")

    for parquet_file in output_dir.glob("*.parquet"):
        # Load data with pandas
        df = pd.read_parquet(parquet_file)

        # Verify the DataFrame is not empty
        assert not df.empty, (
            f"Parquet file {parquet_file.name} is empty.\n"
            f"Shape: {df.shape}, Columns: {df.columns.tolist()}"
        )

        # Check required columns
        required_columns = ["timestamp", "bid", "ask", "volume"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        assert len(missing_columns) == 0, (
            f"Parquet file {parquet_file.name} is missing required columns: {missing_columns}\n"
            f"Found columns: {df.columns.tolist()}\n"
            f"Shape: {df.shape}"
        )

        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Verify data types
        # timestamp should be datetime64 or datetime64[ns, UTC]
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]), (
            f"Parquet file {parquet_file.name}: 'timestamp' column has incorrect type.\n"
            f"Expected: datetime64 or datetime64[ns, UTC], Got: {df['timestamp'].dtype}\n"
            f"Shape: {df.shape}, Columns: {df.columns.tolist()}"
        )

        # bid, ask, volume should be float
        for col in ["bid", "ask", "volume"]:
            assert pd.api.types.is_float_dtype(df[col]), (
                f"Parquet file {parquet_file.name}: '{col}' column has incorrect type.\n"
                f"Expected: float, Got: {df[col].dtype}\n"
                f"Shape: {df.shape}, Columns: {df.columns.tolist()}"
            )

        # Verify ask >= bid in all records
        crossed_spreads = df["ask"] < df["bid"]
        assert not crossed_spreads.any(), (
            f"Parquet file {parquet_file.name}: Found {crossed_spreads.sum()} records "
            f"where ask < bid (crossed spreads).\n"
            f"Shape: {df.shape}, Columns: {df.columns.tolist()}"
        )

        # Execute validator.validate_tick_data() for schema compliance
        is_valid, errors = validator.validate_tick_data(df)
        assert is_valid, (
            f"Parquet file {parquet_file.name} failed schema validation.\n"
            f"Shape: {df.shape}, Columns: {df.columns.tolist()}\n"
            f"Validation errors:\n" + "\n".join(f"  - {error}" for error in errors)
        )
