#!/usr/bin/env python3
"""
AtlasFX End-to-End Integration Tests (v3.3)
-------------------------------------------
Comprehensive enterprise-grade validation of the AtlasFX data pipeline.
Ensures that every stage — from merge to normalization —
produces valid, deterministic, schema-compliant, and logically consistent outputs.

Test Focus:
✅ Full pipeline execution (multi-step)
✅ Schema & data integrity (via DataValidator)
✅ Deterministic reproducibility (hash-based)
✅ Parquet readability and compliance
✅ CLI validator consistency
"""

from __future__ import annotations

import os
import sys
import shutil
import subprocess
import tempfile
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from atlasfx.data.validators import DataValidator
from atlasfx.utils.pathing import cd_repo_root, resolve_path, resolve_repo_root, get_fixtures_dir
from atlasfx.utils.logging import log


# ============================================================
# FIXTURES & HELPERS
# ============================================================

@pytest.fixture(scope="function")
def cleanup_output_dir():
    """Clean test output directories after each test."""
    yield
    output_path = get_fixtures_dir() / "e2e_test_output"
    if output_path.exists():
        shutil.rmtree(output_path)


def cleanup_directory(directory: str | Path) -> None:
    """Remove a directory and its contents."""
    directory = Path(directory)
    if directory.exists():
        shutil.rmtree(directory)


def hash_file(path: Path) -> str:
    """Compute SHA-256 hash of file contents."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_parquet_file(parquet_file: Path, validator: DataValidator) -> None:
    """Perform extended validation on a single parquet file."""
    df = pd.read_parquet(parquet_file)
    assert not df.empty, f"❌ {parquet_file.name} is empty."

    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Required columns
    for col in ["timestamp", "bid", "ask"]:
        assert col in df.columns, f"Missing column {col} in {parquet_file.name}"

    # Type checks
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"]), "timestamp must be datetime"
    assert pd.api.types.is_float_dtype(df["bid"]), "bid must be float"
    assert pd.api.types.is_float_dtype(df["ask"]), "ask must be float"

    # Logical consistency
    assert not (df["ask"] < df["bid"]).any(), f"Crossed spreads found in {parquet_file.name}"
    assert not df["timestamp"].duplicated().any(), f"Duplicate timestamps in {parquet_file.name}"

    # Schema-level validation
    is_valid, errors = validator.validate_tick_data(df)
    assert is_valid, f"Schema validation failed for {parquet_file.name}:\n" + "\n".join(errors)


def run_pipeline_with_config(config_path: Path, env_overrides: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run pipeline with deterministic environment settings."""
    env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
        "ATLASFX_SEED": "42",
    }
    if env_overrides:
        env.update(env_overrides)

    cd_repo_root()
    command = [sys.executable, "scripts/run_data_pipeline.py", "--config", str(config_path)]
    log.info(f"▶ Running pipeline: {' '.join(command)}")

    return subprocess.run(command, capture_output=True, text=True, env=env)


# ============================================================
# MAIN PIPELINE TESTS
# ============================================================

@pytest.mark.integration
class TestPipelineE2E:
    """Comprehensive integration tests for the full pipeline."""

    @pytest.mark.parametrize(
        "pairs,output_directory",
        [
            (["testusd"], "tests/fixtures/e2e_test_output_A"),
            (["eurusd"], "tests/fixtures/e2e_test_output_B"),
            (["gbpusd"], "tests/fixtures/e2e_test_output_C"),
            (["testusd", "eurusd"], "tests/fixtures/e2e_test_output_D"),
            (["testusd", "eurusd", "gbpusd"], "tests/fixtures/e2e_test_output_E"),
        ],
    )
    def test_pipeline_runs_and_validates(self, pairs: list[str], output_directory: str) -> None:
        """Run pipeline with different currency pair sets and validate outputs."""
        output_directory = resolve_path(output_directory)
        cleanup_directory(output_directory)

        # Load and customize config
        config_path = get_fixtures_dir() / "e2e_pipeline_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        pair_configs = []
        for pair in pairs:
            folder = get_fixtures_dir() / "e2e_test_data" / (pair if pair != "testusd" else "")
            pair_configs.append({"symbol": pair, "folder_path": str(folder)})

        config["merge"]["pairs"] = pair_configs
        config["output_directory"] = str(output_directory)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.safe_dump(config, tmp)
            temp_config = Path(tmp.name)

        try:
            result = run_pipeline_with_config(temp_config)
            assert result.returncode == 0, (
                f"Pipeline failed for {pairs}\n"
                f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

            parquet_files = list(output_directory.glob("*.parquet"))
            assert parquet_files, f"No parquet files generated for {pairs}"

            validator = DataValidator(schema_path="configs/schema.yaml")
            for file in parquet_files:
                validate_parquet_file(file, validator)

            # Determinism check
            hashes = {f.name: hash_file(f) for f in parquet_files}
            result2 = run_pipeline_with_config(temp_config)
            assert result2.returncode == 0, "Pipeline second run failed."
            hashes2 = {f.name: hash_file(f) for f in parquet_files}
            for k, v in hashes.items():
                assert v == hashes2[k], f"Non-deterministic output detected for {k}"

        finally:
            if temp_config.exists():
                temp_config.unlink()
            cleanup_directory(output_directory)

    def test_pipeline_output_quality(self, cleanup_output_dir) -> None:
        """Validate output quality metrics from sample pipeline run."""
        result = run_pipeline_with_config(get_fixtures_dir() / "e2e_pipeline_config.yaml")
        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"

        output_dir = get_fixtures_dir() / "e2e_test_output"
        parquet_files = list(output_dir.glob("*.parquet"))

        for file in parquet_files:
            df = pd.read_parquet(file)
            assert not df.empty, f"{file.name} is empty"
            assert (df["bid"] > 0).all(), f"Non-positive bid values in {file.name}"
            assert (df["ask"] > 0).all(), f"Non-positive ask values in {file.name}"
            assert df["timestamp"].is_monotonic_increasing, f"Timestamps not sorted in {file.name}"
            assert not (df["ask"] < df["bid"]).any(), f"Crossed spreads found in {file.name}"


# ============================================================
# CONFIGURATION VALIDATION TESTS
# ============================================================

@pytest.mark.integration
class TestPipelineConfigValidation:
    """Ensure configuration file has all required structure and fields."""

    def test_e2e_config_structure(self) -> None:
        config_path = get_fixtures_dir() / "e2e_pipeline_config.yaml"
        assert config_path.exists(), "Missing E2E pipeline config."

        with open(config_path) as f:
            config = yaml.safe_load(f)

        for key in ["steps", "merge", "output_directory"]:
            assert key in config, f"Missing '{key}' in pipeline config."

        merge_cfg = config["merge"]
        for field in ["time_column", "pairs"]:
            assert field in merge_cfg, f"Missing '{field}' in merge section."


# ============================================================
# VALIDATOR CLI TESTS
# ============================================================

@pytest.mark.integration
class TestValidatorCLI:
    """Validate that the CLI-based DataValidator behaves correctly."""

    def test_validator_cli_accepts_valid_data(self) -> None:
        """Ensure validator CLI passes for valid sample CSV."""
        result = subprocess.run(
            [
                sys.executable,
                "-m", "atlasfx.data.validators",
                "--sample", str(get_fixtures_dir() / "sample_ticks.csv"),
                "--type", "tick_data",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Validator failed: {result.stderr}"
        assert "✅ Validation passed" in result.stdout

    def test_validator_cli_rejects_invalid_data(self, tmp_path: Path) -> None:
        """Ensure validator CLI rejects crossed-spread data."""
        invalid_file = tmp_path / "invalid_ticks.csv"
        invalid_file.write_text(
            "timestamp,bid,ask,volume\n"
            "2025-01-01 00:00:00,1.0500,1.0400,100.0\n"
            "2025-01-01 00:00:01,1.0501,1.0506,150.0\n"
        )

        result = subprocess.run(
            [
                sys.executable,
                "-m", "atlasfx.data.validators",
                "--sample", str(invalid_file),
                "--type", "tick_data",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, "Validator should fail for invalid data."
        assert "❌ Validation failed" in result.stdout
