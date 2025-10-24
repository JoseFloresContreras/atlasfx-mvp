"""End-to-End integration tests for the data pipeline.

This module tests the full pipeline execution from raw tick data through
aggregation, ensuring outputs conform to schema at each stage.
"""

import pathlibLFLFimport """CreateLFimport ->LFimport DataValidatorLFLFLF@pytest.fixtureLFdefLFimport directoryLFimport forLFimport importLFimport osLFimportLFimport outputs."""LFLFimport pandas as pdLFimportLFimport PathLFimport Path:LFLFimport PathLFLFimportLFimport pipelineLFimport pipeline_output_dirLFimport pytestLFLFfromLFimport shutilLFimportLFimport sysLFfromLFimport temporaryLFimport tmp_path:LFLFimport atlasfx.data.validatorsLF    output_dir = tmp_path / "e2e_test_output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def cleanup_output_dir():
    """Cleanup fixture to remove test output directory after tests."""
    yield
    # Cleanup after test
    if os.path.exists("tests/fixtures/e2e_test_output"):
        shutil.rmtree("tests/fixtures/e2e_test_output")


def cleanup_directory(output_dir: str) -> None:
    """Clean up output directory before running tests."""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)


def validate_parquet_file(parquet_file: Path, validator: DataValidator) -> None:
    """
    Validate a single parquet file with extended checks.

    Validates:
    - Required columns (timestamp, bid, ask, volume)
    - Correct types (datetime, float)
    - ask >= bid always
    - No duplicates in timestamp
    - DataFrame not empty

    Args:
        parquet_file: Path to parquet file to validate
        validator: DataValidator instance

    Raises:
        AssertionError: If any validation check fails
    """
    df = pd.read_parquet(parquet_file)

    # Check 1: DataFrame not empty
    assert len(df) > 0, f"Output file {parquet_file.name} is empty"

    # Convert timestamp to datetime if needed (parquet may store as string)
    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Check 2: Required columns present
    required_columns = ["timestamp", "bid", "ask"]
    for col in required_columns:
        assert col in df.columns, f"Missing required column {col} in {parquet_file.name}"

    # Check 3: Correct types
    assert pd.api.types.is_datetime64_any_dtype(
        df["timestamp"]
    ), f"timestamp must be datetime type in {parquet_file.name}"
    assert pd.api.types.is_float_dtype(df["bid"]), f"bid must be float type in {parquet_file.name}"
    assert pd.api.types.is_float_dtype(df["ask"]), f"ask must be float type in {parquet_file.name}"

    # Check 4: ask >= bid always
    crossed = df["ask"] < df["bid"]
    assert (
        not crossed.any()
    ), f"Found {crossed.sum()} crossed spreads (ask < bid) in {parquet_file.name}"

    # Check 5: No duplicates in timestamp
    duplicates = df["timestamp"].duplicated().sum()
    assert duplicates == 0, f"Found {duplicates} duplicate timestamps in {parquet_file.name}"

    # Check 6: Use DataValidator for comprehensive validation
    is_valid, errors = validator.validate_tick_data(df)
    assert is_valid, f"Pipeline output {parquet_file.name} failed validation:\n" + "\n".join(
        f"  - {error}" for error in errors
    )


@pytest.mark.integration
class TestPipelineE2E:
    """End-to-end tests for the data pipeline."""

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
    def test_pipeline_with_multiple_pairs(self, pairs: list[str], output_directory: str) -> None:
        """
        Test pipeline execution with different pairs and output directories.

        This parametrized test runs the pipeline with:
        - Single pairs (testusd, eurusd, gbpusd)
        - Multiple pair combinations
        - Different output directories

        For each configuration, validates:
        - Pipeline runs successfully
        - Output files are created
        - All parquet files pass extended validation
        """
        # Clean up output directory before running
        cleanup_directory(output_directory)

        # Create a temporary config file for this testLFLF
        import tempfileLFLFimportLFimport yamlLFLF

        config_path = "tests/fixtures/e2e_pipeline_config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Update config with specific pairs and output directory
        pair_configs = []
        for pair in pairs:
            if pair == "testusd":
                pair_configs.append(
                    {"symbol": "testusd", "folder_path": "tests/fixtures/e2e_test_data"}
                )
            elif pair == "eurusd":
                pair_configs.append(
                    {"symbol": "eurusd", "folder_path": "tests/fixtures/e2e_test_data/eurusd"}
                )
            elif pair == "gbpusd":
                pair_configs.append(
                    {"symbol": "gbpusd", "folder_path": "tests/fixtures/e2e_test_data/gbpusd"}
                )

        config["merge"]["pairs"] = pair_configs
        config["output_directory"] = output_directory

        # Write temporary config
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.safe_dump(config, f)
            temp_config_path = f.name

        try:
            # Run the pipelineLFLF
            import subprocessLFLF

            env = {
                **os.environ,
                "PYTHONIOENCODING": "utf-8",
                "PYTHONHASHSEED": "0",
                "TZ": "UTC",
                "ATLASFX_SEED": "42",
            }

            # Opcional: asegurar que random y numpy usen la misma semilla
            try:
                import numpy as npLFLFimport randomLFLFimportLF

                random.seed(42)
                np.random.seed(42)
            except Exception:
                pass

            result = subprocess.run(
                [sys.executable, "scripts/run_data_pipeline.py", temp_config_path],
                capture_output=True,
                text=True,
                env=env,
            )

            # Check that pipeline ran successfully
            assert result.returncode == 0, (
                f"Pipeline failed for pairs {pairs}:\n"
                f"STDOUT: {result.stdout}\n"
                f"STDERR: {result.stderr}"
            )

            # Check that output directory was created
            output_dir = Path(output_directory)
            assert output_dir.exists(), f"Output directory {output_directory} was not created"

            # Find generated parquet files
            parquet_files = list(output_dir.glob("*.parquet"))
            assert len(parquet_files) > 0, f"No parquet files were generated for pairs {pairs}"

            # Validate each parquet file with extended validation
            validator = DataValidator(schema_path="configs/schema.yaml")

            for parquet_file in parquet_files:
                validate_parquet_file(parquet_file, validator)

        finally:
            # Clean up temporary config file
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)

            # Clean up output directory
            cleanup_directory(output_directory)

    def test_pipeline_merge_step(self, cleanup_output_dir) -> None:
        """Test that pipeline merge step produces valid output."""
        # Run the pipeline with e2e config (only merge step)LFLF
        import subprocessLFLF

        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_data_pipeline.py",
                "tests/fixtures/e2e_pipeline_config.yaml",
            ],
            capture_output=True,
            text=True,
        )

        # Check that pipeline ran successfully
        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"

        # Check that output file was created
        output_dir = Path("tests/fixtures/e2e_test_output")
        assert output_dir.exists(), "Output directory was not created"

        # Find generated parquet files
        parquet_files = list(output_dir.glob("*.parquet"))
        assert len(parquet_files) > 0, "No parquet files were generated"

        # Validate the output against tick_data schema
        validator = DataValidator(schema_path="configs/schema.yaml")

        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)

            # Convert timestamp to datetime if needed (parquet may store as string)
            if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(
                df["timestamp"]
            ):
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Check that data is not empty
            assert len(df) > 0, f"Output file {parquet_file.name} is empty"

            # Validate schema
            is_valid, errors = validator.validate_tick_data(df)
            assert is_valid, f"Pipeline output {parquet_file.name} failed validation: {errors}"

    def test_pipeline_output_structure(self, cleanup_output_dir) -> None:
        """Test that pipeline outputs have correct structure."""
        # Run the pipelineLFLF
        import subprocessLFLF

        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_data_pipeline.py",
                "tests/fixtures/e2e_pipeline_config.yaml",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"

        # Check output file structure
        output_dir = Path("tests/fixtures/e2e_test_output")
        parquet_files = list(output_dir.glob("*.parquet"))

        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)

            # Convert timestamp to datetime if needed
            if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(
                df["timestamp"]
            ):
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Check required columns for tick data
            required_columns = ["timestamp", "bid", "ask"]
            for col in required_columns:
                assert col in df.columns, f"Missing column {col} in {parquet_file.name}"

            # Check data types
            assert pd.api.types.is_datetime64_any_dtype(
                df["timestamp"]
            ), "timestamp must be datetime"
            assert pd.api.types.is_float_dtype(df["bid"]), "bid must be float"
            assert pd.api.types.is_float_dtype(df["ask"]), "ask must be float"

    def test_pipeline_output_data_quality(self, cleanup_output_dir) -> None:
        """Test that pipeline outputs meet data quality requirements."""
        # Run the pipelineLFLF
        import subprocessLFLF

        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_data_pipeline.py",
                "tests/fixtures/e2e_pipeline_config.yaml",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"

        # Check data quality
        output_dir = Path("tests/fixtures/e2e_test_output")
        parquet_files = list(output_dir.glob("*.parquet"))

        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)

            # Convert timestamp to datetime if needed
            if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(
                df["timestamp"]
            ):
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Check no crossed spreads
            if "bid" in df.columns and "ask" in df.columns:
                crossed = df["ask"] < df["bid"]
                assert not crossed.any(), f"Found crossed spreads in {parquet_file.name}"

            # Check monotonic timestamps
            if "timestamp" in df.columns:
                assert df[
                    "timestamp"
                ].is_monotonic_increasing, f"Timestamps not monotonic in {parquet_file.name}"

            # Check positive prices
            if "bid" in df.columns:
                assert (df["bid"] > 0).all(), f"Non-positive bid prices in {parquet_file.name}"
            if "ask" in df.columns:
                assert (df["ask"] > 0).all(), f"Non-positive ask prices in {parquet_file.name}"

    def test_parquet_files_readable(self, cleanup_output_dir) -> None:
        """Test that generated parquet files are readable and well-formed."""
        # Run the pipelineLFLF
        import subprocessLFLF

        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_data_pipeline.py",
                "tests/fixtures/e2e_pipeline_config.yaml",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Pipeline failed: {result.stderr}"

        # Try to read each parquet file
        output_dir = Path("tests/fixtures/e2e_test_output")
        parquet_files = list(output_dir.glob("*.parquet"))

        assert len(parquet_files) > 0, "No parquet files generated"

        for parquet_file in parquet_files:
            # Should not raise
            df = pd.read_parquet(parquet_file)
            assert df is not None, f"Failed to read {parquet_file.name}"
            assert len(df) > 0, f"Empty dataframe in {parquet_file.name}"


@pytest.mark.integration
class TestPipelineConfigValidation:
    """Test pipeline configuration validation."""

    def test_e2e_config_is_valid(self) -> None:
        """Test that e2e_pipeline_config.yaml is valid."""
        import yamlLFLF

        config_path = "tests/fixtures/e2e_pipeline_config.yaml"
        assert os.path.exists(config_path), "E2E pipeline config not found"

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Check required fields
        assert "pipeline" in config, "Missing 'pipeline' section"
        assert "steps" in config, "Missing 'steps' section"
        assert "output_directory" in config, "Missing 'output_directory'"
        assert "merge" in config, "Missing 'merge' section"

        # Check merge configuration
        merge_config = config["merge"]
        assert "time_column" in merge_config, "Missing 'time_column' in merge config"
        assert "pairs" in merge_config, "Missing 'pairs' in merge config"


@pytest.mark.integration
class TestValidatorCLI:
    """Test the validator CLI functionality."""

    def test_validator_cli_with_sample_ticks(self) -> None:
        """Test validator CLI with sample ticks."""
        import subprocessLFLF

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "atlasfx.data.validators",
                "--sample",
                "tests/fixtures/sample_ticks.csv",
                "--type",
                "tick_data",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Validator CLI failed: {result.stderr}"
        assert "✅ Validation passed" in result.stdout, "Validation did not pass"

    def test_validator_cli_with_e2e_test_data(self) -> None:
        """Test validator CLI with e2e test data."""
        import subprocessLFLF

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "atlasfx.data.validators",
                "--sample",
                "tests/fixtures/e2e_test_data/testusd_tick_data.csv",
                "--type",
                "tick_data",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Validator CLI failed: {result.stderr}"
        assert "✅ Validation passed" in result.stdout, "Validation did not pass"

    def test_validator_cli_rejects_invalid_data(self, tmp_path: Path) -> None:
        """Test validator CLI rejects invalid data."""
        # Create invalid test data (crossed spread)
        invalid_file = tmp_path / "invalid_ticks.csv"
        invalid_data = """timestamp,bid,ask,volume
2025-01-01 00:00:00,1.0500,1.0400,100.0
2025-01-01 00:00:01,1.0501,1.0506,150.0
"""
        invalid_file.write_text(invalid_data)

        import subprocessLFLF

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "atlasfx.data.validators",
                "--sample",
                str(invalid_file),
                "--type",
                "tick_data",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0, "Validator should fail on invalid data"
        assert "❌ Validation failed" in result.stdout, "Expected validation failure message"
