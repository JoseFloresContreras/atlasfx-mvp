"""
Tests for schema validation against configs/schema.yaml.

These tests ensure that:
1. The schema file is properly structured
2. Fixture data complies with schema requirements
3. Column types match schema definitions
4. Constraints defined in the schema are enforced
"""

import pandas as pd
import pytest
import yaml

from atlasfx.data.validators import DataValidator, ValidationError


class TestSchemaStructure:
    """Test that the schema file is properly structured."""

    def test_schema_file_exists(self) -> None:
        """Test that schema.yaml exists and can be loaded."""
        with open("configs/schema.yaml") as f:
            schema = yaml.safe_load(f)
        
        assert schema is not None
        assert isinstance(schema, dict)

    def test_schema_has_required_sections(self) -> None:
        """Test that schema has all required data type sections."""
        with open("configs/schema.yaml") as f:
            schema = yaml.safe_load(f)
        
        # Check for major data type sections
        assert "tick_data" in schema
        assert "ohlc_data" in schema
        assert "feature_matrix" in schema
        assert "pipeline_config" in schema
        assert "instruments" in schema

    def test_tick_data_schema_structure(self) -> None:
        """Test that tick_data schema has proper structure."""
        with open("configs/schema.yaml") as f:
            schema = yaml.safe_load(f)
        
        tick_data = schema["tick_data"]
        assert "description" in tick_data
        assert "required_columns" in tick_data
        assert "cross_column_constraints" in tick_data
        assert "quality_checks" in tick_data
        
        # Check required columns
        required_columns = tick_data["required_columns"]
        assert "timestamp" in required_columns
        assert "bid" in required_columns
        assert "ask" in required_columns
        assert "volume" in required_columns

    def test_ohlc_data_schema_structure(self) -> None:
        """Test that ohlc_data schema has proper structure."""
        with open("configs/schema.yaml") as f:
            schema = yaml.safe_load(f)
        
        ohlc_data = schema["ohlc_data"]
        assert "required_columns" in ohlc_data
        
        # Check OHLC columns
        required_columns = ohlc_data["required_columns"]
        assert "timestamp" in required_columns
        assert "open" in required_columns
        assert "high" in required_columns
        assert "low" in required_columns
        assert "close" in required_columns
        assert "volume" in required_columns
        assert "tick_count" in required_columns


class TestSchemaColumnTypes:
    """Test that column type definitions in schema are correct."""

    def test_tick_data_column_types(self) -> None:
        """Test that tick_data columns have correct type specifications."""
        with open("configs/schema.yaml") as f:
            schema = yaml.safe_load(f)
        
        tick_data = schema["tick_data"]["required_columns"]
        
        # Timestamp should be datetime
        assert tick_data["timestamp"]["type"] == "datetime64[ns]"
        assert tick_data["timestamp"]["nullable"] is False
        
        # Bid should be float
        assert tick_data["bid"]["type"] == "float64"
        assert tick_data["bid"]["nullable"] is False
        
        # Ask should be float
        assert tick_data["ask"]["type"] == "float64"
        assert tick_data["ask"]["nullable"] is False
        
        # Volume can be nullable
        assert tick_data["volume"]["type"] == "float64"
        assert tick_data["volume"]["nullable"] is True

    def test_ohlc_data_column_types(self) -> None:
        """Test that ohlc_data columns have correct type specifications."""
        with open("configs/schema.yaml") as f:
            schema = yaml.safe_load(f)
        
        ohlc_data = schema["ohlc_data"]["required_columns"]
        
        # All price columns should be float64
        for col in ["open", "high", "low", "close"]:
            assert ohlc_data[col]["type"] == "float64"
            assert ohlc_data[col]["nullable"] is False
        
        # tick_count should be int64
        assert ohlc_data["tick_count"]["type"] == "int64"
        assert ohlc_data["tick_count"]["nullable"] is False


class TestSchemaConstraints:
    """Test that constraints defined in schema are properly specified."""

    def test_tick_data_price_constraints(self) -> None:
        """Test that price constraints are properly defined in schema."""
        with open("configs/schema.yaml") as f:
            schema = yaml.safe_load(f)
        
        tick_data = schema["tick_data"]["required_columns"]
        
        # Bid constraints
        bid_constraints = tick_data["bid"]["constraints"]
        assert any("min_value" in c for c in bid_constraints)
        assert any("max_value" in c for c in bid_constraints)
        
        # Ask constraints
        ask_constraints = tick_data["ask"]["constraints"]
        assert any("min_value" in c for c in ask_constraints)
        assert any("max_value" in c for c in ask_constraints)

    def test_tick_data_timestamp_constraints(self) -> None:
        """Test that timestamp constraints are properly defined."""
        with open("configs/schema.yaml") as f:
            schema = yaml.safe_load(f)
        
        tick_data = schema["tick_data"]["required_columns"]
        timestamp_constraints = tick_data["timestamp"]["constraints"]
        
        # Should have monotonic_increasing constraint
        assert any("monotonic_increasing" in c for c in timestamp_constraints)
        assert any("timezone" in c for c in timestamp_constraints)

    def test_cross_column_constraints_defined(self) -> None:
        """Test that cross-column constraints are defined in schema."""
        with open("configs/schema.yaml") as f:
            schema = yaml.safe_load(f)
        
        tick_data = schema["tick_data"]
        cross_constraints = tick_data["cross_column_constraints"]
        
        # Should have no_crossed_spreads constraint
        constraint_names = [c["name"] for c in cross_constraints]
        assert "no_crossed_spreads" in constraint_names
        assert "reasonable_spread" in constraint_names

    def test_quality_checks_defined(self) -> None:
        """Test that quality checks are defined in schema."""
        with open("configs/schema.yaml") as f:
            schema = yaml.safe_load(f)
        
        tick_data = schema["tick_data"]
        quality_checks = tick_data["quality_checks"]
        
        check_names = [c["name"] for c in quality_checks]
        assert "no_duplicates" in check_names


class TestFixtureDataCompliance:
    """Test that fixture data complies with schema requirements."""

    def test_sample_ticks_complies_with_schema(self) -> None:
        """Test that sample_ticks.csv complies with tick_data schema."""
        df = pd.read_csv("tests/fixtures/sample_ticks.csv", parse_dates=["timestamp"])
        validator = DataValidator()
        
        is_valid, errors = validator.validate_tick_data(df)
        
        assert is_valid, f"Validation errors: {errors}"

    def test_sample_ticks_has_required_columns(self) -> None:
        """Test that sample_ticks.csv has all required columns."""
        df = pd.read_csv("tests/fixtures/sample_ticks.csv")
        
        # Check required columns from schema
        required_columns = ["timestamp", "bid", "ask", "volume"]
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

    def test_sample_ticks_column_types(self) -> None:
        """Test that sample_ticks.csv has correct column types."""
        df = pd.read_csv("tests/fixtures/sample_ticks.csv", parse_dates=["timestamp"])
        
        # Check types match schema
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert pd.api.types.is_float_dtype(df["bid"])
        assert pd.api.types.is_float_dtype(df["ask"])
        assert pd.api.types.is_float_dtype(df["volume"])

    def test_sample_ticks_price_constraints(self) -> None:
        """Test that sample_ticks.csv satisfies price constraints."""
        df = pd.read_csv("tests/fixtures/sample_ticks.csv")
        
        # Prices should be positive
        assert (df["bid"] > 0).all(), "Bid prices must be positive"
        assert (df["ask"] > 0).all(), "Ask prices must be positive"
        
        # Prices should be reasonable (< 1M as per schema)
        assert (df["bid"] < 1e6).all(), "Bid prices exceed max constraint"
        assert (df["ask"] < 1e6).all(), "Ask prices exceed max constraint"

    def test_sample_ticks_no_crossed_spreads(self) -> None:
        """Test that sample_ticks.csv has no crossed spreads."""
        df = pd.read_csv("tests/fixtures/sample_ticks.csv")
        
        # Ask should be >= Bid
        assert (df["ask"] >= df["bid"]).all(), "Found crossed spreads (ask < bid)"

    def test_sample_ticks_monotonic_timestamps(self) -> None:
        """Test that sample_ticks.csv has monotonically increasing timestamps."""
        df = pd.read_csv("tests/fixtures/sample_ticks.csv", parse_dates=["timestamp"])
        
        assert df["timestamp"].is_monotonic_increasing, "Timestamps not monotonic increasing"

    def test_testusd_tick_data_complies_with_schema(self) -> None:
        """Test that testusd_tick_data.csv complies with tick_data schema."""
        df = pd.read_csv(
            "tests/fixtures/e2e_test_data/testusd_tick_data.csv",
            parse_dates=["timestamp"]
        )
        validator = DataValidator()
        
        is_valid, errors = validator.validate_tick_data(df)
        
        assert is_valid, f"Validation errors: {errors}"

    def test_testusd_tick_data_has_required_columns(self) -> None:
        """Test that testusd_tick_data.csv has all required columns."""
        df = pd.read_csv("tests/fixtures/e2e_test_data/testusd_tick_data.csv")
        
        required_columns = ["timestamp", "bid", "ask", "volume"]
        for col in required_columns:
            assert col in df.columns, f"Missing required column: {col}"

    def test_testusd_tick_data_column_types(self) -> None:
        """Test that testusd_tick_data.csv has correct column types."""
        df = pd.read_csv(
            "tests/fixtures/e2e_test_data/testusd_tick_data.csv",
            parse_dates=["timestamp"]
        )
        
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])
        assert pd.api.types.is_float_dtype(df["bid"])
        assert pd.api.types.is_float_dtype(df["ask"])
        assert pd.api.types.is_float_dtype(df["volume"])

    def test_testusd_tick_data_no_crossed_spreads(self) -> None:
        """Test that testusd_tick_data.csv has no crossed spreads."""
        df = pd.read_csv("tests/fixtures/e2e_test_data/testusd_tick_data.csv")
        
        assert (df["ask"] >= df["bid"]).all(), "Found crossed spreads (ask < bid)"


class TestSchemaValidationIntegration:
    """Test integration between schema and validator."""

    def test_validator_loads_schema_correctly(self) -> None:
        """Test that DataValidator loads schema correctly."""
        validator = DataValidator(schema_path="configs/schema.yaml")
        
        assert validator.schema is not None
        assert "tick_data" in validator.schema
        assert "ohlc_data" in validator.schema
        assert "feature_matrix" in validator.schema

    def test_validator_enforces_column_requirements(self) -> None:
        """Test that validator enforces required columns from schema."""
        validator = DataValidator()
        
        # Missing required column
        df_missing_bid = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="1s"),
            "ask": [1.0505, 1.0506, 1.0507, 1.0508, 1.0509],
            "volume": [100.0, 150.0, 200.0, 180.0, 220.0],
        })
        
        is_valid, errors = validator.validate_tick_data(df_missing_bid)
        assert not is_valid
        assert any("bid" in error for error in errors)

    def test_validator_enforces_type_constraints(self) -> None:
        """Test that validator enforces type constraints from schema."""
        validator = DataValidator()
        
        # Wrong type for timestamp (string instead of datetime)
        df_wrong_type = pd.DataFrame({
            "timestamp": ["2025-01-01 00:00:00"] * 5,  # String, not datetime
            "bid": [1.0500, 1.0501, 1.0502, 1.0503, 1.0504],
            "ask": [1.0505, 1.0506, 1.0507, 1.0508, 1.0509],
            "volume": [100.0, 150.0, 200.0, 180.0, 220.0],
        })
        
        is_valid, errors = validator.validate_tick_data(df_wrong_type)
        assert not is_valid
        assert any("timestamp" in error for error in errors)

    def test_validator_enforces_value_constraints(self) -> None:
        """Test that validator enforces value constraints from schema."""
        validator = DataValidator()
        
        # Negative price (violates min_value constraint)
        df_negative_price = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="1s"),
            "bid": [-1.0, 1.0501, 1.0502, 1.0503, 1.0504],
            "ask": [1.0505, 1.0506, 1.0507, 1.0508, 1.0509],
            "volume": [100.0, 150.0, 200.0, 180.0, 220.0],
        })
        
        is_valid, errors = validator.validate_tick_data(df_negative_price)
        assert not is_valid
        assert any("bid" in error and "0.0" in error for error in errors)

    def test_validator_enforces_cross_column_constraints(self) -> None:
        """Test that validator enforces cross-column constraints from schema."""
        validator = DataValidator()
        
        # Crossed spread (ask < bid)
        df_crossed_spread = pd.DataFrame({
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="1s"),
            "bid": [1.0505, 1.0506, 1.0507, 1.0508, 1.0509],
            "ask": [1.0500, 1.0501, 1.0502, 1.0503, 1.0504],  # Ask < Bid
            "volume": [100.0, 150.0, 200.0, 180.0, 220.0],
        })
        
        is_valid, errors = validator.validate_tick_data(df_crossed_spread)
        assert not is_valid
        assert any("crossed" in error.lower() for error in errors)

    def test_validator_enforces_monotonic_constraint(self) -> None:
        """Test that validator enforces monotonic_increasing constraint from schema."""
        validator = DataValidator()
        
        # Non-monotonic timestamps (going backward)
        timestamps = pd.date_range("2025-01-01", periods=5, freq="1s").tolist()
        timestamps[2] = timestamps[0]  # Make timestamp go backward
        
        df_non_monotonic = pd.DataFrame({
            "timestamp": timestamps,
            "bid": [1.0500, 1.0501, 1.0502, 1.0503, 1.0504],
            "ask": [1.0505, 1.0506, 1.0507, 1.0508, 1.0509],
            "volume": [100.0, 150.0, 200.0, 180.0, 220.0],
        })
        
        is_valid, errors = validator.validate_tick_data(df_non_monotonic)
        assert not is_valid
        assert any("monotonic" in error.lower() for error in errors)
