"""
Tests for data validators.
"""

import numpy as np
import pandas as pd
import pytest

from atlasfx.data.validators import (
    DataValidator,
    ValidationError,
    validate_dataframe,
)


@pytest.fixture
def valid_tick_data() -> pd.DataFrame:
    """Create valid tick data for testing."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="1s"),
            "bid": [1.0500, 1.0501, 1.0502, 1.0503, 1.0504],
            "ask": [1.0505, 1.0506, 1.0507, 1.0508, 1.0509],
            "volume": [100.0, 150.0, 200.0, 180.0, 220.0],
        }
    )


@pytest.fixture
def valid_ohlc_data() -> pd.DataFrame:
    """Create valid OHLC data for testing."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="1min"),
            "open": [1.0500, 1.0505, 1.0510, 1.0508, 1.0512],
            "high": [1.0510, 1.0515, 1.0520, 1.0518, 1.0522],
            "low": [1.0495, 1.0500, 1.0505, 1.0503, 1.0507],
            "close": [1.0505, 1.0510, 1.0508, 1.0512, 1.0520],
            "volume": [1000.0, 1500.0, 2000.0, 1800.0, 2200.0],
            "tick_count": [10, 15, 20, 18, 22],
        }
    )


@pytest.fixture
def valid_feature_matrix() -> pd.DataFrame:
    """Create valid feature matrix for testing."""
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=5, freq="1min"),
            "mid_price": [1.05025, 1.05055, 1.05085, 1.05055, 1.05115],
            "returns": [0.0, 0.0003, 0.0003, -0.0003, 0.0006],
            "spread": [0.0005, 0.0005, 0.0005, 0.0005, 0.0005],
            "volume": [1000.0, 1500.0, 2000.0, 1800.0, 2200.0],
            "vwap": [1.05020, 1.05050, 1.05080, 1.05050, 1.05110],
            "ofi": [0.1, -0.2, 0.3, -0.1, 0.2],
            "micro_price": [1.05023, 1.05053, 1.05083, 1.05053, 1.05113],
        }
    )


class TestDataValidator:
    """Test suite for DataValidator class."""

    def test_validator_initialization(self) -> None:
        """Test validator can be initialized with schema."""
        validator = DataValidator(schema_path="configs/schema.yaml")
        assert validator.schema is not None
        assert "tick_data" in validator.schema

    def test_validate_tick_data_valid(self, valid_tick_data: pd.DataFrame) -> None:
        """Test validation passes for valid tick data."""
        validator = DataValidator(schema_path="configs/schema.yaml")
        is_valid, errors = validator.validate_tick_data(valid_tick_data)

        assert is_valid
        assert len(errors) == 0

    def test_validate_tick_data_missing_column(self, valid_tick_data: pd.DataFrame) -> None:
        """Test validation fails when required column is missing."""
        validator = DataValidator(schema_path="configs/schema.yaml")
        df = valid_tick_data.drop(columns=["bid"])

        is_valid, errors = validator.validate_tick_data(df)

        assert not is_valid
        assert any("bid" in error for error in errors)

    def test_validate_tick_data_negative_price(self, valid_tick_data: pd.DataFrame) -> None:
        """Test validation fails for negative prices."""
        validator = DataValidator(schema_path="configs/schema.yaml")
        df = valid_tick_data.copy()
        df.loc[0, "bid"] = -1.0

        is_valid, errors = validator.validate_tick_data(df)

        assert not is_valid
        assert any("bid" in error and "< 0.0" in error for error in errors)

    def test_validate_tick_data_crossed_spread(self, valid_tick_data: pd.DataFrame) -> None:
        """Test validation detects crossed spreads (ask < bid)."""
        validator = DataValidator(schema_path="configs/schema.yaml")
        df = valid_tick_data.copy()
        df.loc[0, "ask"] = 1.0400  # ask < bid

        is_valid, errors = validator.validate_tick_data(df)

        assert not is_valid
        assert any("crossed" in error.lower() for error in errors)

    def test_validate_tick_data_non_monotonic_timestamp(
        self, valid_tick_data: pd.DataFrame
    ) -> None:
        """Test validation detects non-monotonic timestamps."""
        validator = DataValidator(schema_path="configs/schema.yaml")
        df = valid_tick_data.copy()
        # Make timestamp go backwards to break monotonicity
        df.loc[1, "timestamp"] = df.loc[0, "timestamp"] - pd.Timedelta(seconds=1)

        is_valid, errors = validator.validate_tick_data(df)

        assert not is_valid
        assert any("monotonic" in error.lower() for error in errors)

    def test_validate_ohlc_data_valid(self, valid_ohlc_data: pd.DataFrame) -> None:
        """Test validation passes for valid OHLC data."""
        validator = DataValidator(schema_path="configs/schema.yaml")
        is_valid, errors = validator.validate_ohlc_data(valid_ohlc_data)

        assert is_valid
        assert len(errors) == 0

    def test_validate_ohlc_data_high_less_than_low(
        self, valid_ohlc_data: pd.DataFrame
    ) -> None:
        """Test validation detects high < low."""
        validator = DataValidator(schema_path="configs/schema.yaml")
        df = valid_ohlc_data.copy()
        df.loc[0, "high"] = 1.0490  # high < low

        is_valid, errors = validator.validate_ohlc_data(df)

        assert not is_valid
        assert any("high < low" in error for error in errors)

    def test_validate_ohlc_data_high_less_than_open(
        self, valid_ohlc_data: pd.DataFrame
    ) -> None:
        """Test validation detects high < open."""
        validator = DataValidator(schema_path="configs/schema.yaml")
        df = valid_ohlc_data.copy()
        df.loc[0, "high"] = 1.0495  # high < open

        is_valid, errors = validator.validate_ohlc_data(df)

        assert not is_valid
        assert any("high < open" in error for error in errors)

    def test_validate_ohlc_data_negative_price(self, valid_ohlc_data: pd.DataFrame) -> None:
        """Test validation detects negative prices."""
        validator = DataValidator(schema_path="configs/schema.yaml")
        df = valid_ohlc_data.copy()
        df.loc[0, "close"] = -1.0

        is_valid, errors = validator.validate_ohlc_data(df)

        assert not is_valid
        assert any("non-positive" in error for error in errors)

    def test_validate_feature_matrix_valid(self, valid_feature_matrix: pd.DataFrame) -> None:
        """Test validation passes for valid feature matrix."""
        validator = DataValidator(schema_path="configs/schema.yaml")
        is_valid, errors = validator.validate_feature_matrix(valid_feature_matrix)

        assert is_valid
        assert len(errors) == 0

    def test_validate_feature_matrix_infinite_values(
        self, valid_feature_matrix: pd.DataFrame
    ) -> None:
        """Test validation detects infinite values."""
        validator = DataValidator(schema_path="configs/schema.yaml")
        df = valid_feature_matrix.copy()
        df.loc[0, "returns"] = np.inf

        is_valid, errors = validator.validate_feature_matrix(df)

        assert not is_valid
        assert any("infinite" in error.lower() for error in errors)

    def test_validate_feature_matrix_negative_spread(
        self, valid_feature_matrix: pd.DataFrame
    ) -> None:
        """Test validation detects negative spread."""
        validator = DataValidator(schema_path="configs/schema.yaml")
        df = valid_feature_matrix.copy()
        df.loc[0, "spread"] = -0.0005

        is_valid, errors = validator.validate_feature_matrix(df)

        assert not is_valid
        assert any("negative spread" in error.lower() for error in errors)

    def test_validate_feature_matrix_negative_volume(
        self, valid_feature_matrix: pd.DataFrame
    ) -> None:
        """Test validation detects negative volume."""
        validator = DataValidator(schema_path="configs/schema.yaml")
        df = valid_feature_matrix.copy()
        df.loc[0, "volume"] = -100.0

        is_valid, errors = validator.validate_feature_matrix(df)

        assert not is_valid
        assert any("negative volume" in error.lower() for error in errors)


class TestValidateDataFrameFunction:
    """Test suite for validate_dataframe function."""

    def test_validate_dataframe_tick_data_valid(self, valid_tick_data: pd.DataFrame) -> None:
        """Test validate_dataframe passes for valid tick data."""
        # Should not raise
        validate_dataframe(valid_tick_data, data_type="tick_data")

    def test_validate_dataframe_tick_data_invalid(
        self, valid_tick_data: pd.DataFrame
    ) -> None:
        """Test validate_dataframe raises for invalid tick data."""
        df = valid_tick_data.copy()
        df.loc[0, "bid"] = -1.0

        with pytest.raises(ValidationError):
            validate_dataframe(df, data_type="tick_data")

    def test_validate_dataframe_ohlc_data_valid(self, valid_ohlc_data: pd.DataFrame) -> None:
        """Test validate_dataframe passes for valid OHLC data."""
        # Should not raise
        validate_dataframe(valid_ohlc_data, data_type="ohlc_data")

    def test_validate_dataframe_feature_matrix_valid(
        self, valid_feature_matrix: pd.DataFrame
    ) -> None:
        """Test validate_dataframe passes for valid feature matrix."""
        # Should not raise
        validate_dataframe(valid_feature_matrix, data_type="feature_matrix")

    def test_validate_dataframe_unknown_type(self, valid_tick_data: pd.DataFrame) -> None:
        """Test validate_dataframe raises for unknown data type."""
        with pytest.raises(ValueError, match="Unknown data type"):
            validate_dataframe(valid_tick_data, data_type="unknown_type")
