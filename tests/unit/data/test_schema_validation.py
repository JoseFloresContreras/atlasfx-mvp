"""
Tests for schema validation of pipeline inputs and outputs.

This module validates that data flowing through the pipeline conforms to
the schema defined in configs/schema.yaml at each stage.
"""

import """LoadLFimport ->LFimport dataLFimport DataValidatorLFimport fixtures."""LFLFimport importLFimport pandas as pdLFimportLFimport pd.DataFrame:LFLFimport pytestLFLFfromLFimport sampleLFimport sample_tick_dataLFimport tickLFimport validate_dataframeLFLFLF@pytest.fixtureLFdefLFimport ValidationErrorLFLFimport atlasfx.data.validatorsLF    return pd.read_csv("tests/fixtures/sample_ticks.csv", parse_dates=["timestamp"])


@pytest.fixture
def e2e_tick_data() -> pd.DataFrame:
    """Load e2e tick data from fixtures."""
    return pd.read_csv(
        "tests/fixtures/e2e_test_data/testusd_tick_data.csv", parse_dates=["timestamp"]
    )


class TestTickDataSchemaValidation:
    """Test schema validation for tick data (pipeline input)."""

    def test_sample_ticks_conform_to_schema(self, sample_tick_data: pd.DataFrame) -> None:
        """Test that sample_ticks.csv conforms to tick_data schema."""
        validator = DataValidator(schema_path="configs/schema.yaml")
        is_valid, errors = validator.validate_tick_data(sample_tick_data)

        assert is_valid, f"Sample tick data failed validation: {errors}"
        assert len(errors) == 0

    def test_e2e_tick_data_conform_to_schema(self, e2e_tick_data: pd.DataFrame) -> None:
        """Test that e2e tick data conforms to tick_data schema."""
        validator = DataValidator(schema_path="configs/schema.yaml")
        is_valid, errors = validator.validate_tick_data(e2e_tick_data)

        assert is_valid, f"E2E tick data failed validation: {errors}"
        assert len(errors) == 0

    def test_tick_data_required_columns(self, sample_tick_data: pd.DataFrame) -> None:
        """Test that tick data has all required columns."""
        required_columns = ["timestamp", "bid", "ask", "volume"]
        for col in required_columns:
            assert col in sample_tick_data.columns, f"Missing required column: {col}"

    def test_tick_data_types(self, sample_tick_data: pd.DataFrame) -> None:
        """Test that tick data columns have correct types."""
        assert pd.api.types.is_datetime64_any_dtype(
            sample_tick_data["timestamp"]
        ), "timestamp must be datetime"
        assert pd.api.types.is_float_dtype(sample_tick_data["bid"]), "bid must be float"
        assert pd.api.types.is_float_dtype(sample_tick_data["ask"]), "ask must be float"
        assert pd.api.types.is_float_dtype(sample_tick_data["volume"]), "volume must be float"

    def test_tick_data_no_crossed_spreads(self, sample_tick_data: pd.DataFrame) -> None:
        """Test that ask >= bid (no crossed spreads)."""
        crossed = sample_tick_data["ask"] < sample_tick_data["bid"]
        assert not crossed.any(), f"Found {crossed.sum()} crossed spreads"

    def test_tick_data_positive_prices(self, sample_tick_data: pd.DataFrame) -> None:
        """Test that bid and ask prices are positive."""
        assert (sample_tick_data["bid"] > 0).all(), "All bid prices must be positive"
        assert (sample_tick_data["ask"] > 0).all(), "All ask prices must be positive"

    def test_tick_data_monotonic_timestamps(self, sample_tick_data: pd.DataFrame) -> None:
        """Test that timestamps are monotonically increasing."""
        assert sample_tick_data[
            "timestamp"
        ].is_monotonic_increasing, "Timestamps must be monotonically increasing"


class TestOHLCDataSchemaValidation:
    """Test schema validation for OHLC data (pipeline intermediate output)."""

    @pytest.fixture
    def sample_ohlc_data(self) -> pd.DataFrame:
        """Create sample OHLC data for testing."""
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

    def test_ohlc_data_conform_to_schema(self, sample_ohlc_data: pd.DataFrame) -> None:
        """Test that OHLC data conforms to ohlc_data schema."""
        validator = DataValidator(schema_path="configs/schema.yaml")
        is_valid, errors = validator.validate_ohlc_data(sample_ohlc_data)

        assert is_valid, f"OHLC data failed validation: {errors}"
        assert len(errors) == 0

    def test_ohlc_required_columns(self, sample_ohlc_data: pd.DataFrame) -> None:
        """Test that OHLC data has all required columns."""
        required_columns = ["timestamp", "open", "high", "low", "close", "volume", "tick_count"]
        for col in required_columns:
            assert col in sample_ohlc_data.columns, f"Missing required column: {col}"

    def test_ohlc_relationships(self, sample_ohlc_data: pd.DataFrame) -> None:
        """Test OHLC relationships (high >= low, high >= open/close, etc.)."""
        # High >= Low
        assert (sample_ohlc_data["high"] >= sample_ohlc_data["low"]).all(), "High must be >= Low"

        # High >= Open
        assert (sample_ohlc_data["high"] >= sample_ohlc_data["open"]).all(), "High must be >= Open"

        # High >= Close
        assert (
            sample_ohlc_data["high"] >= sample_ohlc_data["close"]
        ).all(), "High must be >= Close"

        # Low <= Open
        assert (sample_ohlc_data["low"] <= sample_ohlc_data["open"]).all(), "Low must be <= Open"

        # Low <= Close
        assert (sample_ohlc_data["low"] <= sample_ohlc_data["close"]).all(), "Low must be <= Close"


class TestFeatureMatrixSchemaValidation:
    """Test schema validation for feature matrix (pipeline final output)."""

    @pytest.fixture
    def sample_feature_matrix(self) -> pd.DataFrame:
        """Create sample feature matrix for testing."""
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

    def test_feature_matrix_conform_to_schema(self, sample_feature_matrix: pd.DataFrame) -> None:
        """Test that feature matrix conforms to feature_matrix schema."""
        validator = DataValidator(schema_path="configs/schema.yaml")
        is_valid, errors = validator.validate_feature_matrix(sample_feature_matrix)

        assert is_valid, f"Feature matrix failed validation: {errors}"
        assert len(errors) == 0

    def test_feature_matrix_no_infinite_values(self, sample_feature_matrix: pd.DataFrame) -> None:
        """Test that feature matrix has no infinite values."""
        import numpy as npLFLF

        for col in sample_feature_matrix.select_dtypes(include=[np.number]).columns:
            assert not np.isinf(
                sample_feature_matrix[col]
            ).any(), f"Column {col} contains infinite values"

    def test_feature_matrix_non_negative_constraints(
        self, sample_feature_matrix: pd.DataFrame
    ) -> None:
        """Test that spread and volume are non-negative."""
        if "spread" in sample_feature_matrix.columns:
            assert (sample_feature_matrix["spread"] >= 0).all(), "Spread must be non-negative"

        if "volume" in sample_feature_matrix.columns:
            assert (sample_feature_matrix["volume"] >= 0).all(), "Volume must be non-negative"


class TestPipelineSchemaIntegration:
    """Test schema validation across pipeline stages."""

    def test_validate_dataframe_function_tick_data(self, sample_tick_data: pd.DataFrame) -> None:
        """Test validate_dataframe function with tick data."""
        # Should not raise
        validate_dataframe(sample_tick_data, data_type="tick_data")

    def test_validate_dataframe_function_ohlc_data(self) -> None:
        """Test validate_dataframe function with OHLC data."""
        ohlc_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=3, freq="1min"),
                "open": [1.0500, 1.0505, 1.0510],
                "high": [1.0510, 1.0515, 1.0520],
                "low": [1.0495, 1.0500, 1.0505],
                "close": [1.0505, 1.0510, 1.0515],
                "volume": [1000.0, 1500.0, 2000.0],
                "tick_count": [10, 15, 20],
            }
        )
        # Should not raise
        validate_dataframe(ohlc_data, data_type="ohlc_data")

    def test_validate_dataframe_function_feature_matrix(self) -> None:
        """Test validate_dataframe function with feature matrix."""
        feature_matrix = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=3, freq="1min"),
                "mid_price": [1.05025, 1.05055, 1.05085],
                "returns": [0.0, 0.0003, 0.0003],
                "spread": [0.0005, 0.0005, 0.0005],
                "volume": [1000.0, 1500.0, 2000.0],
                "vwap": [1.05020, 1.05050, 1.05080],
                "ofi": [0.1, -0.2, 0.3],
                "micro_price": [1.05023, 1.05053, 1.05083],
            }
        )
        # Should not raise
        validate_dataframe(feature_matrix, data_type="feature_matrix")

    def test_schema_validation_rejects_invalid_data(self) -> None:
        """Test that schema validation properly rejects invalid data."""
        # Create invalid tick data (crossed spread)
        invalid_tick_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-01-01", periods=3, freq="1s"),
                "bid": [1.0500, 1.0501, 1.0502],
                "ask": [1.0400, 1.0506, 1.0507],  # First ask < bid
                "volume": [100.0, 150.0, 200.0],
            }
        )

        with pytest.raises(ValidationError):
            validate_dataframe(invalid_tick_data, data_type="tick_data")
