"""
Unit tests for aggregator functions.

These tests validate that each aggregator function correctly processes
time-windowed tick data and returns expected outputs.
"""

import numpy as np
import pandas as pd
import pytest

# Import aggregator functions
from atlasfx.data.aggregators import (
    close,
    high,
    low,
    mean,
    micro_price,
    ofi,
    tick_count,
    volume,
    vwap,
)


class TestBasicAggregators:
    """Tests for basic aggregator functions (mean, high, low, close)."""

    def test_mean_with_data(self, sample_tick_data):
        """Test mean aggregator with valid data."""
        start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        duration = pd.Timedelta(minutes=5)

        result = mean(start_time, duration, sample_tick_data)

        assert isinstance(result, dict)
        assert "mean" in result
        assert isinstance(result["mean"], (int, float))
        assert not np.isnan(result["mean"])

        # Verify mean is between bid and ask
        mid_prices = (sample_tick_data["askPrice"] + sample_tick_data["bidPrice"]) / 2
        expected_mean = mid_prices.mean()
        assert result["mean"] == pytest.approx(expected_mean)

    def test_mean_with_empty_data(self):
        """Test mean aggregator with empty DataFrame."""
        start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        duration = pd.Timedelta(minutes=5)
        # Create empty DataFrame with required columns
        empty_df = pd.DataFrame(columns=["askPrice", "bidPrice"])

        result = mean(start_time, duration, empty_df)

        assert isinstance(result, dict)
        assert "mean" in result
        assert np.isnan(result["mean"])

    def test_high_with_data(self, sample_tick_data):
        """Test high aggregator with valid data."""
        start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        duration = pd.Timedelta(minutes=5)

        result = high(start_time, duration, sample_tick_data)

        assert isinstance(result, dict)
        assert "high" in result
        assert not np.isnan(result["high"])

        # Verify high is the maximum mid price
        mid_prices = (sample_tick_data["askPrice"] + sample_tick_data["bidPrice"]) / 2
        expected_high = mid_prices.max()
        assert result["high"] == pytest.approx(expected_high)

    def test_low_with_data(self, sample_tick_data):
        """Test low aggregator with valid data."""
        start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        duration = pd.Timedelta(minutes=5)

        result = low(start_time, duration, sample_tick_data)

        assert isinstance(result, dict)
        assert "low" in result
        assert not np.isnan(result["low"])

        # Verify low is the minimum mid price
        mid_prices = (sample_tick_data["askPrice"] + sample_tick_data["bidPrice"]) / 2
        expected_low = mid_prices.min()
        assert result["low"] == pytest.approx(expected_low)

    def test_close_with_data(self, sample_tick_data):
        """Test close aggregator with valid data."""
        start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        duration = pd.Timedelta(minutes=5)

        result = close(start_time, duration, sample_tick_data)

        assert isinstance(result, dict)
        assert "close" in result
        assert not np.isnan(result["close"])

        # Verify close is the last mid price
        mid_prices = (sample_tick_data["askPrice"] + sample_tick_data["bidPrice"]) / 2
        expected_close = mid_prices.iloc[-1]
        assert result["close"] == pytest.approx(expected_close)


class TestVolumeAggregators:
    """Tests for volume-related aggregators."""

    def test_tick_count_with_data(self, sample_tick_data):
        """Test tick_count aggregator with valid data."""
        start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        duration = pd.Timedelta(minutes=5)

        result = tick_count(start_time, duration, sample_tick_data)

        assert isinstance(result, dict)
        assert "tick_count" in result
        assert result["tick_count"] == len(sample_tick_data)

    def test_tick_count_with_empty_data(self):
        """Test tick_count aggregator with empty DataFrame."""
        start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        duration = pd.Timedelta(minutes=5)
        empty_df = pd.DataFrame()

        result = tick_count(start_time, duration, empty_df)

        assert isinstance(result, dict)
        assert "tick_count" in result
        assert result["tick_count"] == 0

    def test_volume_with_data(self, sample_tick_data):
        """Test volume aggregator with valid data."""
        start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        duration = pd.Timedelta(minutes=5)

        result = volume(start_time, duration, sample_tick_data)

        assert isinstance(result, dict)
        assert "volume" in result
        # Volume is sum of askVolume and bidVolume
        expected_volume = sample_tick_data["askVolume"].sum() + sample_tick_data["bidVolume"].sum()
        assert result["volume"] == pytest.approx(expected_volume)


class TestMicrostructureAggregators:
    """Tests for market microstructure aggregators (VWAP, OFI, micro_price)."""

    def test_vwap_with_data(self, sample_tick_data):
        """Test VWAP aggregator with valid data."""
        start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        duration = pd.Timedelta(minutes=5)

        result = vwap(start_time, duration, sample_tick_data)

        assert isinstance(result, dict)
        assert "vwap" in result
        assert not np.isnan(result["vwap"])

        # VWAP should be weighted average of mid prices by total volume
        mid_prices = (sample_tick_data["askPrice"] + sample_tick_data["bidPrice"]) / 2
        total_volumes = sample_tick_data["askVolume"] + sample_tick_data["bidVolume"]
        expected_vwap = (mid_prices * total_volumes).sum() / total_volumes.sum()
        assert result["vwap"] == pytest.approx(expected_vwap)

    def test_ofi_with_data(self, sample_tick_data):
        """Test OFI aggregator with valid data."""
        start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        duration = pd.Timedelta(minutes=5)

        result = ofi(start_time, duration, sample_tick_data)

        assert isinstance(result, dict)
        assert "ofi" in result
        # OFI should be a numeric value (can be positive, negative, or zero)
        assert isinstance(result["ofi"], (int, float, np.integer, np.floating))
        assert not np.isnan(result["ofi"])

    def test_micro_price_with_data(self, sample_tick_data):
        """Test micro_price aggregator with valid data."""
        start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        duration = pd.Timedelta(minutes=5)

        result = micro_price(start_time, duration, sample_tick_data)

        assert isinstance(result, dict)
        assert "micro_price" in result
        assert not np.isnan(result["micro_price"])

        # Micro price should be between the min bid and max ask across all ticks
        # (it's a volume-weighted price that should fall within the bid-ask range)
        min_price = sample_tick_data["bidPrice"].min()
        max_price = sample_tick_data["askPrice"].max()
        assert min_price <= result["micro_price"] <= max_price


class TestAggregatorEdgeCases:
    """Tests for edge cases and error handling."""

    def test_missing_required_columns(self):
        """Test that aggregators raise errors when required columns are missing."""
        start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        duration = pd.Timedelta(minutes=5)

        # Create DataFrame without required columns
        invalid_df = pd.DataFrame({"invalid_column": [1, 2, 3]})

        with pytest.raises(ValueError, match="Missing required columns"):
            mean(start_time, duration, invalid_df)

    def test_single_row_data(self):
        """Test aggregators with single row of data."""
        start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        duration = pd.Timedelta(minutes=5)

        single_row = pd.DataFrame(
            {
                "timestamp": [1704067200000],
                "askPrice": [1.1],
                "bidPrice": [1.0],
                "volume": [100],
            }
        )

        result_mean = mean(start_time, duration, single_row)
        assert result_mean["mean"] == 1.05  # (1.1 + 1.0) / 2

        result_high = high(start_time, duration, single_row)
        assert result_high["high"] == 1.05

        result_low = low(start_time, duration, single_row)
        assert result_low["low"] == 1.05
