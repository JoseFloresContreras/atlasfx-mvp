"""
Pytest configuration and shared fixtures for AtlasFX MVP tests.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add data-pipeline directory to Python path for imports
project_root = Path(__file__).parent.parent
data_pipeline_path = project_root / "data-pipeline"
sys.path.insert(0, str(data_pipeline_path))


@pytest.fixture
def sample_tick_data():
    """
    Create sample tick data for testing.

    Returns:
        pd.DataFrame: Sample tick data with timestamp, askPrice, bidPrice, askVolume, bidVolume columns
    """
    n_rows = 100
    start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")

    # Create timestamps (1 second apart)
    timestamps = [start_time + pd.Timedelta(seconds=i) for i in range(n_rows)]

    # Ensure bid < ask for realistic data
    bid_prices = np.random.uniform(1.0, 1.1, n_rows)
    ask_prices = bid_prices + np.random.uniform(0.0001, 0.001, n_rows)  # Small spread

    data = pd.DataFrame({
        'timestamp': [int(ts.timestamp() * 1000) for ts in timestamps],
        'askPrice': ask_prices,
        'bidPrice': bid_prices,
        'askVolume': np.random.uniform(100, 1000, n_rows),
        'bidVolume': np.random.uniform(100, 1000, n_rows),
    })

    return data


@pytest.fixture
def sample_aggregated_data():
    """
    Create sample aggregated data for testing.

    Returns:
        pd.DataFrame: Sample aggregated data with OHLC and other features
    """
    n_rows = 50
    start_time = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")

    # Create timestamps (5 minutes apart)
    timestamps = [start_time + pd.Timedelta(minutes=5*i) for i in range(n_rows)]

    data = pd.DataFrame({
        'start_time': timestamps,
        'tick_count': np.random.randint(10, 100, n_rows),
        'high': np.random.uniform(1.05, 1.15, n_rows),
        'low': np.random.uniform(0.95, 1.05, n_rows),
        'close': np.random.uniform(1.0, 1.1, n_rows),
        'volume': np.random.uniform(1000, 10000, n_rows),
        'vwap': np.random.uniform(1.0, 1.1, n_rows),
        'ofi': np.random.uniform(-0.1, 0.1, n_rows),
        'micro_price': np.random.uniform(1.0, 1.1, n_rows),
    })

    data.set_index('start_time', inplace=True)
    return data


@pytest.fixture
def sample_config():
    """
    Create sample configuration dictionary for testing.

    Returns:
        dict: Sample configuration
    """
    return {
        'time_window': '5min',
        'output_directory': 'data',
        'split': {
            'train': 0.7,
            'val': 0.15,
            'test': 0.15
        }
    }
