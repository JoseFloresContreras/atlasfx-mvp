"""
Aggregator functions for time-based data processing.
Each function follows the signature: function_name(start_time, duration, data)
where:
- start_time: pd.Timestamp - Start time of the window
- duration: pd.Timedelta - Duration of the window
- data: pd.DataFrame - Data for the window (can be empty)

Assumes data has 'askPrice' and 'bidPrice' columns.

Each aggregator returns a dictionary where keys are output names and values are the calculated metrics.
This design supports both single-output (current) and multi-output (future) aggregators.
"""

import numpy as npLFimport pandas as pdLFLFfrom atlasfx.utils.logging import logLFLFLFdef mean(start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame) -> dict[str, float]:LF    """
    Calculate the mean mid-price for a time window.

    Args:
        start_time: Start time of the window
        duration: Duration of the window
        data: DataFrame containing data for the window (can be empty)

    Returns:
        dict[str, float]: Dictionary with 'mean' key containing the mean mid-price or np.nan if no data
    """
    # Validate required columns
    required_columns = ["askPrice", "bidPrice"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing required columns for mean aggregator: {missing_columns}"
        log.error(error_msg)
        raise ValueError(error_msg)

    if data.empty:
        return {"mean": np.nan}

    # Calculate mid price from askPrice and bidPrice
    mid_price = (data["askPrice"] + data["bidPrice"]) / 2
    return {"mean": mid_price.mean()}


def high(start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame) -> dict[str, float]:
    """
    Calculate the high mid-price for a time window.

    Args:
        start_time: Start time of the window
        duration: Duration of the window
        data: DataFrame containing data for the window (can be empty)

    Returns:
        dict[str, float]: Dictionary with 'high' key containing the high mid-price or np.nan if no data
    """
    # Validate required columns
    required_columns = ["askPrice", "bidPrice"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing required columns for high aggregator: {missing_columns}"
        log.error(error_msg)
        raise ValueError(error_msg)

    if data.empty:
        return {"high": np.nan}

    # Calculate mid price from askPrice and bidPrice, then find the maximum
    mid_price = (data["askPrice"] + data["bidPrice"]) / 2
    return {"high": mid_price.max()}


def low(start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame) -> dict[str, float]:
    """
    Calculate the low mid-price for a time window.

    Args:
        start_time: Start time of the window
        duration: Duration of the window
        data: DataFrame containing data for the window (can be empty)

    Returns:
        dict[str, float]: Dictionary with 'low' key containing the low mid-price or np.nan if no data
    """
    # Validate required columns
    required_columns = ["askPrice", "bidPrice"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing required columns for low aggregator: {missing_columns}"
        log.error(error_msg)
        raise ValueError(error_msg)

    if data.empty:
        return {"low": np.nan}

    # Calculate mid price from askPrice and bidPrice, then find the minimum
    mid_price = (data["askPrice"] + data["bidPrice"]) / 2
    return {"low": mid_price.min()}


def volume(
    start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame
) -> dict[str, float]:
    """
    Calculate the total volume for a time window.

    Args:
        start_time: Start time of the window
        duration: Duration of the window
        data: DataFrame containing data for the window (can be empty)

    Returns:
        dict[str, float]: Dictionary with 'volume' key containing the total volume or np.nan if no data
    """
    # Validate required columns
    required_columns = ["askVolume", "bidVolume"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing required columns for volume aggregator: {missing_columns}"
        log.error(error_msg)
        raise ValueError(error_msg)

    if data.empty:
        return {"volume": np.nan}

    # Sum both ask and bid volumes
    total_volume = data["askVolume"].sum() + data["bidVolume"].sum()
    return {"volume": total_volume}


def volatility(
    start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame
) -> dict[str, float]:
    """
    Calculate the volatility for a time window.

    Args:
        start_time: Start time of the window
        duration: Duration of the window
        data: DataFrame containing data for the window (can be empty)

    Returns:
        dict[str, float]: Dictionary with 'volatility' key containing the volatility measure or np.nan if no data
    """
    # Validate required columns
    required_columns = ["askPrice", "bidPrice"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing required columns for volatility aggregator: {missing_columns}"
        log.error(error_msg)
        raise ValueError(error_msg)

    if data.empty or len(data) < 2:
        return {"volatility": np.nan}

    # Calculate mid price changes
    mid_price = (data["askPrice"] + data["bidPrice"]) / 2
    price_changes = mid_price.diff().dropna()

    # Calculate standard deviation of price changes
    volatility = price_changes.std()
    return {"volatility": volatility}


def spread(
    start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame
) -> dict[str, float]:
    """
    Calculate the average spread for a time window.

    Args:
        start_time: Start time of the window
        duration: Duration of the window
        data: DataFrame containing data for the window (can be empty)

    Returns:
        dict[str, float]: Dictionary with 'spread' key containing the average spread or np.nan if no data
    """
    # Validate required columns
    required_columns = ["askPrice", "bidPrice"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing required columns for spread aggregator: {missing_columns}"
        log.error(error_msg)
        raise ValueError(error_msg)

    if data.empty:
        return {"spread": np.nan}

    # Calculate spread (ask - bid)
    spread = data["askPrice"] - data["bidPrice"]
    return {"spread": spread.mean()}


def open(start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame) -> dict[str, float]:
    """
    Calculate the opening mid-price for a time window.

    Args:
        start_time: Start time of the window
        duration: Duration of the window
        data: DataFrame containing data for the window (can be empty)

    Returns:
        dict[str, float]: Dictionary with 'open' key containing the opening mid-price or np.nan if no data
    """
    # Validate required columns
    required_columns = ["askPrice", "bidPrice"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing required columns for open aggregator: {missing_columns}"
        log.error(error_msg)
        raise ValueError(error_msg)

    if data.empty:
        return {"open": np.nan}

    # Get the first tick's mid price
    first_tick = data.iloc[0]
    open = (first_tick["askPrice"] + first_tick["bidPrice"]) / 2
    return {"open": open}


def close(start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame) -> dict[str, float]:
    """
    Calculate the closing mid-price for a time window.

    Args:
        start_time: Start time of the window
        duration: Duration of the window
        data: DataFrame containing data for the window (can be empty)

    Returns:
        dict[str, float]: Dictionary with 'close' key containing the closing mid-price or np.nan if no data
    """
    # Validate required columns
    required_columns = ["askPrice", "bidPrice"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing required columns for close aggregator: {missing_columns}"
        log.error(error_msg)
        raise ValueError(error_msg)

    if data.empty:
        return {"close": np.nan}

    # Get the last tick's mid price
    last_tick = data.iloc[-1]
    close = (last_tick["askPrice"] + last_tick["bidPrice"]) / 2
    return {"close": close}


def tick_count(
    start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame
) -> dict[str, int]:
    """
    Calculate the number of ticks in a time window.

    Args:
        start_time: Start time of the window
        duration: Duration of the window
        data: DataFrame containing data for the window (can be empty)

    Returns:
        dict[str, int]: Dictionary with 'tick_count' key containing the number of ticks or 0 if no data
    """
    if data.empty:
        return {"tick_count": 0}

    return {"tick_count": len(data)}


def ofi(start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame) -> dict[str, float]:
    """
    Calculate the Order Flow Imbalance (OFI) for a time window.

    OFI is calculated by classifying each tick based on price movement,
    assigning a direction (+1 for buy, -1 for sell), and summing the
    product of direction and volume over the interval.

    Args:
        start_time: Start time of the window
        duration: Duration of the window
        data: DataFrame containing tick data (can be empty)

    Returns:
        dict[str, float]: Dictionary with 'ofi' key containing the OFI value or np.nan if no data
    """
    # Validate required columns
    required_columns = ["askPrice", "bidPrice", "askVolume", "bidVolume"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing required columns for OFI aggregator: {missing_columns}"
        log.error(error_msg)
        raise ValueError(error_msg)

    if data.empty or len(data) < 2:
        return {"ofi": np.nan}

    # Calculate mean price and volume for each tick
    data = data.copy()
    data["price"] = (data["askPrice"] + data["bidPrice"]) / 2
    data["volume"] = data["askVolume"] + data["bidVolume"]

    # Calculate price differences
    price_diff = data["price"].diff()

    # Initialize tick direction array with NaNs (same size as price_diff)
    tick_direction = np.full(len(price_diff), np.nan)

    # Set direction based on price changes (1 for increase, -1 for decrease)
    tick_direction[price_diff > 0] = 1  # Buy
    tick_direction[price_diff < 0] = -1  # Sell

    # Set first value to 0 (no previous price to compare with)
    tick_direction[0] = 0

    # Forward fill the unchanged prices (NaN values) with previous direction
    tick_direction = pd.Series(tick_direction).ffill().fillna(0).astype(int).values

    # Calculate OFI as the sum of (tick_direction * volume)
    # Drop first volume element since tick_direction corresponds to price differences
    volume_for_direction = data["volume"].values
    ofi_value = (tick_direction * volume_for_direction).sum()

    return {"ofi": ofi_value}


def vwap(start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame) -> dict[str, float]:
    """
    Calculate the Volume Weighted Average Price (VWAP) for a time window.

    VWAP is calculated as the sum of (price * volume) divided by the total volume.
    Price is the mid-price: (askPrice + bidPrice) / 2
    Volume is the sum of askVolume and bidVolume per tick.

    Args:
        start_time: Start time of the window
        duration: Duration of the window
        data: DataFrame containing tick data (can be empty)

    Returns:
        dict[str, float]: Dictionary with 'vwap' key containing the VWAP value or np.nan if no data
    """
    # Validate required columns
    required_columns = ["askPrice", "bidPrice", "askVolume", "bidVolume"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing required columns for VWAP aggregator: {missing_columns}"
        log.error(error_msg)
        raise ValueError(error_msg)

    if data.empty:
        return {"vwap": np.nan}

    # Calculate mid price and total volume per tick
    data = data.copy()
    data["price"] = (data["askPrice"] + data["bidPrice"]) / 2
    data["volume"] = data["askVolume"] + data["bidVolume"]

    # Compute VWAP
    total_volume = data["volume"].sum()
    if total_volume == 0:
        return {"vwap": np.nan}

    weighted_price_sum = (data["price"] * data["volume"]).sum()
    vwap_value = weighted_price_sum / total_volume

    return {"vwap": vwap_value}


def micro_price(
    start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame
) -> dict[str, float]:
    """
    Calculate the microprice for a time window.

    Microprice is calculated as: (ask * bid_vol + bid * ask_vol) / (bid_vol + ask_vol)
    This provides a volume-weighted price that reflects the true market price better than mid-price.

    Args:
        start_time: Start time of the window
        duration: Duration of the window
        data: DataFrame containing tick data (can be empty)

    Returns:
        dict[str, float]: Dictionary with 'micro_price' key containing the microprice value or np.nan if no data
    """
    # Validate required columns
    required_columns = ["askPrice", "bidPrice", "askVolume", "bidVolume"]
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing required columns for micro_price aggregator: {missing_columns}"
        log.error(error_msg)
        raise ValueError(error_msg)

    if data.empty:
        return {"micro_price": np.nan}

    # Extract values
    A = data["askPrice"]  # ask price
    B = data["bidPrice"]  # bid price
    Va = data["askVolume"]  # ask volume
    Vb = data["bidVolume"]  # bid volume

    # Calculate microprice: (A * Vb + B * Va) / (Vb + Va)
    # Handle division by zero by checking total volume
    total_volume = Vb + Va
    if total_volume.sum() == 0:
        return {"micro_price": np.nan}

    # Calculate microprice for each tick and then take the mean
    microprice_per_tick = (A * Vb + B * Va) / total_volume
    microprice_value = microprice_per_tick.mean()

    return {"micro_price": microprice_value}
