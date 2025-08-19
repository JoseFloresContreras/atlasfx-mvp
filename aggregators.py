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

import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict
from logger import log

def mean(start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate the mean mid-price for a time window.
    
    Args:
        start_time: Start time of the window
        duration: Duration of the window
        data: DataFrame containing data for the window (can be empty)
        
    Returns:
        Dict[str, float]: Dictionary with 'mean' key containing the mean mid-price or np.nan if no data
    """
    # Validate required columns
    required_columns = ['askPrice', 'bidPrice']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing required columns for mean aggregator: {missing_columns}"
        log.error(error_msg)
        raise ValueError(error_msg)
    
    if data.empty:
        return {'mean': np.nan}
    
    # Calculate mid price from askPrice and bidPrice
    mid_price = (data['askPrice'] + data['bidPrice']) / 2
    return {'mean': mid_price.mean()}

def high(start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate the high ask price for a time window.
    
    Args:
        start_time: Start time of the window
        duration: Duration of the window
        data: DataFrame containing data for the window (can be empty)
        
    Returns:
        Dict[str, float]: Dictionary with 'high' key containing the high ask price or np.nan if no data
    """
    # Validate required columns
    required_columns = ['askPrice']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing required columns for high aggregator: {missing_columns}"
        log.error(error_msg)
        raise ValueError(error_msg)
    
    if data.empty:
        return {'high': np.nan}
    
    return {'high': data['askPrice'].max()}

def low(start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate the low bid price for a time window.
    
    Args:
        start_time: Start time of the window
        duration: Duration of the window
        data: DataFrame containing data for the window (can be empty)
        
    Returns:
        Dict[str, float]: Dictionary with 'low' key containing the low bid price or np.nan if no data
    """
    # Validate required columns
    required_columns = ['bidPrice']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing required columns for low aggregator: {missing_columns}"
        log.error(error_msg)
        raise ValueError(error_msg)
    
    if data.empty:
        return {'low': np.nan}
    
    return {'low': data['bidPrice'].min()}

def volume(start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate the total volume for a time window.
    
    Args:
        start_time: Start time of the window
        duration: Duration of the window
        data: DataFrame containing data for the window (can be empty)
        
    Returns:
        Dict[str, float]: Dictionary with 'volume' key containing the total volume or np.nan if no data
    """
    # Validate required columns
    required_columns = ['askVolume', 'bidVolume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing required columns for volume aggregator: {missing_columns}"
        log.error(error_msg)
        raise ValueError(error_msg)
    
    if data.empty:
        return {'volume': np.nan}
    
    # Sum both ask and bid volumes
    total_volume = data['askVolume'].sum() + data['bidVolume'].sum()
    return {'volume': total_volume}

def volatility(start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate the volatility for a time window.
    
    Args:
        start_time: Start time of the window
        duration: Duration of the window
        data: DataFrame containing data for the window (can be empty)
        
    Returns:
        Dict[str, float]: Dictionary with 'volatility' key containing the volatility measure or np.nan if no data
    """
    # Validate required columns
    required_columns = ['askPrice', 'bidPrice']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing required columns for volatility aggregator: {missing_columns}"
        log.error(error_msg)
        raise ValueError(error_msg)
    
    if data.empty or len(data) < 2:
        return {'volatility': np.nan}
    
    # Calculate mid price changes
    mid_price = (data['askPrice'] + data['bidPrice']) / 2
    price_changes = mid_price.diff().dropna()
    
    # Calculate standard deviation of price changes
    volatility = price_changes.std()
    return {'volatility': volatility}

def spread(start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate the average spread for a time window.
    
    Args:
        start_time: Start time of the window
        duration: Duration of the window
        data: DataFrame containing data for the window (can be empty)
        
    Returns:
        Dict[str, float]: Dictionary with 'spread' key containing the average spread or np.nan if no data
    """
    # Validate required columns
    required_columns = ['askPrice', 'bidPrice']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing required columns for spread aggregator: {missing_columns}"
        log.error(error_msg)
        raise ValueError(error_msg)
    
    if data.empty:
        return {'spread': np.nan}
    
    # Calculate spread (ask - bid)
    spread = data['askPrice'] - data['bidPrice']
    return {'spread': spread.mean()}

def open(start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate the opening mid-price for a time window.
    
    Args:
        start_time: Start time of the window
        duration: Duration of the window
        data: DataFrame containing data for the window (can be empty)
        
    Returns:
        Dict[str, float]: Dictionary with 'open' key containing the opening mid-price or np.nan if no data
    """
    # Validate required columns
    required_columns = ['askPrice', 'bidPrice']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing required columns for open aggregator: {missing_columns}"
        log.error(error_msg)
        raise ValueError(error_msg)
    
    if data.empty:
        return {'open': np.nan}
    
    # Get the first tick's mid price
    first_tick = data.iloc[0]
    open = (first_tick['askPrice'] + first_tick['bidPrice']) / 2
    return {'open': open}

def close(start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate the closing mid-price for a time window.
    
    Args:
        start_time: Start time of the window
        duration: Duration of the window
        data: DataFrame containing data for the window (can be empty)
        
    Returns:
        Dict[str, float]: Dictionary with 'close' key containing the closing mid-price or np.nan if no data
    """
    # Validate required columns
    required_columns = ['askPrice', 'bidPrice']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing required columns for close aggregator: {missing_columns}"
        log.error(error_msg)
        raise ValueError(error_msg)
    
    if data.empty:
        return {'close': np.nan}
    
    # Get the last tick's mid price
    last_tick = data.iloc[-1]
    close = (last_tick['askPrice'] + last_tick['bidPrice']) / 2
    return {'close': close}

def tick_count(start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame) -> Dict[str, int]:
    """
    Calculate the number of ticks in a time window.
    
    Args:
        start_time: Start time of the window
        duration: Duration of the window
        data: DataFrame containing data for the window (can be empty)
        
    Returns:
        Dict[str, int]: Dictionary with 'tick_count' key containing the number of ticks or 0 if no data
    """
    if data.empty:
        return {'tick_count': 0}
    
    return {'tick_count': len(data)}

def ofi(start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame) -> Dict[str, float]:
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
        Dict[str, float]: Dictionary with 'ofi' key containing the OFI value or np.nan if no data
    """
    # Validate required columns
    required_columns = ['askPrice', 'bidPrice', 'askVolume', 'bidVolume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing required columns for OFI aggregator: {missing_columns}"
        log.error(error_msg)
        raise ValueError(error_msg)
    
    if data.empty or len(data) < 2:
        return {'ofi': np.nan}

    # Calculate mean price and volume for each tick
    data = data.copy()
    data['price'] = (data['askPrice'] + data['bidPrice']) / 2
    data['volume'] = data['askVolume'] + data['bidVolume']

    # Initialize tick direction array
    tick_direction = np.zeros(len(data), dtype=int)

    # Determine direction for each tick
    for i in range(1, len(data)):
        if data['price'].iloc[i] > data['price'].iloc[i - 1]:
            tick_direction[i] = 1  # Buy
        elif data['price'].iloc[i] < data['price'].iloc[i - 1]:
            tick_direction[i] = -1  # Sell
        else:
            tick_direction[i] = tick_direction[i - 1]  # Use previous direction if price is unchanged

    # Calculate OFI as the sum of (tick_direction * volume)
    data['tick_direction'] = tick_direction
    ofi_value = (data['tick_direction'] * data['volume']).sum()

    return {'ofi': ofi_value}

def vwap(start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame) -> Dict[str, float]:
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
        Dict[str, float]: Dictionary with 'vwap' key containing the VWAP value or np.nan if no data
    """
    # Validate required columns
    required_columns = ['askPrice', 'bidPrice', 'askVolume', 'bidVolume']
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        error_msg = f"Missing required columns for VWAP aggregator: {missing_columns}"
        log.error(error_msg)
        raise ValueError(error_msg)

    if data.empty:
        return {'vwap': np.nan}

    # Calculate mid price and total volume per tick
    data = data.copy()
    data['price'] = (data['askPrice'] + data['bidPrice']) / 2
    data['volume'] = data['askVolume'] + data['bidVolume']

    # Compute VWAP
    total_volume = data['volume'].sum()
    if total_volume == 0:
        return {'vwap': np.nan}
    
    weighted_price_sum = (data['price'] * data['volume']).sum()
    vwap_value = weighted_price_sum / total_volume

    return {'vwap': vwap_value}


# Example of a future multi-output aggregator:
# def price_stats(start_time: pd.Timestamp, duration: pd.Timedelta, data: pd.DataFrame) -> Dict[str, float]:
#     """
#     Calculate multiple price statistics for a time window.
#     
#     Returns:
#         Dict[str, float]: Dictionary with multiple price statistics
#     """
#     # Validate required columns
#     required_columns = ['askPrice', 'bidPrice']
#     missing_columns = [col for col in required_columns if col not in data.columns]
#     if missing_columns:
#         error_msg = f"Missing required columns for price_stats aggregator: {missing_columns}"
#         log.error(error_msg)
#         raise ValueError(error_msg)
#     
#     if data.empty:
#         return {
#             'price_mean': np.nan,
#             'price_std': np.nan,
#             'price_min': np.nan,
#             'price_max': np.nan
#         }
#     
#     # Calculate mid prices
#     mid_prices = (data['askPrice'] + data['bidPrice']) / 2
#     
#     return {
#         'price_mean': mid_prices.mean(),
#         'price_std': mid_prices.std(),
#         'price_min': mid_prices.min(),
#         'price_max': mid_prices.max()
#     } 