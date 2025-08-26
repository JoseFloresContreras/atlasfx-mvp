"""
Featurizer functions for time-series data processing.
Each function follows the signature: function_name(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame
where:
- dataframe: pd.DataFrame - Input dataframe with time index
- config: Dict[str, Any] - Dictionary containing configuration parameters (e.g., window_size, etc.)

Each featurizer returns a DataFrame with the same time index as input, but may have NaN values
for indices where the rolling window cannot be fully formed.

This design supports both single-output and multi-output featurizers with flexible configuration.
"""

import numpy as np
import pandas as pd
import pandas_ta as ta
from typing import Dict, List, Optional, Any
from logger import log
from zoneinfo import ZoneInfo
import datetime as dt

def log_pct_change(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate log percentage change for all numeric columns.
    
    Args:
        dataframe: Input dataframe with time index
        config: Dictionary containing configuration parameters
            - no configurations required as log percentage change is calculated from one timestep to other. window_size=1
        
    Returns:
        pd.DataFrame: DataFrame with log percentage change columns (original columns with '_log_pct_change' suffix)
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Get numeric columns only
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns

    if len(numeric_columns) == 0:
        log.warning("⚠️  No numeric columns found for log percentage change calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    for col in numeric_columns:
        # Calculate log percentage change: log(current_price / previous_price)
        # This is equivalent to log(current_price) - log(previous_price)
        result_df[f'{col} | log_pct_change'] = np.log1p(dataframe[col]).diff()
    
    return result_df

def _session_flag_local(index_utc: pd.DatetimeIndex, tz: str, open_h: int, close_h: int):
    """Return int8 mask where local time in tz is within [open_h:00, close_h:00)."""
    idx_local = index_utc.tz_convert(tz)
    t = idx_local.time
    open_t = dt.time(open_h, 0)
    close_t = dt.time(close_h, 0)
    if open_t < close_t:  # same-day session (most)
        mask = (t >= open_t) & (t < close_t)
    else:                 # overnight session (rare if using local market hours)
        mask = (t >= open_t) | (t < close_t)
    return mask.astype(np.int8)

def sydney_session(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create Sydney trading session flags.
    
    Sydney local trading hours ~ 08:00–17:00; UTC becomes 21:00–06:00 or 22:00–07:00 depending on DST.
    
    Args:
        dataframe: Input dataframe with time index
        config: Dictionary containing configuration parameters (optional)
        
    Returns:
        pd.DataFrame: DataFrame with Sydney session flag column
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    m = _session_flag_local(dataframe.index, 'Australia/Sydney', 8, 17)
    return pd.DataFrame({'sydney_session': m}, index=dataframe.index)

def tokyo_session(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create Tokyo trading session flags.
    
    Tokyo has no DST; local 09:00–18:00 → ~00:00–09:00 UTC year-round.
    
    Args:
        dataframe: Input dataframe with time index
        config: Dictionary containing configuration parameters (optional)
        
    Returns:
        pd.DataFrame: DataFrame with Tokyo session flag column
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    m = _session_flag_local(dataframe.index, 'Asia/Tokyo', 9, 18)
    return pd.DataFrame({'tokyo_session': m}, index=dataframe.index)

def london_session(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create London trading session flags.
    
    London local 08:00–17:00; UTC shows the 1h DST swing automatically.
    
    Args:
        dataframe: Input dataframe with time index
        config: Dictionary containing configuration parameters (optional)
        
    Returns:
        pd.DataFrame: DataFrame with London session flag column
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    m = _session_flag_local(dataframe.index, 'Europe/London', 8, 17)
    return pd.DataFrame({'london_session': m}, index=dataframe.index)

def ny_session(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create New York trading session flags.
    
    New York local 08:00–17:00; UTC becomes 13:00–22:00 (EST) or 12:00–21:00 (EDT).
    
    Args:
        dataframe: Input dataframe with time index
        config: Dictionary containing configuration parameters (optional)
        
    Returns:
        pd.DataFrame: DataFrame with New York session flag column
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    m = _session_flag_local(dataframe.index, 'America/New_York', 8, 17)
    return pd.DataFrame({'ny_session': m}, index=dataframe.index)

def weekend(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Create weekend flags based on tick count columns.
    
    Returns 1 if all columns containing 'tick_count' are 0 (indicating no trading activity),
    otherwise returns 0. This helps identify weekend periods when markets are closed.
    
    Args:
        dataframe: Input dataframe with time index
        config: Dictionary containing configuration parameters (optional)
        
    Returns:
        pd.DataFrame: DataFrame with weekend flag column
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Find all columns containing 'tick_count'
    tick_count_columns = [col for col in dataframe.columns if 'tick_count' in col]
    
    if not tick_count_columns:
        log.critical("⚠️  No columns containing 'tick_count' found for weekend detection")
        return pd.DataFrame()
    
    # Check if all tick_count columns are 0 for each row
    # If all tick_count columns are 0, it's likely a weekend
    weekend_flag = (dataframe[tick_count_columns] == 0).all(axis=1).astype(np.int8)
    
    return pd.DataFrame({'weekend': weekend_flag}, index=dataframe.index)

def vwap_deviation(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate VWAP deviation for all trading pairs.
    
    The deviation is calculated as: (close_price - vwap) / vwap
    This provides a normalized mean-reversion signal indicating how far the close price
    has deviated from the volume-weighted average price.
    
    Args:
        dataframe: Input dataframe with time index and columns in format "{symbol} | {metric}"
        config: Dictionary containing configuration parameters (optional)
        
    Returns:
        pd.DataFrame: DataFrame with VWAP deviation columns for each pair
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Find all VWAP columns (format: "{symbol} | vwap")
    vwap_columns = [col for col in dataframe.columns if '| vwap' in col]
    
    if not vwap_columns:
        log.critical("⚠️  No VWAP columns found (expected format: '{symbol} | vwap')")
        return pd.DataFrame()
    
    # Extract symbols from VWAP columns and find corresponding close columns
    pairs = []
    for vwap_col in vwap_columns:
        # Extract symbol from VWAP column name (e.g., "audusd | vwap" -> "audusd")
        symbol = vwap_col.split(' | ')[0]
        close_col = f"{symbol} | close"
        
        if close_col in dataframe.columns:
            pairs.append((symbol, close_col, vwap_col))
        else:
            log.critical(f"⚠️  No corresponding close column found for {vwap_col}. Expected: {close_col}")
            return pd.DataFrame()

    if not pairs:
        log.critical("⚠️  No valid VWAP-close pairs found for deviation calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate VWAP deviation for each pair
    for symbol, close_col, vwap_col in pairs:
        vwap_values = dataframe[vwap_col]
        close_values = dataframe[close_col]
        
        # Calculate deviation
        deviation = close_values - vwap_values
        
        # Create column name
        col_name = f"{symbol} | vwap_deviation"
        result_df[col_name] = deviation
    
    return result_df

def ofi_rolling_mean(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate rolling mean of OFI (Order Flow Imbalance) for all columns containing 'ofi'.
    
    Args:
        dataframe: Input dataframe with time index and columns in format "{symbol} | {metric}"
        config: Dictionary containing configuration parameters
            - window_size: Window size for rolling mean calculation (default: 14)
        
    Returns:
        pd.DataFrame: DataFrame with OFI rolling mean columns for each pair
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Extract configuration
    window_size = config.get('window_size')
    
    # Find all OFI columns (format: "{symbol} | ofi")
    ofi_columns = [col for col in dataframe.columns if '| ofi' in col]
    
    if not ofi_columns:
        log.critical("⚠️  No OFI columns found (expected format: '{symbol} | ofi')")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate rolling mean for each OFI column
    for ofi_col in ofi_columns:
        # Extract symbol from OFI column name (e.g., "audusd | ofi" -> "audusd")
        symbol = ofi_col.split(' | ')[0]
        
        # Calculate rolling mean
        rolling_mean = dataframe[ofi_col].rolling(window=window_size, min_periods=1).mean()
        
        # Create column name
        col_name = f"{symbol} | ofi_rolling_mean_{window_size}"
        result_df[col_name] = rolling_mean
    
    return result_df

def atr(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate Average True Range (ATR) for all trading pairs.
    
    ATR measures market volatility by decomposing the entire range of an asset price for that period.
    True Range is the greatest of the following:
    1. Current High - Current Low
    2. |Current High - Previous Close|
    3. |Current Low - Previous Close|
    
    Args:
        dataframe: Input dataframe with time index and columns in format "{symbol} | {metric}"
        config: Dictionary containing configuration parameters
            - window_size: Window size for ATR calculation (default: 14)
        
    Returns:
        pd.DataFrame: DataFrame with ATR columns for each pair
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Extract configuration
    window_size = config.get('window_size', 14)  # Default to 14 if not specified
    
    # Find all high, low, and close columns for each symbol
    high_columns = [col for col in dataframe.columns if '| high' in col]
    low_columns = [col for col in dataframe.columns if '| low' in col]
    close_columns = [col for col in dataframe.columns if '| close' in col]
    
    if not high_columns or not low_columns or not close_columns:
        log.critical("⚠️  Missing required OHLC columns for ATR calculation")
        return pd.DataFrame()
    
    # Create a mapping of symbols to their OHLC columns
    symbol_ohlc = {}
    for high_col in high_columns:
        symbol = high_col.split(' | ')[0]
        low_col = f"{symbol} | low"
        close_col = f"{symbol} | close"
        
        if low_col in low_columns and close_col in close_columns:
            symbol_ohlc[symbol] = {
                'high': high_col,
                'low': low_col,
                'close': close_col
            }
    
    if not symbol_ohlc:
        log.critical("⚠️  No valid OHLC column combinations found for ATR calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate ATR for each symbol
    for symbol, ohlc_cols in symbol_ohlc.items():
        try:
            high = dataframe[ohlc_cols['high']]
            low = dataframe[ohlc_cols['low']]
            close = dataframe[ohlc_cols['close']]
            
            # Calculate True Range
            close_prev = close.shift(1)
            tr1 = high - low
            tr2 = abs(high - close_prev)
            tr3 = abs(low - close_prev)
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR as rolling mean of True Range
            atr_values = true_range.rolling(window=window_size, min_periods=1).mean()
            
            # Create column name for ATR values
            col_name = f"{symbol} | atr_{window_size}"
            result_df[col_name] = atr_values.astype(np.float32)
            
        except Exception as e:
            log.error(f"Error calculating ATR for {symbol}: {str(e)}")
            continue
    
    return result_df

def atr_log_pct_change(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate log percentage change of Average True Range (ATR) for all trading pairs.
    
    This function reuses the atr() function to calculate ATR values and then returns
    the log percentage change for symmetric distribution.
    
    Args:
        dataframe: Input dataframe with time index and columns in format "{symbol} | {metric}"
        config: Dictionary containing configuration parameters
            - window_size: Window size for ATR calculation (default: 14)
        
    Returns:
        pd.DataFrame: DataFrame with ATR log percentage change columns for each pair
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Extract configuration
    window_size = config.get('window_size', 14)  # Default to 14 if not specified
    
    # Reuse the atr function to get ATR values
    atr_df = atr(dataframe, config)
    
    if atr_df.empty:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate log percentage change for each ATR column
    for col in atr_df.columns:
        try:
            # Extract symbol from ATR column name (e.g., "audusd | atr_14" -> "audusd")
            symbol = col.split(' | ')[0]
            
            # Get ATR values
            atr_values = atr_df[col]
            
            # Calculate log percentage change: log(current_atr / previous_atr)
            # This is equivalent to log(current_atr) - log(previous_atr)
            atr_log_pct_change = np.log1p(atr_values).diff()
            
            # Create column name for ATR log percentage change
            log_pct_col_name = f"{symbol} | atr_{window_size}_log_pct_change"
            result_df[log_pct_col_name] = atr_log_pct_change.astype(np.float32)
            
        except Exception as e:
            log.critical(f"Error calculating ATR log percentage change for {col}: {str(e)}")
            return pd.DataFrame()
    
    return result_df
    
def volatility_ratio(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate volatility ratio (short-term ATR / long-term ATR) for all trading pairs.
    
    This ratio helps detect regime changes in volatility. A ratio > 1 indicates
    increasing volatility (short-term > long-term), while < 1 indicates decreasing volatility.
    
    This function reuses the atr() function to calculate both short-term and long-term ATR values.
    
    Args:
        dataframe: Input dataframe with time index and OHLC columns in format "{symbol} | {metric}"
        config: Dictionary containing configuration parameters
            - short_window: Window size for short-term ATR (default: 14)
            - long_window: Window size for long-term ATR (default: 60)
        
    Returns:
        pd.DataFrame: DataFrame with volatility ratio columns for each pair
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Extract configuration
    short_window = config.get('short_window', 14)  # Default to 14 if not specified
    long_window = config.get('long_window', 60)    # Default to 60 if not specified
    
    # Get short-term ATR values
    short_config = {'window_size': short_window}
    short_atr_df = atr(dataframe, short_config)
    
    # Get long-term ATR values
    long_config = {'window_size': long_window}
    long_atr_df = atr(dataframe, long_config)
    
    if short_atr_df.empty or long_atr_df.empty:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate volatility ratio for each symbol
    for short_col in short_atr_df.columns:
        try:
            # Extract symbol from short ATR column name (e.g., "audusd | atr_14" -> "audusd")
            symbol = short_col.split(' | ')[0]
            
            # Find corresponding long ATR column
            long_col = f"{symbol} | atr_{long_window}"
            
            if long_col not in long_atr_df.columns:
                log.critical(f"⚠️  Missing long-term ATR column for {symbol}: {long_col}")
                return pd.DataFrame()
            
            short_atr = short_atr_df[short_col]
            long_atr = long_atr_df[long_col]
            
            # Calculate volatility ratio, avoiding division by zero
            volatility_ratio_values = np.log1p(short_atr / long_atr).diff()
            
            # Create column name
            col_name = f"{symbol} | volatility_ratio_{short_window}_{long_window}"
            result_df[col_name] = volatility_ratio_values.astype(np.float32)
            
        except Exception as e:
            log.critical(f"Error calculating volatility ratio for {short_col}: {str(e)}")
            return pd.DataFrame()
    
    return result_df

def bipower_variation_features(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compute two features per symbol from columns named like "{symbol} | close":
      - "{symbol} | bipower_variation_continuous_{window_size}"
      - "{symbol} | bipower_variation_jump_{window_size}"

    Uses:
      - log returns (r_t = log P_t - log P_{t-1})
      - per-step average BV and RV:
          BV_t = (π/2) * mean_{window_size-1} ( |r_i||r_{i-1}| )
          RV_t = mean_{window_size} ( r_i^2 )
      - jump = max(RV - BV, 0)

    Args:
        dataframe: Input dataframe with time index and columns in format "{symbol} | close"
        config: Dictionary containing configuration parameters
            - window_size: Window size for bipower variation calculation (default: 30)

    Returns:
        pd.DataFrame: DataFrame with bipower variation features for each pair
    """
    try:
        if dataframe.empty:
            return pd.DataFrame()

        # Extract configuration
        window_size = config.get('window_size', 30)  # Default to 30 if not specified
        
        # enforce valid window for the |r_t||r_{t-1}| mean
        window_size = max(int(window_size), 2)

        close_cols = [c for c in dataframe.columns if " | close" in c]
        if not close_cols:
            return pd.DataFrame()

        out = pd.DataFrame(index=dataframe.index)

        for close_col in close_cols:
            symbol = close_col.split(" | ")[0]
            px = dataframe[close_col].astype(float)

            # log returns (protect against non-positive prices)
            r = np.log(px.where(px > 0, np.nan)).diff()

            abs_r = r.abs()
            prod = abs_r * abs_r.shift(1)  # |r_t| * |r_{t-1}|

            # Bipower Variation: per-step average over (window_size-1) pairs
            bv = (np.pi / 2.0) * prod.rolling(
                window=window_size - 1,
                min_periods=window_size - 1
            ).mean()

            # Realized variance: per-step average over window_size
            rv = (r ** 2).rolling(window=window_size, min_periods=window_size).mean()

            jump = (rv - bv).clip(lower=0)

            bv_norm = np.log1p(np.sqrt(bv)).diff()
            jump_norm = np.log1p(np.sqrt(jump)).diff() 
            

            out[f"{symbol} | bipower_variation_continuous_{window_size}"] = bv_norm.astype(np.float32)
            out[f"{symbol} | bipower_variation_jump_{window_size}"] = jump_norm.astype(np.float32)

        return out

    except Exception:
        return pd.DataFrame()

def rsi(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate RSI (Relative Strength Index) for all trading pairs.
    
    RSI is a momentum oscillator that measures the speed and magnitude of recent price changes
    to evaluate overbought or oversold conditions. RSI ranges from 0 to 100.
    
    Args:
        dataframe: Input dataframe with time index and columns in format "{symbol} | close"
        config: Dictionary containing configuration parameters
            - window_size: Window size for RSI calculation (default: 14)
        
    Returns:
        pd.DataFrame: DataFrame with RSI columns for each pair
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Extract configuration
    window_size = config.get('window_size', 14)  # Default to 14 if not specified
    
    # Find all close columns (format: "{symbol} | close")
    close_columns = [col for col in dataframe.columns if '| close' in col]
    
    if not close_columns:
        log.warning("⚠️  No columns containing '| close' found for RSI calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate RSI for each close column
    for close_col in close_columns:
        try:
            # Extract symbol from close column name (e.g., "audusd | close" -> "audusd")
            symbol = close_col.split(' | ')[0]
            close = dataframe[close_col]
            
            # Calculate RSI using pandas_ta
            rsi_values = ta.rsi(close, length=window_size)
            
            # Create column name for RSI values
            col_name = f"{symbol} | rsi_{window_size}"
            result_df[col_name] = rsi_values.astype(np.float32)
            
        except Exception as e:
            log.error(f"Error calculating RSI for {close_col}: {str(e)}")
            continue
    
    return result_df

def bollinger_width(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate Bollinger Band Width for all trading pairs.
    
    Bollinger Band Width = (Upper Band - Lower Band) / Middle Band (SMA)
    This indicator helps detect volatility compression and expansion phases.
    Lower values indicate volatility compression, higher values indicate expansion.
    
    Args:
        dataframe: Input dataframe with time index and columns in format "{symbol} | close"
        config: Dictionary containing configuration parameters
            - window_size: Window size for Bollinger Bands calculation (default: 20)
        
    Returns:
        pd.DataFrame: DataFrame with Bollinger Band Width columns for each pair
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Extract configuration
    window_size = config.get('window_size', 20)  # Default to 20 if not specified
    
    # Find all close columns (format: "{symbol} | close")
    close_columns = [col for col in dataframe.columns if '| close' in col]
    
    if not close_columns:
        log.warning("⚠️  No columns containing '| close' found for Bollinger Band Width calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate Bollinger Band Width for each close column
    for close_col in close_columns:
        try:
            # Extract symbol from close column name (e.g., "audusd | close" -> "audusd")
            symbol = close_col.split(' | ')[0]
            close = dataframe[close_col]
            
            # Calculate Bollinger Bands using pandas_ta
            bb = ta.bbands(close, length=window_size)
            
            # Extract the bands
            upper_band = bb['BBU_20_2.0']  # Upper band
            lower_band = bb['BBL_20_2.0']  # Lower band
            middle_band = bb['BBM_20_2.0']  # Middle band (SMA)
            
            # Calculate Bollinger Band Width: (Upper - Lower) / Middle
            bb_width = (upper_band - lower_band) / middle_band
            bb_width = np.log1p(bb_width).diff()
            
            # Create column name for Bollinger Band Width values
            col_name = f"{symbol} | bollinger_width_{window_size}"
            result_df[col_name] = bb_width.astype(np.float32)
            
        except Exception as e:
            log.error(f"Error calculating Bollinger Band Width for {close_col}: {str(e)}")
            continue
    
    return result_df

def return_skewness(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate return skewness for all trading pairs.
    
    Return skewness measures the asymmetry of log returns distribution over a rolling window.
    Positive skewness indicates more upside moves, negative skewness indicates more downside moves.
    
    Args:
        dataframe: Input dataframe with time index and columns in format "{symbol} | close"
        config: Dictionary containing configuration parameters
            - window_size: Window size for skewness calculation (default: 60)
        
    Returns:
        pd.DataFrame: DataFrame with return skewness columns for each pair
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Extract configuration
    window_size = config.get('window_size', 60)  # Default to 60 if not specified
    
    # Find all close columns (format: "{symbol} | close")
    close_columns = [col for col in dataframe.columns if '| close' in col]
    
    if not close_columns:
        log.warning("⚠️  No columns containing '| close' found for return skewness calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate return skewness for each close column
    for close_col in close_columns:
        try:
            # Extract symbol from close column name (e.g., "audusd | close" -> "audusd")
            symbol = close_col.split(' | ')[0]
            close = dataframe[close_col]
            
            # Calculate log returns
            log_returns = np.log(close.where(close > 0, np.nan)).diff()
            
            # Calculate rolling skewness of log returns
            return_skew = log_returns.rolling(window=window_size, min_periods=window_size).skew()
            
            # Create column name for return skewness values
            col_name = f"{symbol} | return_skewness_{window_size}"
            result_df[col_name] = return_skew.astype(np.float32)
            
        except Exception as e:
            log.error(f"Error calculating return skewness for {close_col}: {str(e)}")
            continue
    
    return result_df

def return_kurtosis(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate return kurtosis for all trading pairs.
    
    Return kurtosis measures the "tailedness" of log returns distribution over a rolling window.
    Higher kurtosis indicates more extreme moves and "fat tails", lower kurtosis indicates more normal distribution.
    
    Args:
        dataframe: Input dataframe with time index and columns in format "{symbol} | close"
        config: Dictionary containing configuration parameters
            - window_size: Window size for kurtosis calculation (default: 60)
        
    Returns:
        pd.DataFrame: DataFrame with return kurtosis columns for each pair
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Extract configuration
    window_size = config.get('window_size', 60)  # Default to 60 if not specified
    
    # Find all close columns (format: "{symbol} | close")
    close_columns = [col for col in dataframe.columns if '| close' in col]
    
    if not close_columns:
        log.warning("⚠️  No columns containing '| close' found for return kurtosis calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate return kurtosis for each close column
    for close_col in close_columns:
        try:
            # Extract symbol from close column name (e.g., "audusd | close" -> "audusd")
            symbol = close_col.split(' | ')[0]
            close = dataframe[close_col]
            
            # Calculate log returns
            log_returns = np.log(close.where(close > 0, np.nan)).diff()
            
            # Calculate rolling kurtosis of log returns
            return_kurt = log_returns.rolling(window=window_size, min_periods=window_size).kurt()
            
            # Create column name for return kurtosis values
            col_name = f"{symbol} | return_kurtosis_{window_size}"
            result_df[col_name] = return_kurt.astype(np.float32)
            
        except Exception as e:
            log.error(f"Error calculating return kurtosis for {close_col}: {str(e)}")
            continue
    
    return result_df

def csi(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate Currency Strength Index (CSI) for all currencies.
    
    CSI measures the relative strength of each currency against all other currencies.
    It normalizes each currency pair's close price and aggregates the strength based on
    whether the currency is the base or quote currency in each pair.
    
    Args:
        dataframe: Input dataframe with time index and columns in format "{symbol} | close"
        config: Dictionary containing configuration parameters
            - window_size: Window size for rolling normalization (default: 20160)
        
    Returns:
        pd.DataFrame: DataFrame with CSI columns for each currency
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Extract configuration
    window_size = config.get('window_size', 14)  # Default to 20160 if not specified
    
    # Find all close columns for pairs only (format: "{symbol}-pair | close")
    close_columns = [col for col in dataframe.columns if '| close' in col and '-pair' in col]
    
    if not close_columns:
        log.warning("⚠️  No pair columns containing '| close' found for CSI calculation")
        return pd.DataFrame()
    
    # Extract currency pairs and currencies from column names
    pairs = []
    currencies = set()
    
    for close_col in close_columns:
        # Extract symbol from close column name (e.g., "audusd-pair | close" -> "audusd")
        symbol_with_suffix = close_col.split(' | ')[0].lower()
        symbol = symbol_with_suffix[:-5]
        
        # Only process 6-character currency pairs (3+3 format)
        if len(symbol) == 6:
            base_currency = symbol[:3].upper()
            quote_currency = symbol[3:].upper()
            pairs.append(symbol)
            currencies.add(base_currency)
            currencies.add(quote_currency)
        else:
            log.warning(f"⚠️  Skipping {symbol}: not a valid 6-character currency pair")
    
    if not pairs or not currencies:
        log.warning("⚠️  No valid currency pairs found for CSI calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate CSI for each currency
    for currency in currencies:
        # Initialize CSI
        csi_values = pd.Series(0.0, index=dataframe.index, dtype=np.float32)
        count = 0
        
        for pair in pairs:
            # Look for pair column specifically
            pair_col = f"{pair}-pair | close"
            
            if pair_col not in dataframe.columns:
                continue
            
            # Normalize close price in rolling window
            close = dataframe[pair_col]
            rolling_mean = close.rolling(window=window_size, min_periods=1).mean()
            rolling_std = close.rolling(window=window_size, min_periods=1).std()
            normalized = (close - rolling_mean) / (rolling_std + 1e-6)
            
            # Adjust based on currency position in pair
            base, quote = pair[:3].upper(), pair[3:].upper()
            if currency.upper() == base:
                csi_values += normalized
                count += 1
            elif currency.upper() == quote:
                csi_values -= normalized
                count += 1
        
        # Scale to 0-100
        if count > 0:
            csi_values = csi_values / count
            result_df[f'csi_{currency.lower()}'] = csi_values.astype(np.float32)
        else:
            result_df[f'csi_{currency.lower()}'] = np.nan
    
    return result_df


def minute_of_day(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate minute of day features with cyclical encoding.
    
    Creates sine and cosine components for the minute of day to capture cyclical patterns.
    This is useful for identifying intraday trading patterns and market session effects.
    
    Args:
        dataframe: Input dataframe with time index
        config: Dictionary containing configuration parameters (optional)
        
    Returns:
        pd.DataFrame: DataFrame with minute_of_day_sin and minute_of_day_cos columns
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Check if index is datetime
    if not isinstance(dataframe.index, pd.DatetimeIndex):
        log.error("❌ Index must be a DatetimeIndex for minute_of_day feature")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate minute of day (0-1439 for 24 hours * 60 minutes)
    minute_of_day_raw = dataframe.index.hour * 60 + dataframe.index.minute
    
    # Convert to cyclical features using sine and cosine
    # Use 1440 (24*60) as the period for full day cycle
    minute_of_day_sin = np.sin(2 * np.pi * minute_of_day_raw / 1440)
    minute_of_day_cos = np.cos(2 * np.pi * minute_of_day_raw / 1440)
    
    # Add to result dataframe
    result_df['minute_of_day_sin'] = minute_of_day_sin.astype(np.float32)
    result_df['minute_of_day_cos'] = minute_of_day_cos.astype(np.float32)
    
    return result_df


def day_of_week(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate day of week features with cyclical encoding.
    
    Creates sine and cosine components for the day of week to capture cyclical patterns.
    This is useful for identifying weekly trading patterns and weekend effects.
    
    Args:
        dataframe: Input dataframe with time index
        config: Dictionary containing configuration parameters (optional)
        
    Returns:
        pd.DataFrame: DataFrame with day_of_week_sin and day_of_week_cos columns
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Check if index is datetime
    if not isinstance(dataframe.index, pd.DatetimeIndex):
        log.error("❌ Index must be a DatetimeIndex for day_of_week feature")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate day of week (0=Monday, 6=Sunday)
    day_of_week_raw = dataframe.index.dayofweek
    
    # Convert to cyclical features using sine and cosine
    # Use 7 as the period for full week cycle
    day_of_week_sin = np.sin(2 * np.pi * day_of_week_raw / 7)
    day_of_week_cos = np.cos(2 * np.pi * day_of_week_raw / 7)
    
    # Add to result dataframe
    result_df['day_of_week_sin'] = day_of_week_sin.astype(np.float32)
    result_df['day_of_week_cos'] = day_of_week_cos.astype(np.float32)
    
    return result_df


def week_of_month(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate week of month features with cyclical encoding.
    
    Creates sine and cosine components for the week of month to capture cyclical patterns.
    This is useful for identifying monthly patterns and weekly cycles within months.
    
    Args:
        dataframe: Input dataframe with time index
        config: Dictionary containing configuration parameters (optional)
        
    Returns:
        pd.DataFrame: DataFrame with week_of_month_sin and week_of_month_cos columns
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Check if index is datetime
    if not isinstance(dataframe.index, pd.DatetimeIndex):
        log.error("❌ Index must be a DatetimeIndex for week_of_month feature")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate week of month (1-5, where 1 is the first week of the month)
    # Day of month divided by 7, rounded up, gives us the week number
    week_of_month_raw = np.ceil(dataframe.index.day / 7).astype(int)
    
    # Convert to cyclical features using sine and cosine
    # Use 5 as the period for full month cycle (maximum weeks in a month)
    week_of_month_sin = np.sin(2 * np.pi * week_of_month_raw / 5)
    week_of_month_cos = np.cos(2 * np.pi * week_of_month_raw / 5)
    
    # Add to result dataframe
    result_df['week_of_month_sin'] = week_of_month_sin.astype(np.float32)
    result_df['week_of_month_cos'] = week_of_month_cos.astype(np.float32)
    
    return result_df


def month_of_year(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate month of year features with cyclical encoding.
    
    Creates sine and cosine components for the month of year to capture cyclical patterns.
    This is useful for identifying seasonal patterns and monthly trading cycles.
    
    Args:
        dataframe: Input dataframe with time index
        config: Dictionary containing configuration parameters (optional)
        
    Returns:
        pd.DataFrame: DataFrame with month_of_year_sin and month_of_year_cos columns
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Check if index is datetime
    if not isinstance(dataframe.index, pd.DatetimeIndex):
        log.error("❌ Index must be a DatetimeIndex for month_of_year feature")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate month of year (1-12)
    month_of_year_raw = dataframe.index.month
    
    # Convert to cyclical features using sine and cosine
    # Use 12 as the period for full year cycle
    month_of_year_sin = np.sin(2 * np.pi * month_of_year_raw / 12)
    month_of_year_cos = np.cos(2 * np.pi * month_of_year_raw / 12)
    
    # Add to result dataframe
    result_df['month_of_year_sin'] = month_of_year_sin.astype(np.float32)
    result_df['month_of_year_cos'] = month_of_year_cos.astype(np.float32)
    
    return result_df


def pair_correlations(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate rolling correlations between specified pairs.
    
    Creates correlation features for pairs of currency pairs over a specified window.
    This is useful for identifying relationships and co-movements between different currency pairs.
    
    Args:
        dataframe: Input dataframe with time index and columns in format "{symbol}-pair | close"
        config: Dictionary containing configuration parameters
            - window_size: Window size for rolling correlation calculation (default: 30)
            - pairs: List of pair combinations to calculate correlations for
                    Format: [["pair1", "pair2"], ["pair1", "pair3"], ...]
        
    Returns:
        pd.DataFrame: DataFrame with correlation columns for each pair combination
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Check if index is datetime
    if not isinstance(dataframe.index, pd.DatetimeIndex):
        log.error("❌ Index must be a DatetimeIndex for pair_correlations feature")
        return pd.DataFrame()
    
    # Extract configuration
    window_size = config.get('window_size', 30)
    pairs_config = config.get('pairs', [])
    
    if not pairs_config:
        log.warning("⚠️  No pairs specified for correlation calculation")
        return pd.DataFrame()
    
    # Find all close columns for pairs
    close_columns = [col for col in dataframe.columns if '| close' in col and '-pair' in col]
    
    if not close_columns:
        log.warning("⚠️  No pair close columns found for correlation calculation")
        return pd.DataFrame()
    
    # Extract available pairs from column names
    available_pairs = {}
    for close_col in close_columns:
        # Extract symbol from close column name (e.g., "audusd-pair | close" -> "audusd")
        symbol_with_suffix = close_col.split(' | ')[0].lower()
        if symbol_with_suffix.endswith('-pair'):
            symbol = symbol_with_suffix[:-5]  # Remove '-pair' suffix
            available_pairs[symbol] = close_col
    
    if not available_pairs:
        log.warning("⚠️  No valid pairs found for correlation calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Process each pair combination
    for pair_combo in pairs_config:
        if len(pair_combo) != 2:
            log.warning(f"⚠️  Skipping invalid pair combination: {pair_combo} (expected 2 pairs)")
            continue
        
        pair1, pair2 = pair_combo[0].lower(), pair_combo[1].lower()
        
        # Check if both pairs are available
        if pair1 not in available_pairs or pair2 not in available_pairs:
            log.warning(f"⚠️  Skipping {pair1}-{pair2}: one or both pairs not found in data")
            continue
        
        # Get the close price columns for both pairs
        col1 = available_pairs[pair1]
        col2 = available_pairs[pair2]
        
        try:
            # Calculate rolling correlation using vectorized operations
            # Use pandas rolling correlation for efficiency
            correlation = dataframe[col1].rolling(
                window=window_size, 
                min_periods=1
            ).corr(dataframe[col2])
            
            # Create column name
            col_name = f"{pair1}_{pair2}_correlation_{window_size}"
            result_df[col_name] = correlation.astype(np.float32)
            
        except Exception as e:
            log.error(f"❌ Error calculating correlation for {pair1}-{pair2}: {str(e)}")
            continue
    
    return result_df


