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

def rolling_mean(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate rolling mean for all numeric columns.
    
    Args:
        dataframe: Input dataframe with time index
        config: Dictionary containing configuration parameters
            - window_size: Size of the rolling window
        
    Returns:
        pd.DataFrame: DataFrame with rolling mean columns (original columns with '_rolling_mean' suffix)
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Extract configuration
    window_size = config.get('window_size', 12)  # Default to 12 if not specified
    
    # Get numeric columns only
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) == 0:
        log.warning("⚠️  No numeric columns found for rolling mean calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    for col in numeric_columns:
        result_df[f'{col}_rolling_mean'] = dataframe[col].rolling(window=window_size, min_periods=1).mean()
    
    return result_df

def rolling_std(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate rolling standard deviation for all numeric columns.
    
    Args:
        dataframe: Input dataframe with time index
        config: Dictionary containing configuration parameters
            - window_size: Size of the rolling window
        
    Returns:
        pd.DataFrame: DataFrame with rolling std columns (original columns with '_rolling_std' suffix)
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Extract configuration
    window_size = config.get('window_size', 12)  # Default to 12 if not specified
    
    # Get numeric columns only
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) == 0:
        log.warning("⚠️  No numeric columns found for rolling std calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    for col in numeric_columns:
        result_df[f'{col}_rolling_std'] = dataframe[col].rolling(window=window_size, min_periods=1).std()
    
    return result_df

def rolling_min(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate rolling minimum for all numeric columns.
    
    Args:
        dataframe: Input dataframe with time index
        window_size: Size of the rolling window
        
    Returns:
        pd.DataFrame: DataFrame with rolling min columns (original columns with '_rolling_min' suffix)
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Get numeric columns only
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) == 0:
        log.warning("⚠️  No numeric columns found for rolling min calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    for col in numeric_columns:
        result_df[f'{col}_rolling_min'] = dataframe[col].rolling(window=window_size, min_periods=1).min()
    
    return result_df

def rolling_max(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate rolling maximum for all numeric columns.
    
    Args:
        dataframe: Input dataframe with time index
        window_size: Size of the rolling window
        
    Returns:
        pd.DataFrame: DataFrame with rolling max columns (original columns with '_rolling_max' suffix)
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Get numeric columns only
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) == 0:
        log.warning("⚠️  No numeric columns found for rolling max calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    for col in numeric_columns:
        result_df[f'{col}_rolling_max'] = dataframe[col].rolling(window=window_size, min_periods=1).max()
    
    return result_df

def rolling_median(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate rolling median for all numeric columns.
    
    Args:
        dataframe: Input dataframe with time index
        window_size: Size of the rolling window
        
    Returns:
        pd.DataFrame: DataFrame with rolling median columns (original columns with '_rolling_median' suffix)
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Get numeric columns only
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) == 0:
        log.warning("⚠️  No numeric columns found for rolling median calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    for col in numeric_columns:
        result_df[f'{col}_rolling_median'] = dataframe[col].rolling(window=window_size, min_periods=1).median()
    
    return result_df

def rolling_skew(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate rolling skewness for all numeric columns.
    
    Args:
        dataframe: Input dataframe with time index
        window_size: Size of the rolling window
        
    Returns:
        pd.DataFrame: DataFrame with rolling skew columns (original columns with '_rolling_skew' suffix)
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Get numeric columns only
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) == 0:
        log.warning("⚠️  No numeric columns found for rolling skew calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    for col in numeric_columns:
        result_df[f'{col}_rolling_skew'] = dataframe[col].rolling(window=window_size, min_periods=1).skew()
    
    return result_df

def rolling_kurt(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate rolling kurtosis for all numeric columns.
    
    Args:
        dataframe: Input dataframe with time index
        window_size: Size of the rolling window
        
    Returns:
        pd.DataFrame: DataFrame with rolling kurtosis columns (original columns with '_rolling_kurt' suffix)
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Get numeric columns only
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) == 0:
        log.warning("⚠️  No numeric columns found for rolling kurtosis calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    for col in numeric_columns:
        result_df[f'{col}_rolling_kurt'] = dataframe[col].rolling(window=window_size, min_periods=1).kurt()
    
    return result_df

def rolling_var(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate rolling variance for all numeric columns.
    
    Args:
        dataframe: Input dataframe with time index
        window_size: Size of the rolling window
        
    Returns:
        pd.DataFrame: DataFrame with rolling variance columns (original columns with '_rolling_var' suffix)
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Get numeric columns only
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) == 0:
        log.warning("⚠️  No numeric columns found for rolling variance calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    for col in numeric_columns:
        result_df[f'{col}_rolling_var'] = dataframe[col].rolling(window=window_size, min_periods=1).var()
    
    return result_df

def rolling_quantile_25(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate rolling 25th percentile for all numeric columns.
    
    Args:
        dataframe: Input dataframe with time index
        window_size: Size of the rolling window
        
    Returns:
        pd.DataFrame: DataFrame with rolling 25th percentile columns (original columns with '_rolling_q25' suffix)
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Get numeric columns only
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) == 0:
        log.warning("⚠️  No numeric columns found for rolling 25th percentile calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    for col in numeric_columns:
        result_df[f'{col}_rolling_q25'] = dataframe[col].rolling(window=window_size, min_periods=1).quantile(0.25)
    
    return result_df

def rolling_quantile_75(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate rolling 75th percentile for all numeric columns.
    
    Args:
        dataframe: Input dataframe with time index
        window_size: Size of the rolling window
        
    Returns:
        pd.DataFrame: DataFrame with rolling 75th percentile columns (original columns with '_rolling_q75' suffix)
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Get numeric columns only
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) == 0:
        log.warning("⚠️  No numeric columns found for rolling 75th percentile calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    for col in numeric_columns:
        result_df[f'{col}_rolling_q75'] = dataframe[col].rolling(window=window_size, min_periods=1).quantile(0.75)
    
    return result_df 

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
        result_df[f'{col} | log_pct_change'] = np.log(dataframe[col].replace(0, np.nan)).diff()
    
    return result_df

def rsi_indicators(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate RSI indicators for all columns ending with '_close'.
    
    Args:
        dataframe: Input dataframe with time index and columns ending with '_close'
        config: Dictionary containing configuration parameters
            - window_size: Window size for RSI calculations
        
    Returns:
        pd.DataFrame: DataFrame with RSI indicators
    """
    if dataframe.empty:
        log.warning("⚠️  Empty dataframe provided for RSI calculation")
        return pd.DataFrame()
    
    # Extract configuration
    window_size = config.get('window_size', 14)  # Default to 14 if not specified
    
    # Find all columns ending with '_close'
    close_columns = [col for col in dataframe.columns if col.endswith('_close')]
    
    if not close_columns:
        log.warning("⚠️  No columns ending with '_close' found for RSI calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate RSI for each close column
    for col in close_columns:
        close = dataframe[col]
        result_df[f'{col}_rsi_{window_size}'] = ta.rsi(close, length=window_size)
    
    return result_df

def macd_indicators(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate MACD indicators for all columns ending with '_close'.
    
    Args:
        dataframe: Input dataframe with time index and columns ending with '_close'
        config: Dictionary containing configuration parameters (optional)
        
    Returns:
        pd.DataFrame: DataFrame with MACD indicators
    """
    if dataframe.empty:
        log.warning("⚠️  Empty dataframe provided for MACD calculation")
        return pd.DataFrame()
    
    # Find all columns ending with '_close'
    close_columns = [col for col in dataframe.columns if col.endswith('_close')]
    
    if not close_columns:
        log.warning("⚠️  No columns ending with '_close' found for MACD calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate MACD for each close column
    for col in close_columns:
        close = dataframe[col]
        macd = ta.macd(close)
        result_df[f'{col}_macd'] = macd['MACD_12_26_9']
        result_df[f'{col}_macd_signal'] = macd['MACDs_12_26_9']
        result_df[f'{col}_macd_diff'] = macd['MACDh_12_26_9']
    
    return result_df

def ema_indicators(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate EMA indicators for all columns ending with '_close'.
    
    Args:
        dataframe: Input dataframe with time index and columns ending with '_close'
        window_size: Window size for EMA calculation
        
    Returns:
        pd.DataFrame: DataFrame with EMA indicators
    """
    if dataframe.empty:
        log.warning("⚠️  Empty dataframe provided for EMA calculation")
        return pd.DataFrame()
    
    # Find all columns ending with '_close'
    close_columns = [col for col in dataframe.columns if col.endswith('_close')]
    
    if not close_columns:
        log.warning("⚠️  No columns ending with '_close' found for EMA calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate EMA for each close column
    for col in close_columns:
        close = dataframe[col]
        result_df[f'{col}_ema_{window_size}'] = ta.ema(close, length=window_size)
    
    return result_df

def atr_indicators(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate ATR indicators for OHLC data.
    
    Args:
        dataframe: Input dataframe with time index and OHLC columns
        window_size: Window size for ATR calculation
        
    Returns:
        pd.DataFrame: DataFrame with ATR indicators
    """
    required_cols = ['high', 'low', 'close']
    if dataframe.empty or not all(col in dataframe.columns for col in required_cols):
        log.warning("⚠️  Missing required OHLC columns for ATR calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    high = dataframe['high']
    low = dataframe['low']
    close = dataframe['close']
    
    # Calculate ATR with specified window
    result_df['atr'] = ta.atr(high, low, close, length=window_size)
    
    return result_df

def bollinger_bands(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate Bollinger Bands for all columns ending with '_close'.
    
    Args:
        dataframe: Input dataframe with time index and columns ending with '_close'
        window_size: Window size for Bollinger Bands calculation
        
    Returns:
        pd.DataFrame: DataFrame with Bollinger Bands indicators
    """
    if dataframe.empty:
        log.warning("⚠️  Empty dataframe provided for Bollinger Bands calculation")
        return pd.DataFrame()
    
    # Find all columns ending with '_close'
    close_columns = [col for col in dataframe.columns if col.endswith('_close')]
    
    if not close_columns:
        log.warning("⚠️  No columns ending with '_close' found for Bollinger Bands calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate Bollinger Bands for each close column
    for col in close_columns:
        close = dataframe[col]
        bb = ta.bbands(close, length=window_size)
        result_df[f'{col}_bb_upper'] = bb['BBU_20_2.0']
        result_df[f'{col}_bb_lower'] = bb['BBL_20_2.0']
        result_df[f'{col}_bb_width'] = bb['BBW_20_2.0']
    
    return result_df

def adx_indicator(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate ADX indicator for OHLC data.
    
    Args:
        dataframe: Input dataframe with time index and OHLC columns
        window_size: Window size for ADX calculation
        
    Returns:
        pd.DataFrame: DataFrame with ADX indicator
    """
    required_cols = ['high', 'low', 'close']
    if dataframe.empty or not all(col in dataframe.columns for col in required_cols):
        log.warning("⚠️  Missing required OHLC columns for ADX calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    high = dataframe['high']
    low = dataframe['low']
    close = dataframe['close']
    
    # Calculate ADX with specified window
    adx = ta.adx(high, low, close, length=window_size)
    result_df['adx'] = adx['ADX_14']
    
    return result_df

def cci_indicator(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate CCI indicator for OHLC data.
    
    Args:
        dataframe: Input dataframe with time index and OHLC columns
        window_size: Window size for CCI calculation
        
    Returns:
        pd.DataFrame: DataFrame with CCI indicator
    """
    required_cols = ['high', 'low', 'close']
    if dataframe.empty or not all(col in dataframe.columns for col in required_cols):
        log.warning("⚠️  Missing required OHLC columns for CCI calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    high = dataframe['high']
    low = dataframe['low']
    close = dataframe['close']
    
    # Calculate CCI with specified window
    result_df['cci'] = ta.cci(high, low, close, length=window_size)
    
    return result_df

def stochastic_oscillator(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate Stochastic Oscillator for OHLC data.
    
    Args:
        dataframe: Input dataframe with time index and OHLC columns
        window_size: Not used for Stochastic (uses default parameters)
        
    Returns:
        pd.DataFrame: DataFrame with Stochastic Oscillator indicators
    """
    required_cols = ['high', 'low', 'close']
    if dataframe.empty or not all(col in dataframe.columns for col in required_cols):
        log.warning("⚠️  Missing required OHLC columns for Stochastic Oscillator calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    high = dataframe['high']
    low = dataframe['low']
    close = dataframe['close']
    
    # Calculate Stochastic Oscillator
    stoch = ta.stoch(high, low, close)
    result_df['stoch_k'] = stoch['STOCHk_14_3_3']
    result_df['stoch_d'] = stoch['STOCHd_14_3_3']
    
    return result_df

def williams_r(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate Williams %R indicator for OHLC data.
    
    Args:
        dataframe: Input dataframe with time index and OHLC columns
        window_size: Window size for Williams %R calculation
        
    Returns:
        pd.DataFrame: DataFrame with Williams %R indicator
    """
    required_cols = ['high', 'low', 'close']
    if dataframe.empty or not all(col in dataframe.columns for col in required_cols):
        log.warning("⚠️  Missing required OHLC columns for Williams %R calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    high = dataframe['high']
    low = dataframe['low']
    close = dataframe['close']
    
    # Calculate Williams %R with specified window
    result_df['willr'] = ta.willr(high, low, close, length=window_size)
    
    return result_df

def roc_indicator(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate Rate of Change indicator for all columns ending with '_close'.
    
    Args:
        dataframe: Input dataframe with time index and columns ending with '_close'
        window_size: Window size for ROC calculation
        
    Returns:
        pd.DataFrame: DataFrame with ROC indicator
    """
    if dataframe.empty:
        log.warning("⚠️  Empty dataframe provided for ROC calculation")
        return pd.DataFrame()
    
    # Find all columns ending with '_close'
    close_columns = [col for col in dataframe.columns if col.endswith('_close')]
    
    if not close_columns:
        log.warning("⚠️  No columns ending with '_close' found for ROC calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate ROC for each close column
    for col in close_columns:
        close = dataframe[col]
        result_df[f'{col}_roc'] = ta.roc(close, length=window_size)
    
    return result_df

def chaikin_money_flow(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate Chaikin Money Flow indicator for OHLCV data.
    
    Args:
        dataframe: Input dataframe with time index and OHLCV columns
        window_size: Window size for CMF calculation
        
    Returns:
        pd.DataFrame: DataFrame with CMF indicator
    """
    required_cols = ['high', 'low', 'close', 'volume']
    if dataframe.empty or not all(col in dataframe.columns for col in required_cols):
        log.warning("⚠️  Missing required OHLCV columns for CMF calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    high = dataframe['high']
    low = dataframe['low']
    close = dataframe['close']
    volume = dataframe['volume']
    
    # Calculate CMF with specified window
    result_df['cmf'] = ta.cmf(high, low, close, volume, length=window_size)
    
    return result_df

def ichimoku_indicators(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate Ichimoku indicators for OHLC data.
    
    Args:
        dataframe: Input dataframe with time index and OHLC columns
        window_size: Not used for Ichimoku (uses default parameters)
        
    Returns:
        pd.DataFrame: DataFrame with Ichimoku indicators
    """
    required_cols = ['high', 'low', 'close']
    if dataframe.empty or not all(col in dataframe.columns for col in required_cols):
        log.warning("⚠️  Missing required OHLC columns for Ichimoku calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    high = dataframe['high']
    low = dataframe['low']
    close = dataframe['close']
    
    # Calculate Ichimoku indicators
    ichimoku = ta.ichimoku(high, low, close)
    result_df['ichimoku_tenkan'] = ichimoku['ITS_9']
    result_df['ichimoku_kijun'] = ichimoku['IKS_26']
    result_df['ichimoku_senkou_a'] = ichimoku['ISA_9']
    result_df['ichimoku_senkou_b'] = ichimoku['ISB_26']
    result_df['ichimoku_chikou'] = close.shift(-26)
    
    return result_df

def sma_indicators(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate SMA indicators for all columns ending with '_close'.
    
    Args:
        dataframe: Input dataframe with time index and columns ending with '_close'
        window_size: Window size for SMA calculation
        
    Returns:
        pd.DataFrame: DataFrame with SMA indicators
    """
    if dataframe.empty:
        log.warning("⚠️  Empty dataframe provided for SMA calculation")
        return pd.DataFrame()
    
    # Find all columns ending with '_close'
    close_columns = [col for col in dataframe.columns if col.endswith('_close')]
    
    if not close_columns:
        log.warning("⚠️  No columns ending with '_close' found for SMA calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate SMA for each close column
    for col in close_columns:
        close = dataframe[col]
        result_df[f'{col}_sma_{window_size}'] = ta.sma(close, length=window_size)
    
    return result_df

def keltner_channels(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate Keltner Channels for OHLC data.
    
    Args:
        dataframe: Input dataframe with time index and OHLC columns
        window_size: Window size for Keltner Channels calculation
        
    Returns:
        pd.DataFrame: DataFrame with Keltner Channels indicators
    """
    required_cols = ['high', 'low', 'close']
    if dataframe.empty or not all(col in dataframe.columns for col in required_cols):
        log.warning("⚠️  Missing required OHLC columns for Keltner Channels calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    high = dataframe['high']
    low = dataframe['low']
    close = dataframe['close']
    
    # Calculate Keltner Channels with specified window
    kc = ta.kc(high, low, close, length=window_size)
    result_df['kc_upper'] = kc['KCUe_20_2']
    result_df['kc_lower'] = kc['KCLe_20_2']
    
    return result_df

def ultimate_oscillator(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate Ultimate Oscillator for OHLC data.
    
    Args:
        dataframe: Input dataframe with time index and OHLC columns
        window_size: Not used for Ultimate Oscillator (uses default parameters)
        
    Returns:
        pd.DataFrame: DataFrame with Ultimate Oscillator indicator
    """
    required_cols = ['high', 'low', 'close']
    if dataframe.empty or not all(col in dataframe.columns for col in required_cols):
        log.warning("⚠️  Missing required OHLC columns for Ultimate Oscillator calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    high = dataframe['high']
    low = dataframe['low']
    close = dataframe['close']
    
    # Calculate Ultimate Oscillator
    result_df['ultimate_osc'] = ta.uo(high, low, close)
    
    return result_df

def dpo_indicator(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate Detrended Price Oscillator for all columns ending with '_close'.
    
    Args:
        dataframe: Input dataframe with time index and columns ending with '_close'
        window_size: Window size for DPO calculation
        
    Returns:
        pd.DataFrame: DataFrame with DPO indicator
    """
    if dataframe.empty:
        log.warning("⚠️  Empty dataframe provided for DPO calculation")
        return pd.DataFrame()
    
    # Find all columns ending with '_close'
    close_columns = [col for col in dataframe.columns if col.endswith('_close')]
    
    if not close_columns:
        log.warning("⚠️  No columns ending with '_close' found for DPO calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate DPO for each close column
    for col in close_columns:
        close = dataframe[col]
        result_df[f'{col}_dpo'] = ta.dpo(close, length=window_size)
    
    return result_df

def vwap(dataframe: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate Volume Weighted Average Price using 'volume' and 'mean' columns.
    
    Args:
        dataframe: Input dataframe with time index, 'volume' and 'mean' columns
        config: Dictionary containing configuration parameters
            - window_size: Window size for VWAP calculation
        
    Returns:
        pd.DataFrame: DataFrame with VWAP indicator
    """
    required_cols = ['volume', 'mean']
    if dataframe.empty or not all(col in dataframe.columns for col in required_cols):
        log.warning("⚠️  Missing required 'volume' or 'mean' columns for VWAP calculation")
        return pd.DataFrame()
    
    # Extract configuration
    window_size = config.get('window_size', 12)  # Default to 12 if not specified
    
    result_df = pd.DataFrame(index=dataframe.index)
    volume = dataframe['volume']
    price = dataframe['mean']
    
    # Calculate VWAP over the specified window
    # VWAP = Σ(Price × Volume) / Σ(Volume)
    price_volume = price * volume
    
    # Use rolling window to calculate VWAP
    rolling_pv_sum = price_volume.rolling(window=window_size, min_periods=1).sum()
    rolling_volume_sum = volume.rolling(window=window_size, min_periods=1).sum()
    
    # Calculate VWAP
    result_df['vwap'] = rolling_pv_sum / rolling_volume_sum
    
    return result_df

def historical_volatility(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate Historical Volatility for close price.
    
    Args:
        dataframe: Input dataframe with time index and 'close' column
        window_size: Window size for volatility calculation
        
    Returns:
        pd.DataFrame: DataFrame with Historical Volatility indicators
    """
    if dataframe.empty or 'close' not in dataframe.columns:
        log.warning("⚠️  No 'close' column found for Historical Volatility calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    close = dataframe['close']
    
    # Calculate Historical Volatility (60-period)
    returns = close.pct_change()
    result_df['hv_60'] = returns.rolling(60).std() * np.sqrt(252 * 1440)
    
    return result_df

def garman_klass_volatility(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate Garman-Klass Volatility for OHLC data.
    
    Args:
        dataframe: Input dataframe with time index and OHLC columns
        window_size: Not used for GK volatility (point-in-time calculation)
        
    Returns:
        pd.DataFrame: DataFrame with Garman-Klass Volatility indicator
    """
    required_cols = ['high', 'low', 'close', 'open']
    if dataframe.empty or not all(col in dataframe.columns for col in required_cols):
        log.warning("⚠️  Missing required OHLC columns for Garman-Klass Volatility calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    high = dataframe['high']
    low = dataframe['low']
    close = dataframe['close']
    open_price = dataframe['open']
    
    # Calculate Garman-Klass Volatility
    result_df['gk_vol'] = np.sqrt(0.5 * np.log(high/low)**2 - (2*np.log(2)-1) * np.log(close/open_price.fillna(close))**2)
    
    return result_df

def spread_volatility(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate Spread Volatility for spread data.
    
    Args:
        dataframe: Input dataframe with time index and 'spread_mean' column
        window_size: Window size for spread volatility calculation
        
    Returns:
        pd.DataFrame: DataFrame with Spread Volatility indicator
    """
    if dataframe.empty or 'spread_mean' not in dataframe.columns:
        log.warning("⚠️  No 'spread_mean' column found for Spread Volatility calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    spread_mean = dataframe['spread_mean']
    
    # Calculate Spread Volatility
    result_df['spread_volatility'] = spread_mean.rolling(60).std()
    
    return result_df

def trix_indicator(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate TRIX indicator for all columns ending with '_close'.
    
    Args:
        dataframe: Input dataframe with time index and columns ending with '_close'
        window_size: Window size for TRIX calculation
        
    Returns:
        pd.DataFrame: DataFrame with TRIX indicator
    """
    if dataframe.empty:
        log.warning("⚠️  Empty dataframe provided for TRIX calculation")
        return pd.DataFrame()
    
    # Find all columns ending with '_close'
    close_columns = [col for col in dataframe.columns if col.endswith('_close')]
    
    if not close_columns:
        log.warning("⚠️  No columns ending with '_close' found for TRIX calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate TRIX for each close column
    for col in close_columns:
        close = dataframe[col]
        result_df[f'{col}_trix'] = ta.trix(close, length=window_size)
    
    return result_df

def kst_indicator(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate KST indicator for close price.
    
    Args:
        dataframe: Input dataframe with time index and 'close' column
        window_size: Not used for KST (uses default parameters)
        
    Returns:
        pd.DataFrame: DataFrame with KST indicator
    """
    if dataframe.empty or 'close' not in dataframe.columns:
        log.warning("⚠️  No 'close' column found for KST calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    close = dataframe['close']
    
    # Calculate KST
    result_df['kst'] = ta.kst(close)
    
    return result_df

def rolling_skewness(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate rolling skewness for all columns ending with '_close'.
    
    Args:
        dataframe: Input dataframe with time index and columns ending with '_close'
        window_size: Window size for skewness calculation
        
    Returns:
        pd.DataFrame: DataFrame with rolling skewness indicator
    """
    if dataframe.empty:
        log.warning("⚠️  Empty dataframe provided for rolling skewness calculation")
        return pd.DataFrame()
    
    # Find all columns ending with '_close'
    close_columns = [col for col in dataframe.columns if col.endswith('_close')]
    
    if not close_columns:
        log.warning("⚠️  No columns ending with '_close' found for rolling skewness calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate rolling skewness for each close column
    for col in close_columns:
        close = dataframe[col]
        result_df[f'{col}_skew'] = close.pct_change().rolling(window_size).skew()
    
    return result_df

def rolling_kurtosis(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate rolling kurtosis for all columns ending with '_close'.
    
    Args:
        dataframe: Input dataframe with time index and columns ending with '_close'
        window_size: Window size for kurtosis calculation
        
    Returns:
        pd.DataFrame: DataFrame with rolling kurtosis indicator
    """
    if dataframe.empty:
        log.warning("⚠️  Empty dataframe provided for rolling kurtosis calculation")
        return pd.DataFrame()
    
    # Find all columns ending with '_close'
    close_columns = [col for col in dataframe.columns if col.endswith('_close')]
    
    if not close_columns:
        log.warning("⚠️  No columns ending with '_close' found for rolling kurtosis calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate rolling kurtosis for each close column
    for col in close_columns:
        close = dataframe[col]
        result_df[f'{col}_kurt'] = close.pct_change().rolling(window_size).kurt()
    
    return result_df

def zscore_indicator(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate Z-Score for all columns ending with '_close'.
    
    Args:
        dataframe: Input dataframe with time index and columns ending with '_close'
        window_size: Window size for Z-Score calculation
        
    Returns:
        pd.DataFrame: DataFrame with Z-Score indicator
    """
    if dataframe.empty:
        log.warning("⚠️  Empty dataframe provided for Z-Score calculation")
        return pd.DataFrame()
    
    # Find all columns ending with '_close'
    close_columns = [col for col in dataframe.columns if col.endswith('_close')]
    
    if not close_columns:
        log.warning("⚠️  No columns ending with '_close' found for Z-Score calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate Z-Score for each close column
    for col in close_columns:
        close = dataframe[col]
        result_df[f'{col}_zscore'] = (close - close.rolling(window_size).mean()) / close.rolling(window_size).std()
    
    return result_df

def price_acceleration(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate Price Acceleration (second derivative) for close price.
    
    Args:
        dataframe: Input dataframe with time index and 'close' column
        window_size: Not used for price acceleration (point-in-time calculation)
        
    Returns:
        pd.DataFrame: DataFrame with Price Acceleration indicator
    """
    if dataframe.empty or 'close' not in dataframe.columns:
        log.warning("⚠️  No 'close' column found for Price Acceleration calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    close = dataframe['close']
    
    # Calculate Price Acceleration
    result_df['price_accel'] = close.diff().diff()
    
    return result_df

def tick_direction(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate Tick Direction for close price.
    
    Args:
        dataframe: Input dataframe with time index and 'close' column
        window_size: Not used for tick direction (point-in-time calculation)
        
    Returns:
        pd.DataFrame: DataFrame with Tick Direction indicator
    """
    if dataframe.empty or 'close' not in dataframe.columns:
        log.warning("⚠️  No 'close' column found for Tick Direction calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    close = dataframe['close']
    
    # Calculate Tick Direction
    direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
    result_df['tick_direction'] = direction
    
    return result_df

def trend_persistence(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate Trend Persistence for all columns ending with '_close'.
    
    Args:
        dataframe: Input dataframe with time index and columns ending with '_close'
        window_size: Window size for trend persistence calculation
        
    Returns:
        pd.DataFrame: DataFrame with Trend Persistence indicator
    """
    if dataframe.empty:
        log.warning("⚠️  Empty dataframe provided for Trend Persistence calculation")
        return pd.DataFrame()
    
    # Find all columns ending with '_close'
    close_columns = [col for col in dataframe.columns if col.endswith('_close')]
    
    if not close_columns:
        log.warning("⚠️  No columns ending with '_close' found for Trend Persistence calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate Trend Persistence for each close column
    for col in close_columns:
        close = dataframe[col]
        direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
        trend_persistence = direction.rolling(window_size, min_periods=1).sum()
        result_df[f'{col}_trend_persistence'] = trend_persistence
    
    return result_df

def trend_direction(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate Trend Direction for all columns ending with '_close'.
    
    Args:
        dataframe: Input dataframe with time index and columns ending with '_close'
        window_size: Window size for trend direction calculation
        
    Returns:
        pd.DataFrame: DataFrame with Trend Direction indicator
    """
    if dataframe.empty:
        log.warning("⚠️  Empty dataframe provided for Trend Direction calculation")
        return pd.DataFrame()
    
    # Find all columns ending with '_close'
    close_columns = [col for col in dataframe.columns if col.endswith('_close')]
    
    if not close_columns:
        log.warning("⚠️  No columns ending with '_close' found for Trend Direction calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate Trend Direction for each close column
    for col in close_columns:
        close = dataframe[col]
        direction = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
        result_df[f'{col}_trend_direction'] = direction.rolling(window_size, min_periods=1).sum()
    
    return result_df

def realized_volatility(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate Realized Volatility for close price.
    
    Args:
        dataframe: Input dataframe with time index and 'close' column
        window_size: Not used for RV (uses fixed windows)
        
    Returns:
        pd.DataFrame: DataFrame with Realized Volatility indicators
    """
    if dataframe.empty or 'close' not in dataframe.columns:
        log.warning("⚠️  No 'close' column found for Realized Volatility calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    close = dataframe['close']
    
    # Calculate Realized Volatility
    returns = close.pct_change()
    result_df['rv_30'] = np.sqrt((returns**2).ewm(span=30, min_periods=15).mean()) * 5000
    result_df['rv_120'] = np.sqrt((returns**2).rolling(120, min_periods=30).sum()) * 1000
    
    return result_df

def future_sharpe(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate Sharpe ratio for future returns over a specified window.
    
    Args:
        dataframe: Input dataframe with time index and columns ending with '_close'
        window_size: Window size for future Sharpe ratio calculation
        
    Returns:
        pd.DataFrame: DataFrame with future Sharpe ratio indicators
    """
    if dataframe.empty:
        log.warning("⚠️  Empty dataframe provided for future Sharpe calculation")
        return pd.DataFrame()
    
    # Find all columns ending with '_close'
    close_columns = [col for col in dataframe.columns if col.endswith('_close')]
    
    if not close_columns:
        log.warning("⚠️  No columns ending with '_close' found for future Sharpe calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    # Calculate future Sharpe ratio for each close column
    for col in close_columns:
        close = dataframe[col]
        
        # Calculate future gains for the specified window
        # For each point, calculate gain from current price to future price (window_size periods ahead)
        future_prices = close.shift(-1)
        future_gains = future_prices / close
        
        # Calculate rolling statistics for Sharpe ratio
        # Use a longer window for stability (e.g., 252 periods)
        rolling_mean = future_gains.rolling(window=window_size).mean()
        rolling_std = future_gains.rolling(window=window_size).std()
        
        # Calculate Sharpe ratio: (Mean - 1) / Std (since gains have mean of 1 under no drift)
        sharpe = (rolling_mean - 1) / rolling_std
        
        # Annualize the Sharpe ratio (assuming daily data)
        result_df[f'{col}_future_sharpe_{window_size}'] = sharpe
    
    return result_df 