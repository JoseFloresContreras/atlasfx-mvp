"""
Augmentation functions for time-series data processing.
Each function follows the signature: function_name(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame
where:
- dataframe: pd.DataFrame - Input dataframe with time index
- window_size: int - Size of the rolling window

Each augmentation returns a DataFrame with the same time index as input, but may have NaN values
for indices where the rolling window cannot be fully formed.

This design supports both single-output and multi-output augmentations.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional

def rolling_mean(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate rolling mean for all numeric columns.
    
    Args:
        dataframe: Input dataframe with time index
        window_size: Size of the rolling window
        
    Returns:
        pd.DataFrame: DataFrame with rolling mean columns (original columns with '_rolling_mean' suffix)
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Get numeric columns only
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) == 0:
        print("⚠️  No numeric columns found for rolling mean calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    for col in numeric_columns:
        result_df[f'{col}_rolling_mean'] = dataframe[col].rolling(window=window_size, min_periods=1).mean()
    
    return result_df

def rolling_std(dataframe: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """
    Calculate rolling standard deviation for all numeric columns.
    
    Args:
        dataframe: Input dataframe with time index
        window_size: Size of the rolling window
        
    Returns:
        pd.DataFrame: DataFrame with rolling std columns (original columns with '_rolling_std' suffix)
    """
    if dataframe.empty:
        return pd.DataFrame()
    
    # Get numeric columns only
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns
    
    if len(numeric_columns) == 0:
        print("⚠️  No numeric columns found for rolling std calculation")
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
        print("⚠️  No numeric columns found for rolling min calculation")
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
        print("⚠️  No numeric columns found for rolling max calculation")
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
        print("⚠️  No numeric columns found for rolling median calculation")
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
        print("⚠️  No numeric columns found for rolling skew calculation")
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
        print("⚠️  No numeric columns found for rolling kurtosis calculation")
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
        print("⚠️  No numeric columns found for rolling variance calculation")
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
        print("⚠️  No numeric columns found for rolling 25th percentile calculation")
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
        print("⚠️  No numeric columns found for rolling 75th percentile calculation")
        return pd.DataFrame()
    
    result_df = pd.DataFrame(index=dataframe.index)
    
    for col in numeric_columns:
        result_df[f'{col}_rolling_q75'] = dataframe[col].rolling(window=window_size, min_periods=1).quantile(0.75)
    
    return result_df 