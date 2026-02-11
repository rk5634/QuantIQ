
import re
import pandas as pd
from stock_prediction.src.utils import setup_logger

logger = setup_logger("preprocessing")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the data.
    
    Args:
        df (pd.DataFrame): Raw DataFrame.
        
    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    logger.info("Starting data cleaning...")
    
    # Check for duplicates
    initial_rows = len(df)
    df = df.drop_duplicates()
    if len(df) < initial_rows:
        logger.info(f"Dropped {initial_rows - len(df)} duplicate rows.")
    
    # Sort by time
    if 'time' in df.columns:
        df = df.sort_values('time').reset_index(drop=True)
    else:
        logger.warning("'time' column not found. Skipping sort.")
        logger.warning("'time' column not found. Skipping sort and duplicate check by time.")

    # Handle missing values - Forward Fill then Backward Fill for safety
    df = df.ffill().bfill() # Chain for safety
    
    # Check for remaining NaNs (should be none after ffill().bfill() unless entire columns are NaN)
    if df.isnull().sum().sum() > 0:
        logger.warning(f"Remaining NaNs after ffill().bfill():\n{df.isnull().sum()}")
        df = df.dropna()
        logger.info("Dropped rows with remaining NaNs.")

    logger.info("Data cleaning completed.")
    return df

def resample_data(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample data to a different timeframe.
    Supports user-friendly aliases: '5m', '1h', '1d', '1w'.
    
    Args:
        df: DataFrame with 'time' index or column.
        timeframe: User provided timeframe string.
        
    Returns:
        Resampled DataFrame.
    """
    if not timeframe:
        return df
        
    # Parse timeframe string using regex to separate number and unit
    # e.g. "5m" -> 5, "m"
    # "1hour" -> 1, "hour"
    
    match = re.match(r"(\d+)?([a-zA-Z]+)", timeframe)
    if match:
        number = match.group(1) if match.group(1) else "1" # Default to 1 if no number (e.g. "h")
        unit = match.group(2).lower()
        
        # Map to Pandas Alias
        if unit in ['m', 'min', 'mins', 'minute', 'minutes']:
            pandas_unit = 'min'
        elif unit in ['h', 'hr', 'hrs', 'hour', 'hours']:
            pandas_unit = 'H'
        elif unit in ['d', 'day', 'days']:
            pandas_unit = 'D'
        elif unit in ['w', 'wk', 'week', 'weeks']:
            pandas_unit = 'W'
        elif unit in ['mo', 'month', 'months']:
            pandas_unit = 'ME' # Month End
        else:
            pandas_unit = unit # Fallback (e.g. 'S', 'ms')
            
        timeframe = f"{number}{pandas_unit}"
        logger.info(f"Normalized timeframe '{match.group(0)}' to Pandas alias '{timeframe}'")
    
    logger.info(f"Resampling data to {timeframe}...")
    
    # Ensure time is index
    if 'time' in df.columns:
        df = df.set_index('time')
        
    # Resample Logic
    # Open: first, High: max, Low: min, Close: last, Volume: sum
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    # If using senior trader features, might need ADX/ATR aggregation? 
    # Usually easier to recalculate indicators on resampled candles.
    # So we just aggregate Price/Volume here.
    
    df_resampled = df.resample(timeframe).agg(agg_dict)
    
    # Drop rows with NaNs (e.g. gaps in trading hours)
    df_resampled = df_resampled.dropna()
    
    # Reset index to make 'time' a column again
    df_resampled = df_resampled.reset_index()
    
    logger.info(f"Resampling complete. New shape: {df_resampled.shape}")
    return df_resampled
