
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

    # Standardize column names
    if 'Volume' in df.columns:
        df = df.rename(columns={'Volume': 'volume'})
        logger.info("Renamed 'Volume' to 'volume'.")

    # Handle missing values - Forward Fill as per original script
    # FutureWarning: DataFrame.fillna with 'method' is deprecated
    df = df.ffill()
    
    # Check for remaining NaNs
    if df.isnull().sum().sum() > 0:
        logger.warning(f"Remaining NaNs after ffill:\n{df.isnull().sum()}")
        df = df.dropna()
        logger.info("Dropped rows with remaining NaNs.")

    logger.info("Data cleaning completed.")
    return df
