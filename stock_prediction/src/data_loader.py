
import pandas as pd
from pathlib import Path
from stock_prediction.src.utils import setup_logger

logger = setup_logger("data_loader")

def load_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        filepath (str | Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    
    Raises:
        FileNotFoundError: If the file is not found.
        Exception: For other loading errors.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        logger.info(f"Loading data from {filepath}")
        # Parse 'time' or 'date' column
        df = pd.read_csv(filepath)
        
        # Standardize column names
        df.columns = [c.lower() for c in df.columns]
        
        # Rename 'date' to 'time' if present
        if 'date' in df.columns:
            df.rename(columns={'date': 'time'}, inplace=True)
            
        # Ensure 'time' is datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        else:
            logger.error("No 'time' or 'date' column found in CSV.")
            raise ValueError("CSV must contain 'time' or 'date' column.")
            
        logger.info(f"Successfully loaded {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e
