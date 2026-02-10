
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
        # Parse 'time' column as dates
        df = pd.read_csv(filepath, parse_dates=['time'])
        logger.info(f"Successfully loaded {len(df)} rows.")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise e
