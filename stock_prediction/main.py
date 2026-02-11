
import argparse
import sys
from pathlib import Path

# Add project root to python path to allow imports
# This is needed because we are running from inside the package structure without installing it
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

from stock_prediction.src.pipeline import run_pipeline
from stock_prediction.config import settings

def main():
    parser = argparse.ArgumentParser(description="Stock Market Analysis and Prediction Pipeline")
    parser.add_argument(
        "--data_path", 
        type=str, 
        help="Path to the input CSV data file",
        default=str(settings.RAW_DATA_DIR / "NSE_HDFCBANK15.csv") # Default to known file
    )
    
    parser.add_argument(
        "--timeframe", 
        type=str, 
        help="Pandas resampling frequency (e.g., '15T', '1H', '1D')",
        default=settings.TIMEFRAME
    )
    
    args = parser.parse_args()
    
    try:
        run_pipeline(args.data_path, timeframe=args.timeframe)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
