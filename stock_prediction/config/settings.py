
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Log directory
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Plots directory
PLOTS_DIR = DATA_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Feature Engineering Settings
RSI_WINDOW = 14
RSI_WINDOW = 14
RSI_OVERSOLD = 40  # Stricter: Only buy deeper dips to increase Win Rate
RSI_OVERBOUGHT = 55 # Relaxed from 70 to allow more sell signals

MA_WINDOWS = [50, 200]
EMA_SPANS = [12, 26]

BOLLINGER_WINDOW = 20
BOLLINGER_DEV = 2

LAG_FEATURES = 3

# Trading Strategy Parameters
STOP_LOSS_PCT = 0.02
TAKE_PROFIT_PCT = 0.04 # Changed from 0.05

# Senior Trader Mode Parameters
ADX_PERIOD = 14
ADX_THRESHOLD = 25    # Trend Strength Floor
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0  # For Dynamic Stop Loss

# Sniper Strategy Parameters (Phase 10 & 11)
# Sniper Strategy Parameters (Phase 10 & 11)
# Sniper Strategy Parameters (Phase 10 & 11)
SNIPER_CONFIDENCE_THRESHOLD = 0.60 # Dip Buyer Mode: 60% Conf + RSI < 40
VOLUME_CONFIRMATION_THRESHOLD = 1.5 # Volume > 1.5x MA_20
META_LABELING_ENABLED = False # Placeholder for future

# Future Returns Labeling Settings
FUTURE_PERIOD = 5 # Number of candles to look ahead
RETURN_THRESHOLD = 0.001 # 0.1% gain threshold for Buy signal

# Model Settings
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Target Mapping
ACTION_MAPPING = {
    "BUY": 1,
    "SELL": 0, # Maintaining original logic: 0 for Sell/Hold based on analysis
    "HOLD": 0
}

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
