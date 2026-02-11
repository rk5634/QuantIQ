
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Data Processing
TIMEFRAME = None # Default: None (Use original data frequency). Examples: '15T', '1H', '1D'

# Log directory
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Plots directory
PLOTS_DIR = DATA_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Feature Engineering Settings
RSI_WINDOW = 14
# OPTIMIZED for DAILY SWING: RSI < 45 (Buy Pullbacks in Uptrend), not just deep crashes
RSI_OVERSOLD = 45  
RSI_OVERBOUGHT = 70 

MA_WINDOWS = [50, 200]
EMA_SPANS = [12, 26]

BOLLINGER_WINDOW = 20
BOLLINGER_DEV = 2

LAG_FEATURES = 3

# Trading Strategy Parameters
# OPTIMIZED for DAILY SWING: Wider stops, larger targets
STOP_LOSS_PCT = 0.05    # 5% SL (Daily candles have range)
TAKE_PROFIT_PCT = 0.10  # 10% TP (Aim for Swing moves)

# Senior Trader Mode Parameters
ADX_PERIOD = 14
ADX_THRESHOLD = 20    # Lowered to 20 to catch trend starts earlier
ATR_PERIOD = 14
ATR_MULTIPLIER = 2.0  # For Dynamic Stop Loss

# Indian Trader Mode
SUPERTREND_PERIOD = 7
SUPERTREND_MULTIPLIER = 3.0

# Sniper Strategy Parameters
SNIPER_CONFIDENCE_THRESHOLD = 0.55 # Slightly lower confidence needed if technicals align
VOLUME_CONFIRMATION_THRESHOLD = 1.0 # Volume > Avg (Not super strict 1.5x)
META_LABELING_ENABLED = False 

# Future Returns Labeling Settings
FUTURE_PERIOD = 5 # Number of candles to look ahead
RETURN_THRESHOLD = 0.001 # 0.1% gain threshold (Reverted to Phase 14 optimum)

# Model Settings
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Target Mapping
ACTION_MAPPING = {
    "BUY": 1,
    "SELL": 0, 
    "HOLD": 0
}

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
