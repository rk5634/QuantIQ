[]
import pandas as pd
import numpy as np
import ta
from stock_prediction.config import settings
from stock_prediction.src.utils import setup_logger

logger = setup_logger("features")


















def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the DataFrame.
    
    Args:
        df: DataFrame with 'close' column.
        
    Returns:
        DataFrame with added indicators.
    """
    logger.info("Adding technical indicators...")
    
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(close=df['close'], window=settings.RSI_WINDOW).rsi()
    
    # Moving Averages
    for window in settings.MA_WINDOWS:
        df[f'ma_{window}'] = df['close'].rolling(window=window).mean()
        
    # Volume Moving Average (for Sniper Strategy)
    df['vol_ma_20'] = df['volume'].rolling(window=20).mean()
        
    # EMAs
        
    # EMAs
    for span in settings.EMA_SPANS:
        df[f'ema_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
        
    # MACD
    macd = ta.trend.MACD(close=df['close'])
    df['macd_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(
        close=df['close'], 
        window=settings.BOLLINGER_WINDOW, 
        window_dev=settings.BOLLINGER_DEV
    )
    df['bollinger_mavg'] = bollinger.bollinger_mavg()
    df['bollinger_hband'] = bollinger.bollinger_hband()
    df['bollinger_lband'] = bollinger.bollinger_lband()
    
    # --- Senior Trader Indicators ---
    # ADX (Trend Strength)
    adx_ind = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=settings.ADX_PERIOD)
    df['adx'] = adx_ind.adx()
    
    # ATR (Volatility)
    atr_ind = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=settings.ATR_PERIOD)
    df['atr'] = atr_ind.average_true_range()
    
    # OBV (Volume Flow)
    obv_ind = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
    df['obv'] = obv_ind.on_balance_volume()
    # OBV Slope (Rate of change of OBV to normalize it)
    df['obv_slope'] = df['obv'].pct_change(5) # 5-period slope
    
    # Price Change
    
    # Price Change
    df['price_change_15m'] = df['close'].pct_change(1) * 100
    
    logger.info("Technical indicators added.")
    return df

def add_lag_features(df: pd.DataFrame, lags: int = settings.LAG_FEATURES) -> pd.DataFrame:
    """
    Add lag features for Open, High, Low, Close.
    
    Args:
        df: DataFrame.
        lags: Number of lags to create.
        
    Returns:
        DataFrame with lag features.
    """
    logger.info(f"Adding {lags} lag features...")
    cols_to_lag = ['open', 'high', 'low', 'close']
    
    for lag in range(1, lags + 1):
        for col in cols_to_lag:
            df[f'lag_{col}_{lag}'] = df[col].shift(lag)
            
    return df

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate Buy/Sell signals and targets.
    
    Args:
        df: DataFrame with indicators.
        
    Returns:
        DataFrame with signal columns.
    """
    logger.info("Generating trading signals...")
    
    # Buy Condition
    # RSI < 30 and Price > MA_50
    # Note: Using ma_50 as per original script logic
    if 'ma_50' not in df.columns:
        logger.warning("MA_50 not found via settings loops? Checking...")
        # Should be there if settings.MA_WINDOWS includes 50
        
    buy_condition = (df['rsi'] < settings.RSI_OVERSOLD) & (df['close'] > df['ma_50'])
    
    # Sell Condition
    # RSI > 70 OR Price < MA_50
    sell_condition = (df['rsi'] > settings.RSI_OVERBOUGHT) | (df['close'] < df['ma_50'])
    
    df['buy_signal'] = 0
    df['sell_signal'] = 0
    
    # Vectorized assignment
    df.loc[buy_condition, 'buy_signal'] = 1
    df.loc[sell_condition, 'sell_signal'] = 1
    
    # Stop Loss and Take Profit logic (Vectorized implementation suggested)
    # Original code iterates rows, which is slow.
    # Logic: If buy signal, set SL/TP based on Close price. Then ffill.
    
    df['stop_loss'] = np.nan
    df['take_profit'] = np.nan
    
    buy_indices = df[df['buy_signal'] == 1].index
    
    if not buy_indices.empty:
        buy_prices = df.loc[buy_indices, 'close']
        df.loc[buy_indices, 'stop_loss'] = buy_prices * (1 - settings.STOP_LOSS_PCT)
        df.loc[buy_indices, 'take_profit'] = buy_prices * (1 + settings.TAKE_PROFIT_PCT)
        
        # Forward fill as per original logic - this propagates the last buy signal's SL/TP
        df['stop_loss'] = df['stop_loss'].ffill()
        df['take_profit'] = df['take_profit'].ffill()
    else:
        logger.warning("No buy signals generated.")

    # Create Target 'action'
    # 1 = Buy, 0 = Hold/Sell as per simplified logic in original script analysis
    # "df.loc[df['buy_signal'] == 1, 'action'] = 1"
    # "df.loc[df['sell_signal'] == 1, 'action'] = 0"
    # initialized to 0.5 (Hold?)
    
    # Refined Logic based on Implementation Plan:
    # 1 = Buy
    # 0 = Hold (default)
    # -1 = Sell (if explicit sell signal needed, but models trained on binary mostly in original?)
    # Original script trained classifier on `action`. 
    # Let's map strictly using settings.ACTION_MAPPING
    
    # Future Returns Labeling Strategy
    # Calculate return over FUTURE_PERIOD
    # shift(-FUTURE_PERIOD) looks ahead
    future_close = df['close'].shift(-settings.FUTURE_PERIOD)
    df['future_return'] = (future_close - df['close']) / df['close']
    
    # Define Target
    # Buy (1) if return > threshold
    # Sell/Hold (0) otherwise
    df['action'] = settings.ACTION_MAPPING['HOLD']
    buy_mask = df['future_return'] > settings.RETURN_THRESHOLD
    df.loc[buy_mask, 'action'] = settings.ACTION_MAPPING['BUY']
    
    # Store legacy signals for reference/plotting but don't use for target
    df['buy_signal'] = 0
    df['sell_signal'] = 0
    df.loc[buy_condition, 'buy_signal'] = 1
    df.loc[sell_condition, 'sell_signal'] = 1
    
    logger.info(f"Generated targets based on {settings.FUTURE_PERIOD}-period future returns.")
    logger.info(f"Buy signals (Future): {df[df['action'] == settings.ACTION_MAPPING['BUY']].shape[0]}")
    logger.info(f"Buy signals (Legacy RSI): {df[df['buy_signal'] == 1].shape[0]}")
    
    # Drop rows with NaNs generated by Rolling/Shift/Diff operations
    # But preserve rows if only stop_loss/take_profit are NaN (since they aren't features)
    indicator_cols = [col for col in df.columns if col not in ['stop_loss', 'take_profit']]
    initial_len = len(df)
    df = df.dropna(subset=indicator_cols)
    logger.info(f"Dropped {initial_len - len(df)} rows due to NaN values in indicators.")
    
    # Fill remaining NaNs in stop_loss/take_profit with 0 or other default to stay clean
    df.loc[:, ['stop_loss', 'take_profit']] = df[['stop_loss', 'take_profit']].fillna(0)
    
    logger.info("Signal generation completed.")
    return df
