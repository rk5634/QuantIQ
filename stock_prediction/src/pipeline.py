import pandas as pd
from sklearn.model_selection import train_test_split
from stock_prediction.config import settings
from stock_prediction.src.data_loader import load_data
from stock_prediction.src.preprocessing import clean_data, resample_data
from stock_prediction.src.features import add_technical_indicators, add_lag_features, generate_signals
from stock_prediction.src.models import train_models, save_models
from stock_prediction.src.evaluation import evaluate_models
from stock_prediction.src.backtesting import Backtester
from stock_prediction.src import visualization # Keep this for visualization.generate_all_eda_plots
from stock_prediction.src.utils import setup_logger

logger = setup_logger("pipeline")

def run_pipeline(data_path: str, timeframe: str = None):
    """
    Run the full end-to-end pipeline.
    
    Args:
        data_path: Path to input CSV.
        timeframe: Optional resampling frequency (e.g., '1H').
    """
    logger.info("Starting pipeline execution...")
    
    try:
        # 1. Load Data
        df = load_data(data_path)
        
        # 2. Preprocessing
        df = clean_data(df)
        
        # Resample if timeframe provided
        if timeframe:
            df = resample_data(df, timeframe)
            
        # 3. Feature Engineering
        df = add_technical_indicators(df)
        df = add_lag_features(df)
        df = generate_signals(df)
        
        # Generate EDA Plots
        # visualization.generate_all_eda_plots(df)
        
        # Save Processed Data for inspection
        processed_path = settings.PROCESSED_DATA_DIR / "processed_data.csv"
        logger.info(f"Saving processed data to {processed_path}")
        df.to_csv(processed_path, index=False)
        
        # 4. Prepare for Training
        # Drop non-feature columns for X
        # Keep only feature columns used in training
        feature_cols = [
            'open', 'high', 'low', 'close', 'rsi', 
            'ma_50', 'ma_200', 'ema_12', 'ema_26', 'macd_diff',
            'bollinger_mavg', 'bollinger_hband', 'bollinger_lband', 'past_return',
            'adx', 'atr', 'obv', 'obv_slope', # Senior Trader Features
            'supertrend', 'supertrend_signal' # Indian Trader Features
        ]
        # Add lag columns dynamically
        lag_cols = [col for col in df.columns if col.startswith('lag_')]
        feature_cols.extend(lag_cols)
        
        X = df[feature_cols]
        y = df['action'].astype(int) # Ensure integer type
        
        logger.info(f"Training with {X.shape[1]} features and {X.shape[0]} samples.")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        if y.nunique() < 2:
            logger.error("Target variable 'action' has less than 2 classes. Models cannot be trained.")
            logger.error("Try adjusting trading strategy parameters (RSI thresholds, etc.) in settings.py to generate more signals.")
            return {}

        # Train-Test Split
        train_size = int(len(df) * (1 - settings.TEST_SIZE))
        df_train = df.iloc[:train_size]
        df_test = df.iloc[train_size:]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=settings.TEST_SIZE, 
            shuffle=False # Time series data should not be shuffled
        )
        
        logger.info("xxx Data Split Info xxx")
        logger.info(f"Training Data: {len(df_train)} samples ({df_train['time'].min()} to {df_train['time'].max()})")
        logger.info(f"Testing Data:  {len(df_test)} samples ({df_test['time'].min()} to {df_test['time'].max()})")
        logger.info("xxxxxxxxxxxxxxxxxxxxxxx")
        
        # 5. Train Models
        trained_models = train_models(X_train, y_train)
        
        # 6. Evaluate Models
        results = evaluate_models(trained_models, X_test, y_test)
        
        # 7. Backtesting
        logger.info("Starting Backtesting...")
        
        # We need the corresponding DataFrame slice for the test set to get prices/dates
        split_index = int(len(df) * (1 - settings.TEST_SIZE))
        df_test = df.iloc[split_index:].reset_index(drop=True)
        
        backtester = Backtester()
        backtest_metrics = []
        
        for name, model_info in results.items():
            predictions = model_info['predictions']
            probabilities = model_info.get('probabilities')
            metrics = backtester.run(df_test, predictions, probabilities=probabilities, model_name=name)
            if metrics:
                # Add Timeframe to metrics (Insert at beginning for visibility)
                tf_label = timeframe if timeframe else "Original"
                ordered_metrics = {'Timeframe': tf_label}
                ordered_metrics.update(metrics)
                backtest_metrics.append(ordered_metrics)
            
        # Generate Backtest Summary Table
        visualization.save_backtest_table(backtest_metrics)
        
        logger.info("Backtesting completed.")
        
        # Save Models
        save_models(trained_models)
        
        logger.info("Pipeline completed successfully.")
        return results
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise e
