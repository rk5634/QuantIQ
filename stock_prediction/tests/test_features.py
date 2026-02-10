
import pytest
import pandas as pd
import numpy as np
from stock_prediction.src import features

@pytest.fixture
def sample_data():
    dates = pd.date_range(start='2021-01-01', periods=100, freq='H')
    df = pd.DataFrame({
        'time': dates,
        'open': np.random.rand(100) * 100,
        'high': np.random.rand(100) * 100,
        'low': np.random.rand(100) * 100,
        'close': np.random.rand(100) * 100
    })
    return df

def test_add_technical_indicators(sample_data):
    df = features.add_technical_indicators(sample_data)
    
    # Check if columns are added
    expected_cols = ['rsi', 'ma_50', 'ma_200', 'ema_12', 'ema_26', 
                     'macd_diff', 'bollinger_mavg', 'bollinger_hband', 
                     'bollinger_lband', 'price_change_15m']
    
    for col in expected_cols:
        assert col in df.columns

def test_add_lag_features(sample_data):
    df = features.add_lag_features(sample_data, lags=2)
    
    assert 'lag_close_1' in df.columns
    assert 'lag_close_2' in df.columns
    assert 'lag_open_1' in df.columns
    
def test_generate_signals(sample_data):
    # Mock necessary columns for signal generation
    sample_data['rsi'] = 50
    sample_data['ma_50'] = 50
    features.add_technical_indicators(sample_data) # ensures cols exist
    
    df = features.generate_signals(sample_data)
    
    assert 'buy_signal' in df.columns
    assert 'sell_signal' in df.columns
    assert 'stop_loss' in df.columns
    assert 'take_profit' in df.columns
    assert 'action' in df.columns
