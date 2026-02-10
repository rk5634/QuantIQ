
import pytest
from pathlib import Path
import pandas as pd
from stock_prediction.src.data_loader import load_data

def test_load_data_success(tmp_path):
    # Create a dummy CSV file
    d = tmp_path / "subdir"
    d.mkdir()
    p = d / "test.csv"
    p.write_text("time,close\n2021-01-01 10:00:00,100\n2021-01-02 10:00:00,105")
    
    df = load_data(p)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert 'time' in df.columns
    assert 'close' in df.columns

def test_load_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")
