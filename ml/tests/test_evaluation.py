import sys
import os
# Add the parent directory to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from ml.prediction import test_model, load_scaler, load_model, housekeeping

@pytest.fixture
def config():
    return {
        'region_name': 'Adrian, MI', 
        'granularity': 'month',
        'look_back': 6, 
        'n_folds': 3, 
        'eval_batch_size': 64, 
        'return_data_type': 'sale_price',
        'return_aggr_type': 'median',
        'handle_outliers': False,
        'seed': 42, 
        'data_dir': '../data/zillow', 
        'region_data_store_dir': '../data/zillow/region_data_store',
        'model_store_dir': 'model_store',
    }

@pytest.fixture
def setup_housekeeping(config):
    # Call the housekeeping function
    housekeeping(config)

@pytest.fixture
def test_df():
    data = np.random.rand(10, 5)
    return pd.DataFrame(data)

@pytest.fixture
def train_df():
    data = np.random.rand(20, 5)
    return pd.DataFrame(data)

@patch('evaluation.load_scaler')
@patch('evaluation.load_model')
def test_test_model(mock_load_model, mock_load_scaler, config, test_df, train_df, setup_housekeeping):
    
    # Mock the scaler
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = test_df.values
    mock_scaler.inverse_transform.side_effect = lambda x: x
    mock_load_scaler.return_value = mock_scaler
    
    # Mock the model
    mock_model = MagicMock()
    mock_model.predict.side_effect = lambda x: x
    mock_load_model.return_value = mock_model
    
    # Call the function
    predictions = test_model(config, test_df, train_df)
    
    # Assertions
    assert predictions is not None
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (test_df.shape[1],)

if __name__ == "__main__":
    pytest.main()