import sys
import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from ml.prediction import test_model as model_test
from ml.utils import housekeeping, load_scaler, load_model


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
        'data_dir': os.path.join('..', '..', 'data', 'zillow'), 
        'model_store_dir': os.path.join('..', '..', 'ml', 'model_store'),
        # region_data_store_dir is updated in housekeeping()
        'region_data_store_dir': os.path.join('..', '..', 'ml', 'model_store', 'region_data_store'), 
    }

@pytest.fixture
def setup_housekeeping(config):
    # Call the housekeeping function
    housekeeping(config)
    return config  # Return the updated config

@pytest.fixture
def test_df():
    data = np.random.rand(22, 22)  # 10 samples, 5 features
    return pd.DataFrame(data)

@pytest.fixture
def train_df():
    data = np.random.rand(22, 22)  # 10 samples, 5 features
    return pd.DataFrame(data)

@patch('ml.utils.load_scaler')
@patch('ml.utils.load_model')
def test_test_model(mock_load_model, mock_load_scaler, config, test_df, train_df, setup_housekeeping):
    config = setup_housekeeping  # Use the updated config from housekeeping

    # Mock the scaler
    mock_scaler = MagicMock()
    mock_scaler.transform.return_value = test_df.values
    mock_scaler.inverse_transform.side_effect = lambda x: x
    mock_load_scaler.return_value = mock_scaler

    # Mock the model
    mock_model = MagicMock()
    mock_model.predict.side_effect = lambda x: x
    mock_load_model.return_value = mock_model

    # Ensure the mocked functions do not raise FileNotFoundError
    mock_load_scaler.side_effect = lambda path: mock_scaler if 'scaler' in path else None
    mock_load_model.side_effect = lambda path: mock_model if 'model' in path else None

    # Call the function
    predictions = model_test(config, test_df, train_df)

    # Assertions
    assert predictions is not None
    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (test_df.shape[0],)