import os
import sys
import tempfile

import pytest
from unittest.mock import patch

import numpy as np
import pandas as pd
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import preprocess_dataframe, save_model, load_model

# ----------------- test preprocess_dataframe() -----------------
def test_preprocess_dataframe():
    # Create a sample DataFrame
    data = {
        'feature_name': ['feature1', 'feature2', 'feature3'],
        'value1': [1, 2, 3],
        'value2': [4, 5, 6],
        'value3': [7, 8, 9]
    }
    df = pd.DataFrame(data)
    df.set_index('feature_name', inplace=True)
    
    # Expected output
    expected_data = {
        'value1': [1, 2, 3],
        'value2': [4, 5, 6],
        'value3': [7, 8, 9]
    }
    
    expected_df = pd.DataFrame(expected_data).T
    expected_df.columns = pd.Index(expected_df.columns, dtype='int32')
    expected_feature_names = pd.Series(['feature1', 'feature2', 'feature3'])
    
    # Call the function 
    processed_df, feature_names = preprocess_dataframe(df)
    processed_df.columns = pd.Index(processed_df.columns, dtype='int32')
    expected_feature_names.name = feature_names.name
    
    # Assertions
    pd.testing.assert_frame_equal(processed_df, expected_df)
    assert np.array_equal(processed_df.columns, expected_df.columns), "The columns of the DataFrame are not equal"
    assert np.array_equal(processed_df.index.values, expected_df.index.values), "The index values of the DataFrame are not equal"
    assert feature_names.equals(expected_feature_names), "The values of the Series are not equal"
    

def test_preprocess_dataframe_empty():
    df = pd.DataFrame()
    processed_df, feature_names = preprocess_dataframe(df)
    
    assert processed_df.empty
    assert feature_names.empty

def test_preprocess_dataframe_missing_values():
    data = {
        'feature_name': ['feature1', 'feature2', 'feature3'],
        'value1': [1, None, 3],
        'value2': [4, 5, 6],
        'value3': [None, None, None]
    }
    df = pd.DataFrame(data)
    
    processed_df, feature_names = preprocess_dataframe(df)
    
    assert processed_df.isnull().sum().sum() == 0
    assert feature_names.isnull().sum().sum() == 0

# ----------------- test save_model() -----------------


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

@pytest.fixture
def mock_config(temp_dir):
    return {
        'region_name': 'test_region',
        'granularity': 'test_granularity',
        'region_model_store_dir': temp_dir
    }

@pytest.fixture
def simple_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, input_shape=(5,), activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

def test_save_model_no_fold_idx(mock_config, simple_model, temp_dir):
    '''
    Test that the function saves the model with the correct file name when no fold index is provided.
    Passes the model and config to the function and checks if the model file was created.
    '''
    save_model(mock_config, simple_model)
    expected_file = os.path.join(temp_dir, 'model_test_region_test_granularity_fold_.keras')
    assert os.path.isfile(expected_file), f"Model file {expected_file} was not created."

def test_save_model_with_fold_idx(mock_config, simple_model, temp_dir):
    '''
    Test that the function saves the model with the correct file name when a fold index is provided.
    '''
    fold_idx = 1
    save_model(mock_config, simple_model, fold_idx)
    expected_file = os.path.join(temp_dir, f'model_test_region_test_granularity_fold_{fold_idx}.keras')
    assert os.path.isfile(expected_file), f"Model file {expected_file} was not created."

def test_save_model_invalid_model(mock_config, temp_dir):
    '''
    Test that the function raises a ValueError when the model is invalid.
    Passing a string instead of a model of type tf.keras.models.Sequential.
    '''
    model = 'invalid_model'
    with pytest.raises(ValueError):
        save_model(mock_config, model)

def test_save_model_invalid_config(simple_model):
    '''
    Test that the function raises a ValueError when the config is invalid.
    Passing a string instead of a dictionary.
    '''
    config = 'invalid_config'
    with pytest.raises(ValueError):
        save_model(config, simple_model)

# ----------------- test load_model() -----------------
def test_load_model_no_fold_idx(mock_config, simple_model, temp_dir):
    '''
    Test that the function loads the model with the correct file name when no fold index is provided.
    '''
    save_model(mock_config, simple_model)
    loaded_model = load_model(mock_config)
    assert isinstance(loaded_model, tf.keras.models.Sequential), "The loaded model is not an instance of tf.keras.models.Sequential."


def test_load_model_with_fold_idx(mock_config, simple_model, temp_dir):
    '''
    Test that the function loads the model with the correct file name when a fold index is provided.
    '''
    fold_idx = 1
    save_model(mock_config, simple_model, fold_idx)
    loaded_model = load_model(mock_config, fold_idx)
    assert isinstance(loaded_model, tf.keras.models.Sequential), "The loaded model is not an instance of tf.keras.models.Sequential."
    
def test_load_model_invalid_config(simple_model):
    '''
    Test that the function raises a ValueError when the config is invalid.
    '''
    config = 'invalid_config'
    with pytest.raises(ValueError):
        load_model(config)
        
def test_load_model_invalid_fold_idx(mock_config, simple_model):
    '''
    Test that the function raises a ValueError when the fold index is invalid.
    '''
    fold_idx = 'invalid_fold_idx'
    with pytest.raises(ValueError):
        load_model(mock_config, fold_idx)

def test_load_model_no_model(mock_config):
    '''
    Test that the function raises a FileNotFoundError when the model file does not exist.
    '''
    with pytest.raises(FileNotFoundError):
        load_model(mock_config)

def test_load_model_invalid_model_file(mock_config, simple_model, temp_dir):
    '''
    Test that the function raises a ValueError when the model file is invalid.
    '''
    model_file = os.path.join(temp_dir, 'invalid_model_file')
    with open(model_file, 'w') as f:
        f.write('Invalid model file')
    
    with pytest.raises(FileNotFoundError):
        load_model(mock_config)
    
# ----------------- test housekeeping() -----------------
from utils import housekeeping

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

@pytest.fixture
def mock_config(temp_dir):
    return {
        'data_dir': os.path.join(temp_dir, 'data'),
        'data_viz_dir': os.path.join(temp_dir, 'data_viz'),
        'feature_dir': os.path.join(temp_dir, 'features'),
        'model_store_dir': os.path.join(temp_dir, 'models'),
        'region_name': 'test_region',
        'seed': 42,
        'granularity': 'test_granularity',
        'region_model_store_dir': temp_dir

    }

def test_housekeeping_data_dir_not_exists(mock_config):
    with pytest.raises(FileNotFoundError):
        housekeeping(mock_config)

def test_housekeeping_creates_directories(mock_config, temp_dir):
    os.makedirs(mock_config['data_dir'])  # Ensure data_dir exists to avoid FileNotFoundError

    housekeeping(mock_config)

    assert os.path.exists(mock_config['data_viz_dir']), "data_viz_dir was not created"
    assert os.path.exists(mock_config['feature_dir']), "feature_dir was not created"
    assert os.path.exists(mock_config['model_store_dir']), "model_store_dir was not created"
    assert os.path.exists(os.path.join(mock_config['model_store_dir'], mock_config['region_name'])), "region_model_store_dir was not created"

def test_housekeeping_sets_region_model_store_dir(mock_config, temp_dir):
    os.makedirs(mock_config['data_dir'])  # Ensure data_dir exists to avoid FileNotFoundError

    housekeeping(mock_config)

    expected_region_model_store_dir = os.path.join(mock_config['model_store_dir'], mock_config['region_name'])
    assert mock_config['region_model_store_dir'] == expected_region_model_store_dir, "region_model_store_dir was not set correctly"

@patch('utils.set_seed')
def test_housekeeping_calls_set_seed(mock_set_seed, mock_config, temp_dir):
    os.makedirs(mock_config['data_dir'])  # Ensure data_dir exists to avoid FileNotFoundError

    housekeeping(mock_config)

    mock_set_seed.assert_called_once_with(mock_config['seed'])





