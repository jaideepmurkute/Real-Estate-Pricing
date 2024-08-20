
"""
    This module contains functions for evaluating regional real estate pricing models.

    __author__ = ''
    __email__ = ''
    __version__ = ''
"""

import os
import sys
import logging
from typing import Dict, Optional, Tuple, List, Union

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator

import tensorflow as tf
import pickle
from tqdm import tqdm

from ml.utils import *

'''
Tensorflow logging levels:
0: All logs
1: Filter out INFO logs
2: Filter out WARNING logs
3: Filter out INFO and WARNING logs
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# -------------------------------------------------

class SuppressOutput:
    """
        Redirect standard output and standard error
        Suppresses the standard output and standard error.
        To make the tensorflow predict's output silent.
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout
        sys.stderr = self._original_stderr

    def write(self, _):
        pass

    def flush(self):
        pass

def check_input(config: Dict) -> None:
    # Check if the region name is present in the available regions from the train data
    available_regions = get_available_regions(config)
    if config['region_name'] not in available_regions:
        raise ValueError(f"Region name: {config['region_name']} not found in available regions: {available_regions}")

        
def test_model(config: Dict, test_df: pd.DataFrame, 
                            train_df: Optional[pd.DataFrame]=None) -> np.ndarray:
    """
    Loops through all available folds and performs scaling and model prediction for each fold.
    Aggregates all folds predictions with a simple average.
    Args:
        config (Dict): Configuration dictionary containing model parameters.
        test_df (pd.DataFrame): DataFrame containing test data.
        train_df (Optional[pd.DataFrame], optional): DataFrame containing train data. Defaults to None.
    Returns:
        np.ndarray: Array of aggregated predictions.
    """
    per_fold_preds = []
    for fold_idx in range(config['n_folds']):
        scaler = load_scaler(config, fold_idx)
        testX = scale_data(test_df, scaler=scaler)
        X = create_test_dataset(config, testX)
        model = load_model(config, fold_idx)
        dataset = tf.data.Dataset.from_tensor_slices(X).batch(config['eval_batch_size']).\
                            prefetch(tf.data.AUTOTUNE)
        
        preds = []
        with SuppressOutput():
            for batch in tqdm(dataset, disable=True):
                batch_preds = model.predict(batch)
                preds.append(batch_preds)
        preds = np.concatenate(preds, axis=0)
        preds = scaler.inverse_transform(preds)
        per_fold_preds.append(preds)
    
    per_fold_preds = np.concatenate(per_fold_preds, axis=0)
    aggr_preds = np.average(per_fold_preds, axis=0)
    
    return aggr_preds

def update_forecast_date(config: dict, train_df: pd.DataFrame, forecast_date: Optional[str]=None) -> str:
    """
    Takes in a forecast date and performs adjustments based on the valid values range from the \
    available train data.
    
    If the forecast date is None, then the last date in the train data is considered.
    If the forecast date is less than the first date in the train data, then the first date is considered.
    If the forecast date is greater than the last date in the train data, then the last date is considered.
    If the forecast date is not present in the train data, then the nearest date is considered.
    Else, an error is raised.
    
    Args:
        config (dict): Configuration dictionary.
        train_df (pd.DataFrame): DataFrame containing the train data.
        forecast_date (str, optional): The forecast date to be adjusted. Defaults to None.
    Returns:
        str: The adjusted forecast date.
    Raises:
        ValueError: If the forecast date does not meet the criteria specified in the conditions.
    """
    
    if forecast_date is None:
        forecast_date = train_df.index[-1]
    elif forecast_date < train_df.index[0]:
        forecast_date = train_df.index[0]
    elif forecast_date > train_df.index[-1]:
        forecast_date = train_df.index[-1]
    elif forecast_date not in train_df.index:
        # if forecast date is not present in the data, then take the nearest date
        forecast_date = train_df.index[train_df.index.get_loc(forecast_date, method='nearest')]
    else:
        raise ValueError("Invalid forecast date")
    
    return forecast_date

def predict(config: Dict, forecast_date: Optional[str]=None) -> None:
    """
    Predicts the real estate pricing based on the given configuration and forecast date.
    
    Args:
        config (Dict): A dictionary containing the configuration parameters.
        forecast_date (Optional[str], optional): The forecast date in the format 'YYYY-MM-DD'. Defaults to None.
    Returns:
        None
    """
    train_df = get_region_data(config, config['granularity'], config['region_name'])
    train_df, feature_names = preprocess_dataframe(train_df)
    
    # '2024-06-30' - string in the format 'YYYY-MM-DD'
    forecast_date = update_forecast_date(config, train_df, forecast_date)
    
    # Slice the needed data for prediction
    forecast_date_idx = train_df.index.get_loc(forecast_date)
    train_df = train_df[forecast_date_idx-config['look_back']+1:]
    
    # if we dont have enough data for the look_back period, then use backfill
    if train_df.shape[0] < config['look_back']:
        train_df.fillna(method='bfill', inplace=True)
        
    aggr_preds = test_model(config, train_df)
    aggr_preds_df = pd.Series(aggr_preds, index=feature_names)
    
    # can use following: index for : 
    # 'Metro_median_sale_price_uc_sfrcondo_sm_sa_month'
    # 'Metro_mean_sale_price_uc_sfrcondo_sm_sa_month'
    feat_to_return = f'Metro_{config['return_aggr_type']}_{config['return_data_type']}_uc_sfrcondo_sm_sa_{config["granularity"]}'
    print("prediction: ", aggr_preds_df[feat_to_return])

if __name__ == '__main__':
    
    config = {
        'region_name': 'Adrian, MI', # Adrian, MI
        'granularity': 'month',
        'look_back': 6, 
        'n_folds': 3, 
        'eval_batch_size': 64, 
        
        'return_data_type': 'sale_price',
        'return_aggr_type': 'median',
        
        'handle_outliers': False,
        'seed': 42, 
        
        'data_dir': os.path.join('..', 'data', 'zillow'), 
        'region_data_store_dir': os.path.join('..', 'data', 'zillow', 'region_data_store'),
        'model_store_dir': os.path.join('model_store'),
    }

    housekeeping(config)
    check_input(config)
    predict(config)
    
    

