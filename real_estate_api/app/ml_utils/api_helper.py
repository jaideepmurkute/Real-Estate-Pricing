
"""
    This module handles calls/requests from api_requestor.
    Interacts with ml module to get the historical data and predictions.

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
import tensorflow as tf
from tqdm import tqdm

# from .utils import *
from .utils import *

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
    # Checks if the region name is present in the available regions from the train data
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

def predict(config: Dict, requested_feature: str, forecast_date: Optional[str]=None, 
            return_data: Optional[bool] = False) -> None:
    """
    Predicts the real estate pricing based on the given configuration and forecast date.
    
    Args:
        config (Dict): A dictionary containing the configuration parameters.
        forecast_date (Optional[str], optional): The forecast date in the format 'YYYY-MM-DD'. Defaults to None.
        return_data (Optional[bool], optional): Whether to return the data used for forecasting. Defaults to False.
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
    aggr_preds_df = aggr_preds_df[~aggr_preds_df.index.duplicated(keep=False)]
    
    # can use following: index for : 
    # 'Metro_median_sale_price_uc_sfrcondo_sm_sa_month'
    # 'Metro_mean_sale_price_uc_sfrcondo_sm_sa_month'
    
    # ----- get the feature to return -----
    
    # map the UI feature names to the data feature name patterns; since features names
    # in data are contain feature patterns(ex. mean_sale_price) not UI feature names 'Mean Sale Price'
    data_featname_UI_featname_map = {
        'Mean Sale Price': 'mean_sale_price',
        'Median Sale Price': 'median_sale_price',
        'Pct. Sold Above List': 'pct_sold_above_list',
        'Pct. Sold Below List': 'pct_sold_below_list',
        'Sale Price to List Ratio': 'sale_to_list',
        'Market Temperature Index': 'market_temp_index',
        'Zillow Home Value Index': 'zhvi',
        'Zillow Home Value Forecast': 'zhvf',
    }
    # Note there can be >1 coumns matching the pattern; for now, we are jus returning the first
    # Like ignorring raw / sa / sm_sa etc.
    
    feat_to_return = None
    # for pattern_str in data_featname_UI_featname_map.values():
    pattern_str = data_featname_UI_featname_map[requested_feature]
    for data_featname in aggr_preds_df.index.values:
        if pattern_str in data_featname:
            feat_to_return = data_featname
            break
    
    # ----- generate forecast -----
    if feat_to_return is None:
        forecast_value = 0
    else:
        forecast_value = aggr_preds_df[feat_to_return]
    forecast_value = round(forecast_value, 3)
    
    if return_data:
        train_df.columns = feature_names
        train_df = train_df.reset_index()
        train_df.rename({'index': 'Date'}, axis=1, inplace=True)
        
        if feat_to_return is None:
            ret_df = train_df.tail(n=6)
            ret_df.Price = 0
            return forecast_value, ret_df
        else:
            ret_df = train_df[["Date", feat_to_return]]
            ret_df = ret_df.rename(columns={feat_to_return: 'Price'})
            ret_df['Price'] = ret_df['Price'].round(3)
        
        return forecast_value, ret_df
    
    return forecast_value, None


def get_states_list(config: Dict) -> List[str]:
    # as of now, not choosing between month and week; since as of now we select state first; 
    # without selecting the granularity
    
    monthly_ref_fname = 'Metro_mean_sale_price_uc_sfrcondo_sm_sa_month.csv'
    # weekly_ref_fname = 'Metro_mean_sale_price_uc_sfrcondo_sm_sa_week.csv',
    
    fpath = os.path.join(config['data_dir'], monthly_ref_fname)
    
    df = pd.read_csv(fpath)
    df.dropna(subset=["StateName"], inplace=True)
    states_list = df["StateName"].unique().tolist()
    
    return states_list

def get_state_regions(config: Dict, state: str) -> List[str]:
    monthly_ref_fname = 'Metro_mean_sale_price_uc_sfrcondo_sm_sa_month.csv'
    # weekly_ref_fname = 'Metro_mean_sale_price_uc_sfrcondo_sm_sa_week.csv',
    
    df = pd.read_csv(os.path.join(config['data_dir'], monthly_ref_fname))
    
    df = df[df["StateName"] == state]
    df.dropna(subset=["RegionName"], inplace=True)
    regions_list = df["RegionName"].unique().tolist()
    
    return regions_list
    
def get_features_list(config: Dict, state: str, region: str) -> List[str]:
    # as of now, not choosing between month and week; since as of now we select state first; 
    # without selecting the granularity
    
    config['region_name'] = region
    config['granularity'] = 'month'
    
    df = get_region_data(config, config['granularity'], region)
    
    # patterns in the data file values to be matched with the UI feature names
    data_featname_UI_featname_map = {
        'mean_sale_price': 'Mean Sale Price',
        'median_sale_price': 'Median Sale Price',
        'pct_sold_above_list': 'Pct. Sold Above List',
        'pct_sold_below_list': 'Pct. Sold Below List',
        'sale_to_list': 'Sale Price to List Ratio',
        'market_temp_index': 'Market Temperature Index',
        'zhvi': 'Zillow Home Value Index',
        'zhvf': 'Zillow Home Value Forecast',
    }
    
    supported_features = set()
    for feature_name in df.index.values:
        for key in data_featname_UI_featname_map.keys():
            if key in feature_name:
                ui_feature_name = data_featname_UI_featname_map[key]
                supported_features.add(ui_feature_name)
    
    return list(supported_features)
    

