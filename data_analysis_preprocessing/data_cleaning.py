
import os
from typing import List, Tuple, Union, Dict, Any, Optional

import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize


from utils import *


def drop_missing_val_regions(df: 'pd.DataFrame', cols: 'List[str]', threshold: 'Optional[float]' = 0.6, 
                             verbose: 'Optional[bool]' = False):
    """
    Drop rows with missing values in the 'region' column.
    
    Parameters:
        df (pandas.DataFrame): The input DataFrame.
        cols (list): The list of columns to consider for missing values. 
        threshold (float): The threshold for dropping rows with missing values [0, 1]. Default is 0.6.
        verbose (bool): Whether to print the information before and after dropping. Default is False.
        
    Returns:
        pandas.DataFrame: The cleaned DataFrame.
    """
    nan_pcts = (data_df.isna().sum(axis=1) / len(cols))
    region_nan_pct_df = pd.DataFrame({'RegionName': data_df['RegionName'], 'nan_pct': nan_pcts})
    region_nan_pct_df['nan_pct'] = region_nan_pct_df['nan_pct'].astype(float)
    regions_to_drop = region_nan_pct_df[region_nan_pct_df['nan_pct'] > threshold]['RegionName']

    if verbose:
        print("Number of regions to drop: ", len(regions_to_drop))
        print("regions_to_drop[:10]: ", regions_to_drop[:10])
        print("-"*5)
        print("Before dropping regions:")
        print("Number of regions before dropping: ", len(data_df['RegionName'].unique()))
        print("data_df.shape: ", data_df.shape)
    
    data_df = data_df[~data_df['RegionName'].isin(regions_to_drop)]
    
    if verbose:
        print("-"*5)
        print("After dropping regions:")
        print("Number of regions before dropping: ", len(data_df['RegionName'].unique()))
        print("data_df.shape: ", data_df.shape)
    
    return data_df


def fill_nan_prices(df: pd.DataFrame, cols: list, method='interpolate', verbose=False):
    """
    Fill missing values in the specified columns using the specified method.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        cols (list): The list of columns to consider for missing values.
        method (str): The method to use for filling missing values. Default is 'interpolate'.
        verbose (bool): Whether to print the information before and after filling. Default is False.
    
    Returns:
       pandas.DataFrame: The cleaned DataFrame.
    """
    if verbose:
        print("# of missing values BEFORE interpolation: ", df[cols].isna().sum().sum())
    
    if method == 'interpolate':
        if verbose: print("Filling missing values with linear interpolation...")
        df[cols] = df[cols].interpolate(method='linear', axis=1)
    
    if verbose:
        print("# of missing values AFTER interpolation: ", df[cols].isna().sum().sum())

    return df

def handle_outliers(df: pd.DataFrame, cols: list, method='winsorize', winsorize_limits=(0.05, 0.05), verbose=False):
    """
    Handle outliers in the specified columns using the specified method.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        cols (list): The list of columns to consider for outliers.
        method (str): The method to use for handling outliers. Default is 'winsorize'.
        winsorize_limits (tuple): The lower and upper limits for winsorizing. Default is (0.05, 0.05).
        verbose (bool): Whether to print the information before and after handling outliers. Default is False.
    
    Returns:
       pandas.DataFrame: The cleaned DataFrame.
    """
    if verbose:
        print("Handling outliers in the specified columns...")
    
    if method == 'winsorize':
        if verbose: print("Winsorizing the specified columns...")
        df[cols] = winsorize(df[cols], limits=winsorize_limits, inclusive=(True, True))
    
    return df

def get_data(config: Dict, data_type: str, price_to_use: str, smoothing: bool, seasonal_adjustment: bool, \
             granularity: str) -> pd.DataFrame:
    fname = generate_fname(data_type, price_to_use, smoothing, seasonal_adjustment, granularity)
    fpath = os.path.join(config['data_dir'], fname)
    if not os.path.exists(fpath):
        raise FileNotFoundError(fpath)
    print("Reading data from:", fpath)
    return pd.read_csv(fpath)


def get_col_descriptor(df):
    desc_dict = {}
    desc_dict['id_cols'] = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName']
    desc_dict['non_id_cols'] = [col for col in df.columns if col not in desc_dict['id_cols']]

    return desc_dict

# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------
    
config = {
    'data_dir': os.path.join('..', 'data', 'zillow'), 
    'data_viz_dir': os.path.join('..', 'data_viz', 'zillow'),
    'handle_outliers': False,
}

if not os.path.exists(config['data_dir']):
    raise FileNotFoundError(config['data_dir'])
if not os.path.exists(config['data_viz_dir']):
    os.makedirs(config['data_viz_dir'])

col_desc = get_col_descriptor()




# ------------ Data Loading & Optimizing ------------
# Read the data
med_sp_sm_sa_mon_df = get_data(config, 'sale_price', 'median', True, True, 'month')

# reduce memory usage
med_sp_sm_sa_mon_df = reduce_mem_usage(med_sp_sm_sa_mon_df, verbose=False)

# ------------ Data Cleaning ------------

# Fill infinite values with np.nan
if has_inf(med_sp_sm_sa_mon_df, cols=col_desc['non_id_cols']):
    med_sp_sm_sa_mon_df = fill_inf(med_sp_sm_sa_mon_df, cols=col_desc['non_id_cols'], 
                                   method='value', fill_value=np.nan, verbose=True)

# Drop regions with more than 60% missing(NaN) values
if has_nan(med_sp_sm_sa_mon_df, cols=col_desc['non_id_cols']):
    med_sp_sm_sa_mon_df = drop_missing_val_regions(med_sp_sm_sa_mon_df, cols=col_desc['non_id_cols'], 
                                                threshold=0.6, verbose=True)

# Fill missing values in the specified columns using linear interpolation
med_sp_sm_sa_mon_df = fill_nan_prices(fill_nan_prices, cols=col_desc['non_id_cols'], 
                                      method='interpolate', verbose=True)

# handle outliers
if config['handle_outliers']:
    med_sp_sm_sa_mon_df = handle_outliers(med_sp_sm_sa_mon_df, cols=col_desc['non_id_cols'], method='winsorize',
                                        winsorize_limits=(0.05, 0.05), verbose=True)

# ------------ Data Vizualization ------------

missing_value_heatmap(config, med_sp_sm_sa_mon_df, col_desc['non_id_cols'])
value_distribution(config, med_sp_sm_sa_mon_df, col_desc['non_id_cols'])
plot_per_region_time_series(config, med_sp_sm_sa_mon_df, col_desc['non_id_cols'], rebase=True)
plot_per_state_time_series(config, med_sp_sm_sa_mon_df, col_desc['non_id_cols'], rebase=True)
plot_country_time_series(config, med_sp_sm_sa_mon_df, col_desc['non_id_cols'], rebase=True)

# ------------ Data Preprocessing ------------
# Drop the columns that are not needed
# Scaling/Normalization
# Encoding
# Feature Engineering
# Feature Selection






