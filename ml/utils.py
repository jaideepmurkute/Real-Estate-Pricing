
import os

import pandas as pd
import numpy as np

from typing import List, Optional, Dict
from scipy.stats.mstats import winsorize


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
    nan_pcts = (df[cols].isna().sum(axis=1) / len(cols))
    region_nan_pct_df = pd.DataFrame({'RegionName': df['RegionName'], 'nan_pct': nan_pcts})
    region_nan_pct_df['nan_pct'] = region_nan_pct_df['nan_pct'].astype(float)
    regions_to_drop = region_nan_pct_df[region_nan_pct_df['nan_pct'] > threshold]['RegionName']

    if verbose:
        print("Number of regions to drop: ", len(regions_to_drop))
        print("regions_to_drop[:10]: ", regions_to_drop[:10])
        print("-"*5)
        print("Before dropping regions:")
        print("Number of regions before dropping: ", len(df['RegionName'].unique()))
        print("df.shape: ", df.shape)
    
    df = df[~df['RegionName'].isin(regions_to_drop)]
    
    if verbose:
        print("-"*5)
        print("After dropping regions:")
        print("Number of regions before dropping: ", len(df['RegionName'].unique()))
        print("data_df.shape: ", df.shape)
    
    return df


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
        # df[cols] = df[cols].interpolate(method='linear', axis=1)
        df.loc[:, cols] = df.loc[:, cols].interpolate(method='linear', axis=1)
        
    
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


def generate_fname(data_type: str, price_to_use: str, smoothing: bool, seasonal_adjustment: bool, 
                   granularity: str) -> str:
    """
    Generate a filename based on the given parameters.
        
    Parameters:
        data_type (str): The type of data.
        price_to_use (str): The price to use.
        smoothing (bool): Whether to apply smoothing.
        seasonal_adjustment (bool): Whether to apply seasonal adjustment.
        granularity (str): The granularity of the data.
    
    Returns:
        str: The generated filename.
    
    """
    str_former = []
    if data_type == 'sale_price':
        str_former.append('Metro_')
        str_former.append(price_to_use)
        str_former.append('_sale_price_uc_sfrcondo_')

        if smoothing: str_former.append('sm_')
        if seasonal_adjustment: str_former.append('sa_')
        str_former.append(granularity)
    elif data_type == 'pct_sold_above_list':
        str_former.append('Metro_')
        str_former.append('pct_sold_above_list_uc_sfrcondo_')
        if smoothing: str_former.append('sm_')
        str_former.append(granularity)
    elif data_type == 'pct_sold_below_list':
        str_former.append('Metro_')
        str_former.append('pct_sold_below_list_uc_sfrcondo_')
        if smoothing: str_former.append('sm_')
        str_former.append(granularity)
    elif data_type == 'sale_to_list_ratio':
        # Metro_mean_sale_to_list_uc_sfrcondo_sm_month
        str_former.append('Metro_')
        str_former.append(price_to_use)
        str_former.append('_sale_to_list_uc_sfrcondo_')
        if smoothing: str_former.append('sm_')
        str_former.append(granularity)
    elif data_type == 'market_heat_index':
        # Metro_market_temp_index_uc_sfrcondo_month
        str_former.append('Metro_')
        str_former.append('market_temp_index_uc_sfrcondo_')
        str_former.append(granularity)
    elif data_type == 'home_value_index':
        # Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month
        str_former.append('Metro_')
        str_former.append('zhvi_uc_sfrcondo_tier_0.33_0.67_')
        if smoothing: str_former.append('sm_')
        if seasonal_adjustment: str_former.append('sa_')
        str_former.append(granularity)
    elif data_type == 'home_value_forecast':
        # Metro_zhvf_growth_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv
        str_former.append('Metro_')
        str_former.append('zhvf_growth_uc_sfrcondo_tier_0.33_0.67_')
        if smoothing: str_former.append('sm_')
        if seasonal_adjustment: str_former.append('sa_')
        str_former.append(granularity)
    
    str_former.append('.csv')
    
    return ''.join(str_former)


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


def has_inf(df: 'pd.DataFrame', cols: 'Optional[List[str]]'=None) -> int:
    """
    Function to check if the DataFrame has infinite values.
    If inf values present; returns the number of infinite values in the DataFrame.
    Otherwise, returns 0.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
        cols (list): The list of columns to consider for infinite values.
                Default is None, which considers all columns.
    
    Returns:
        int: The number of infinite values in the DataFrame.
    """
    
    if cols is None:
        return np.isinf(df).sum().sum()
    return np.isinf(df[cols]).sum().sum()
    

def has_nan(df: 'pd.DataFrame', cols: 'Optional[List[str]]'=None) -> int:
    """
    Function to check if the DataFrame has missing values.
    If missing values present; returns the number of missing values in the DataFrame.
    Otherwise, returns 0.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
        cols (list): The list of columns to consider for missing values.
                Default is None, which considers all columns.
    
    Returns:
        int: The number of missing values in the DataFrame.
    """
    if cols is None:
        return df.isna().sum().sum()
    return df[cols].isna().sum().sum()


def fill_inf(df: pd.DataFrame, cols: Optional[List[str]]=None, method: Optional[str]='value', \
             fill_value=np.nan, verbose: Optional[bool]=True) -> pd.DataFrame:
    """
    Function to fill infinite values in the DataFrame.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
        cols (list): The list of columns to consider for infinite values.
                Default is None, which considers all columns.
        method (str): The method to use for filling infinite values. Default is 'value'.
        fill_value: The value to use for filling infinite values. Default is np.nan.
        verbose (bool): Whether to print the information before and after filling. Default is True.
    
    Returns:
        pandas.DataFrame: The DataFrame with filled infinite
    """
    if method == 'value':
        if cols is None:
            df = df.replace([np.inf, -np.inf], fill_value)
        else:
            df[cols] = df[cols].replace([np.inf, -np.inf], fill_value)
    elif method == 'interpolate':
        if cols is None:
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.interpolate(method='linear', axis=1)
        else:
            df[cols] = df[cols].replace([np.inf, -np.inf], np.nan)
            # df[cols] = df[cols].interpolate(method='linear', axis=1)
            df.loc[:, cols] = df.loc[:, cols].interpolate(method='linear', axis=1)
            
    
    return df


def reduce_mem_usage(df: pd.DataFrame, verbose: Optional[bool]=False) -> pd.DataFrame:
    """
    Update the data types of the columns in the DataFrame to reduce memory usage.
    
    Args:
        df (pandas.DataFrame): The input DataFrame.
        verbose (bool): Whether to print the information before and after optimization. Default is False.
    
    Returns:
        pandas.DataFrame: The DataFrame with optimized memory usage.
    """
    if verbose:
        start_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


