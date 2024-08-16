
import os
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


def missing_value_heatmap(config: Dict, df: pd.DataFrame, cols: List[str]) -> None:
    plt.figure(figsize=(12, 6))
    sns.heatmap(df[cols].isnull(), cbar=False, cmap='viridis')
    plt.xlabel('Time')
    plt.ylabel('Regions')
    plt.title('Missing Data Heatmap')
    # plt.show()
    plt.savefig(os.path.join(config['data_viz_dir'], 'missing_data_heatmap.png'), dpi=300)
    plt.close()


def value_distribution(config: Dict, df: pd.DataFrame, cols: List[str]) -> None:
    all_prices = []
    for index, row in df.iterrows():
        all_prices.append(row[cols].values)

    all_prices = np.array(all_prices)

    sns.displot(all_prices.ravel(), kde=True, bins=100)
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.title('Distribution of median prices')
    plt.savefig(os.path.join(config['data_viz_dir'], 'missing_data_heatmap.png'), dpi=300)
    plt.close()


def plot_per_region_time_series(config, df, cols, rebase=True):
    for index, row in df.iterrows():
        prices = row[cols]
        if rebase:
            prices = prices - prices.iloc[0]
        sns.lineplot(data=prices)

    title_suffix = ' - Rebased' if rebase else ''
    plt.title('Region-wise Value Series' + title_suffix)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.xticks([]) # Disable x-axis labels to avoid clutter
    # plt.show()
    plt.savefig(os.path.join(config['data_viz_dir'], 'region_time_series.png'), dpi=300)
    plt.close()


def plot_per_state_time_series(config, df, cols, rebase=True):
    grouper = df[cols].groupby(df['StateName'])
    state_mean_prices = grouper.apply(lambda x: x.mean(axis=0, skipna=True)).reset_index()

    for index, row in state_mean_prices.iterrows():
        prices = row[cols]
        if rebase: prices = prices - prices.iloc[0]
        sns.lineplot(data=prices)

    title_suffix = ' - Rebased' if rebase else ''
    plt.title('State-wise Avg. of Weekly Regional Values' + title_suffix)
    plt.xlabel('Weeks')
    plt.ylabel('Avg. of Regional Median Price')
    plt.xticks([]) # Disable x-axis labels to avoid clutter
    # plt.show()
    plt.savefig(os.path.join(config['data_viz_dir'], 'state_time_series.png'), dpi=300)
    plt.close()


def plot_country_time_series(config, df, cols, rebase=True):
    for index, row in df.iterrows():
        if row.RegionName == 'United States':
            prices = row[cols]
            if rebase: prices = prices - prices.iloc[0]
            sns.lineplot(data=prices)
            break
    
    title_suffix = ' - Rebased' if rebase else ''
    plt.title('United States - Median Value' + title_suffix)
    plt.xlabel('Weeks')
    plt.ylabel('Price')
    plt.xticks([]) # Disable x-axis labels to avoid clutter
    # plt.show()
    plt.savefig(os.path.join(config['data_viz_dir'], 'country_time_series.png'), dpi=300)
    plt.close()



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
            df[cols] = df[cols].interpolate(method='linear', axis=1)
    
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

