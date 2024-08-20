
'''
    This script generates region-wise data files from the Zillow dataset.
    Region-wise data files are generated for each region and granularity (week and month).
    Generated files contain 'features' like prices, ratios, indices, etc.
    
    IMPORTANT NOTE:
    This script is to be run only once for the given data; and should be run after running the
    'generate_cols_to_use.py' script.
    
    __author__ = ''
    __email__ = ''
    __version__ = ''
'''

import os
from typing import Dict, Tuple, Optional, List, Union

import pandas as pd
import numpy as np
import json
from tqdm import tqdm

from utils import *


def verify_data(config: Dict[str, Union[str, List[str]]], region_name: str) -> None:
    """
    Verify the generated data for a specific region.
    Args:
        config (Dict[str, Union[str, List[str]]]): A dictionary containing configuration parameters.
        region_name (str): The name of the region to verify the data for.
    Returns:
        None
    """
    # check data by reading it back and printing
    for granularity in ['week', 'month']:
        save_fname = f'all_data_df_{granularity}_{region_name}.csv'
        save_fpath = os.path.join(config['region_data_store_dir'], save_fname)
        all_data_df = pd.read_csv(save_fpath, index_col=0)
        print('all_data_df.shape:', all_data_df.shape)
        print('all_data_df.columns:', all_data_df.columns)
        print(all_data_df.head())
        print('-'*50)

def data_cleaning(df: pd.DataFrame, fname: str, region_name: str, granularity: str) -> pd.DataFrame:
    """
    Clean the given DataFrame by performing the following steps:
    1. Filter the DataFrame by the 'RegionName' column.
    2. Set the index of the DataFrame to be the filename without the '.csv' extension.
    3. Drop columns that are not in the 'cols_to_use' dictionary based on the given 'granularity'.
    4. Add columns that are in the 'cols_to_use' dictionary but not in the DataFrame, and fill them with NaN values.
    5. Sort the columns of the DataFrame for consistency.
    6. Remove any duplicate rows based on the index.
    Parameters:
        df (pd.DataFrame): The input DataFrame to be cleaned.
        fname (str): The filename of the DataFrame.
        region_name (str): The name of the region to filter by.
        granularity (str): The granularity of the data.
    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    
    # filter by region_name
    df = df[df['RegionName'] == region_name]
    df.index = [fname.split('.csv')[0]] * df.shape[0]

    cols_to_use = cols_to_use_dict[granularity]

    # drop columns df may have that are not in cols_to_use
    df = df.drop([col for col in df.columns if col not in cols_to_use], axis=1)

    # add columns that are in cols_to_use but not in df
    cols_to_add = [col for col in cols_to_use if col not in df.columns]
    for col in cols_to_add:
        df[col] = np.nan

    # sort columns - for consistency across all dfs
    df = df[cols_to_use]

    # drop duplicates; just in case they seep in
    df = df.loc[~df.index.duplicated(keep='first')]
    
    return df                            

                            
def generate_regionwise_data(config: Dict, cols_to_use_dict: Dict) -> None:
    """
    Generate region-wise data files from the Zillow dataset.
    Args:
        config (Dict): A dictionary containing configuration parameters.
        cols_to_use_dict (Dict): A dictionary containing columns to use for each granularity.
    Returns:
        None
    """
    for granularity in ['week', 'month']:
        print("Generating data for granularity:", granularity)

        ref_df_fname = cols_to_use_dict[f'{granularity}_data_ref_fname']
        ref_df = pd.read_csv(os.path.join(config['data_dir'], ref_df_fname))
        regions_to_use = ref_df['RegionName'].tolist()

        for region_name in tqdm(regions_to_use):
            all_data_df = pd.DataFrame()
            for data_type in config['data_types']:
                for price_to_use in ['mean', 'median']:
                    for smoothing in [True, False]:
                        for seasonal_adjustment in [True, False]:
                            df, fname = get_data(config, data_type, price_to_use, smoothing, seasonal_adjustment, granularity, 
                                        skip_if_not_found=True, verbose=False)
                            if df is None: continue
                            
                            df = data_cleaning(df, fname)
                            
                            all_data_df = pd.concat([all_data_df, df], axis=0)

            save_fname = f'all_data_df_{granularity}_{region_name}.csv'
            save_fpath = os.path.join(config['region_data_store_dir'], save_fname)
            all_data_df.to_csv(save_fpath, index=True)      
        
# --------------------------------------------------------------------------------

config = {
    'data_dir': os.path.join('..', 'data', 'zillow'),
    'region_data_store_dir': os.path.join('..', 'data', 'zillow', 'region_data_store'),
    'data_types': ['sale_price', 'pct_sold_above_list', 'pct_sold_below_list', 'sale_to_list_ratio', 
                                'market_heat_index', 'home_value_index', 'home_value_forecast'],
}
if not os.path.exists(config['data_dir']):
    raise FileNotFoundError(config['data_dir'])
if not os.path.exists(config['region_data_store_dir']):
    os.makedirs(config['region_data_store_dir'])

# id_cols = ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName']
cols_to_use_dict = json.load(open('cols_to_use.json'))

# From some of the dataframes; there are regions for which data is not available.
# All rows are set as None for those region's data for that file/feature.
generate_regionwise_data(config, cols_to_use_dict)


verify_data(config, 'New York, NY')

