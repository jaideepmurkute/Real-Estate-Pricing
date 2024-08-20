
'''
    Based on the reference files, generate the columns to use for the dat generaion and model development.
    Save the columns to use in a json file.
    
    IMPOTANT NOTE: 
    This script is to be run only once for the given data; and should be run before running the 
    'generate_regionwise_data.py' script.
    
    __author__ = ''
    __email__ = ''
    __date__ = ''
    __version__ = ''
'''
import os

import json
import pandas as pd


def generate_dict(config: dict) -> None:
    """
    Generates a dictionary containing the columns to use for monthly and weekly data.
    Args:
        config (dict): A dictionary containing the configuration parameters.
    Returns:
        None: This function does not return anything.
    Raises:
        FileNotFoundError: If the specified data files are not found.
    Example Usage:
        >>> config = {
        ...     'data_dir': '/path/to/data',
        ...     'monthly_ref_fname': 'monthly_reference.csv',
        ...     'weekly_ref_fname': 'weekly_reference.csv',
        ...     'id_cols': ['id'],
        ...     'monthly_data_slice_sz': 3,
        ...     'weekly_data_slice_sz': 5,
        ...     'op_store_dir': '/path/to/output'
        ... }
        >>> generate_dict(config)
    """

    monthly_ref_df = pd.read_csv(os.path.join(config['data_dir'], config['monthly_ref_fname']))
    monthly_date_cols = monthly_ref_df.drop(config['id_cols'], axis=1).columns.tolist()
    monthly_date_cols = monthly_date_cols[-config['monthly_data_slice_sz']:]

    weekly_ref_df = pd.read_csv(os.path.join(config['data_dir'], config['weekly_ref_fname']))
    weekly_date_cols = weekly_ref_df.drop(config['id_cols'], axis=1).columns.tolist()
    weekly_date_cols = weekly_date_cols[-config['weekly_data_slice_sz']:]

    # print('monthly_date_cols:', monthly_date_cols)
    # print('-'*50)
    # print('weekly_date_cols:', weekly_date_cols)

    cols_to_use_dict = {
        'month': monthly_date_cols,
        'week': weekly_date_cols,
        'month_data_ref_fname': config['monthly_ref_fname'],
        'week_data_ref_fname': config['weekly_ref_fname'],
    }
    
    save_fpath = os.path.join(config['op_store_dir'], 'cols_to_use.json')
    with open(save_fpath, 'w') as f:
        json.dump(cols_to_use_dict, f, indent=4)

    print(f'Columns to use saved at: {save_fpath}')

# -------------------------------------------------------------------------------------

if __name__ == '__main__':
    config = {
        'data_dir': os.path.join('..', 'data', 'zillow'),
        'monthly_ref_fname': 'Metro_mean_sale_price_uc_sfrcondo_sm_sa_month.csv',
        'weekly_ref_fname': 'Metro_mean_sale_price_uc_sfrcondo_sm_sa_week.csv',
        'id_cols': ['RegionID', 'SizeRank', 'RegionName', 'RegionType', 'StateName'],

        'monthly_data_slice_sz': 72,
        'weekly_data_slice_sz': 340,
        
        'op_store_dir': os.path.join('..'),
    }

    generate_dict(config)

# -------------------------------------------------------------------------------------
'''
# monthly models: 
#     data slice: last 6 years data - 72 columns
#     reference file: Metro_mean_sale_price_uc_sfrcondo_sm_sa_month.csv
# weekly models: 
#     data slice: last 340 weeks ~6.5 years data - 340 columns
#     reference file: Metro_mean_sale_price_uc_sfrcondo_sm_sa_week.csv


# Monthly data:
{'fname': 'Metro_mean_sale_price_uc_sfrcondo_sm_sa_month.csv', 'cols_cnt': 76}
{'fname': 'Metro_mean_sale_price_uc_sfrcondo_sm_month.csv', 'cols_cnt': 200}
{'fname': 'Metro_mean_sale_price_uc_sfrcondo_month.csv', 'cols_cnt': 202}
{'fname': 'Metro_median_sale_price_uc_sfrcondo_sm_sa_month.csv', 'cols_cnt': 76}
{'fname': 'Metro_median_sale_price_uc_sfrcondo_sm_month.csv', 'cols_cnt': 200}
{'fname': 'Metro_median_sale_price_uc_sfrcondo_month.csv', 'cols_cnt': 202}
{'fname': 'Metro_pct_sold_above_list_uc_sfrcondo_sm_month.csv', 'cols_cnt': 81}
{'fname': 'Metro_pct_sold_above_list_uc_sfrcondo_month.csv', 'cols_cnt': 83}
{'fname': 'Metro_pct_sold_below_list_uc_sfrcondo_sm_month.csv', 'cols_cnt': 81}
{'fname': 'Metro_pct_sold_below_list_uc_sfrcondo_month.csv', 'cols_cnt': 83}
{'fname': 'Metro_mean_sale_to_list_uc_sfrcondo_sm_month.csv', 'cols_cnt': 81}
{'fname': 'Metro_mean_sale_to_list_uc_sfrcondo_month.csv', 'cols_cnt': 83}
{'fname': 'Metro_market_temp_index_uc_sfrcondo_month.csv', 'cols_cnt': 84}
{'fname': 'Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv', 'cols_cnt': 300}
{'fname': 'Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_month.csv', 'cols_cnt': 347}
{'fname': 'Metro_zhvf_growth_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv', 'cols_cnt': 9}
{'fname': 'Metro_zhvf_growth_uc_sfrcondo_tier_0.33_0.67_month.csv', 'cols_cnt': 9}
'''

'''
# Weekly data:
{'fname': 'Metro_mean_sale_price_uc_sfrcondo_sm_sa_week.csv', 'cols_cnt': 733}
{'fname': 'Metro_mean_sale_price_uc_sfrcondo_sm_week.csv', 'cols_cnt': 860}
{'fname': 'Metro_mean_sale_price_uc_sfrcondo_week.csv', 'cols_cnt': 863}
{'fname': 'Metro_median_sale_price_uc_sfrcondo_sm_sa_week.csv', 'cols_cnt': 733}
{'fname': 'Metro_median_sale_price_uc_sfrcondo_sm_week.csv', 'cols_cnt': 860}
{'fname': 'Metro_median_sale_price_uc_sfrcondo_week.csv', 'cols_cnt': 863}
{'fname': 'Metro_pct_sold_above_list_uc_sfrcondo_sm_week.csv', 'cols_cnt': 342}
{'fname': 'Metro_pct_sold_above_list_uc_sfrcondo_week.csv', 'cols_cnt': 345}
{'fname': 'Metro_pct_sold_below_list_uc_sfrcondo_sm_week.csv', 'cols_cnt': 342}
{'fname': 'Metro_pct_sold_below_list_uc_sfrcondo_week.csv', 'cols_cnt': 345}
{'fname': 'Metro_mean_sale_to_list_uc_sfrcondo_sm_week.csv', 'cols_cnt': 342}
{'fname': 'Metro_mean_sale_to_list_uc_sfrcondo_week.csv', 'cols_cnt': 345}
'''

