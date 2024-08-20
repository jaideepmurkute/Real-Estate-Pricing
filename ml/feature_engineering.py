
import os
from typing import Dict
import pandas as pd
import numpy as np

from utils import *



def perform_data_cleaning(config: Dict, col_desc: Dict, df: pd.DataFrame, ) -> pd.DataFrame:
    # reduce memory usage
    df = reduce_mem_usage(df, verbose=False)

    # Fill infinite values with np.nan
    if has_inf(df, cols=col_desc['non_id_cols']):
        df = fill_inf(df, cols=col_desc['non_id_cols'], 
                                    method='value', fill_value=np.nan, verbose=False)

    # Drop regions with more than 60% missing(NaN) values
    if has_nan(df, cols=col_desc['non_id_cols']):
        df = drop_missing_val_regions(df, cols=col_desc['non_id_cols'], 
                                                    threshold=0.6, verbose=False)

    # Fill missing values in the specified columns using linear interpolation
    df = fill_nan_prices(df, cols=col_desc['non_id_cols'], 
                                        method='interpolate', verbose=False)

    # handle outliers
    if config['handle_outliers']:
        df = handle_outliers(df, cols=col_desc['non_id_cols'], method='winsorize',
                                            winsorize_limits=(0.05, 0.05), verbose=False)

    return df


# -------------------------------------------------------------------------------------

config = {
    'data_dir': os.path.join('..', 'data', 'zillow'), 
    'data_viz_dir': os.path.join('..', 'data_viz', 'zillow'),
    'feature_dir': os.path.join('..', 'features'),
    'handle_outliers': False,
}

if not os.path.exists(config['data_dir']):
    raise FileNotFoundError(config['data_dir'])
if not os.path.exists(config['data_viz_dir']):
    os.makedirs(config['data_viz_dir'])
if not os.path.exists(config['feature_dir']):
    os.makedirs(config['feature_dir'])
    

feature_set_df = pd.DataFrame()

df = get_data(config, 'sale_price', 'median', True, True, 'month')
col_desc = get_col_descriptor(df)
df = perform_data_cleaning(config, col_desc, df)
date_cols = list(df.columns)
for col in df.columns:
    if col in col_desc['non_id_cols']:
        df.rename(columns={col: 'feat_1_' + col}, inplace=True)
df = df.T.reset_index()
print(df.head(n=10))
raise
print("Feature set feat_1_ shape:", df.shape)
feature_set_df = pd.concat([feature_set_df, df], axis=1)
print("After concat feature_set_df.shape:", feature_set_df.shape)

print('-'*30)
# -------------------------------

df = get_data(config, 'sale_to_list_ratio', price_to_use='mean', smoothing=True, seasonal_adjustment=False, 
              granularity='month')
col_desc = get_col_descriptor(df)
df = perform_data_cleaning(config, col_desc, df)
print("Feature set feat_2_ shape:", df.shape)
# drop columns that are not in the feature set_df currently
df = df.drop(columns=[col for col in df.columns if col not in date_cols])
print("After dropping columns not in Feature set: feat_2_ shape:", df.shape)

for col in df.columns:
    if col in col_desc['non_id_cols']:
        df.rename(columns={col: 'feat_2_' + col}, inplace=True)

feature_set_df = pd.merge(feature_set_df, df, on='RegionID', how='inner')
print("After concat feature_set_df.shape:", feature_set_df.shape)
# -------------------------------

feature_set_df.to_csv(os.path.join(config['feature_dir'], 'feature_set_test.csv'), index=False)


