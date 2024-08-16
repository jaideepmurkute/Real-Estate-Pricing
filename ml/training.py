
import os
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


# Load the data
config = {
    'feature_dir': os.path.join('..', 'features'),
    'model_store_dir': os.path.join('..', 'model_store'),
}
if not os.path.exists(config['feature_dir']):
    raise FileNotFoundError(config['feature_dir'])
if not os.path.exists(config['model_store_dir']):
    os.makedirs(config['model_store_dir'])


feature_set_df = pd.read_csv(os.path.join(config['feature_dir'], 'feature_set_test.csv'))
feature_cols = [col for col in feature_set_df.columns if 'feat' in col]
target_col = 'feat_2_SalePrice_Median_Month'



