
'''
    Module to handle the API requests for the real estate pricing data and forecasting; 
    from the ml module - which interacts with the data, models and generates forecasts.
    
    Functions in this module are called by the FastAPI endpoints in the main.py module.
    
    __author__ = ''
    __email__ = ''
    __version__ = '1.0.0'
'''
import os
import sys
from typing import Dict, Optional, Tuple, List, Union

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# import ml.api_helper as api_helper
from .ml_utils import api_helper as api_helper

def add_paths(config):
    # For non-dockerized version
    # config['data_dir'] = os.path.join('..', 'data', 'zillow')
    # config['region_data_store_dir'] = os.path.join(config['data_dir'], 'region_data_store')
    # config['model_store_dir'] = os.path.join('..', 'ml', 'model_store')

    # for Docker deployment
    config['data_dir'] = os.path.join('data', 'zillow')
    config['region_data_store_dir'] = os.path.join(config['data_dir'], 'region_data_store')
    config['model_store_dir'] = os.path.join('model_store')
    
    return config


def generate_prediction_config(region_name: str, granularity: str, look_back: int) -> Dict:
    '''
        Generate the configuration dictionary for the ml.prediction module
        
        # NOTE: paths in this dict should be relative to the location of the ml module's script since this is being 
        # passed to the ml module
    '''
    # current_file_dir = os.path.dirname(os.path.abspath(__file__))
    config = {
        'region_name': region_name, #'Adrian, MI', 
        'granularity': granularity, # 'month',
        'look_back': look_back, # 6, 
        'return_data_type': 'sale_price',
        'return_aggr_type': 'median',
        'n_folds': 3, 
        
        'eval_batch_size': 64, 
        'handle_outliers': False,
        
        'seed': 42, 
    }
    
    config = add_paths(config)
    
    return config

def get_generatal_config() -> Dict:
    config = add_paths({})
    return config
    
def data_and_forecast_requestor(state: str, region_name: str, feature: str, granularity: str, 
                                look_back: int) -> int:
    """
        Handles Forecasting calls for the real estate pricing for a given region.
        Generates the prediction configuration and calls the ml.prediction module to get the forecast value.
    
    Args:
        region_name (str): The name of the region.
        granularity (str): The granularity of the forecast (e.g., 'monthly', 'quarterly', 'yearly').
        look_back (int): The number of periods to look back for historical data.
    Returns:
        int: The predicted real estate pricing for the given region.
    """
    
    config = generate_prediction_config(region_name, granularity, look_back)
    api_helper.housekeeping(config)
    api_helper.check_input(config)
    return api_helper.predict(config, feature, return_data=True)

def state_list_requestor() -> List[str]:
    """
        Handles the request for the list of states.
        Calls the ml.api_helper module to get the list of states.
    
    Returns:
        List[str]: The list of states.
    """
    config = get_generatal_config()
    return api_helper.get_states_list(config)

def state_regions_requestor(state: str) -> List[str]:
    """
        Handles the request for the list of regions for a given state.
        Calls the ml.api_helper module to get the list of regions for the given state.
    
    Args:
        state (str): The name of the state.
    Returns:
        List[str]: The list of regions for the given state.
    """
    config = get_generatal_config()
    return api_helper.get_state_regions(config, state)

def feature_list_requestor(state: str, region: str) -> List[str]:
    """
        Handles the request for the list of supported features; given state and regions.
        Calls the ml.api_helper module to get the list of features.
    
    Returns:
        List[str]: The list of features.
    """
    config = get_generatal_config()
    return api_helper.get_features_list(config, state, region)


