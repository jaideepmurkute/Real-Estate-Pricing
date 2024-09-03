
'''
    Utility functions for the machine learning pipeline.
    
    __author__ = ""
    __email__ = ""
    __version__ = ""
'''

import os
from typing import Dict, Optional, Tuple, List, Union

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit

import tensorflow as tf
import pickle

# ---------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Set the seed for reproducibility.
    
    Parameters:
        seed (int): The seed value.
    
    Returns:
        None
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def housekeeping(config: Dict) -> None:
    """
    Perform necessary housekeeping tasks such as creating directories and setting seed.
    
    Parameters:
        config (Dict): A dictionary containing configuration parameters.
    
    Returns:
        None
    """
    if not os.path.exists(config['data_dir']):
        raise FileNotFoundError(config['data_dir'])
    
    if 'data_viz_dir' in config:
        if not os.path.exists(config['data_viz_dir']):
            os.makedirs(config['data_viz_dir'])
    if 'feature_dir' in config:
        if not os.path.exists(config['feature_dir']):
            os.makedirs(config['feature_dir'])
    
    if not os.path.exists(config['model_store_dir']):
        os.makedirs(config['model_store_dir'])
    
    config['region_model_store_dir'] = os.path.join(config['model_store_dir'], config['region_name'])
    if not os.path.exists(config['region_model_store_dir']):
        os.makedirs(config['region_model_store_dir'])
    
    set_seed(config['seed'])

def get_available_regions(config: Dict) -> List[str]:
    """
    Get a list of available regions based on the configuration.
    
    Parameters:
        config (Dict): A dictionary containing configuration parameters.
    
    Returns:
        List[str]: A list of available regions.
    """
    region_data_fnames = os.listdir(config['region_data_store_dir'])
    region_data_fnames = [fname for fname in region_data_fnames if config['granularity'] in fname]
    
    regions_lst = []
    for fname in region_data_fnames:
        region_name = fname.split(config['granularity']+'_')[1].split('.csv')[0]
        regions_lst.append(region_name)
    
    return regions_lst


def get_region_data(config: Dict, granularity: str, region_name: str) -> pd.DataFrame:
    """
    Get the region data for a specific granularity and region name.
    
    Parameters:
        config (Dict): A dictionary containing configuration parameters.
        granularity (str): The granularity of the data.
        region_name (str): The name of the region.
    
    Returns:
        pd.DataFrame: The region data.
    """
    fpath = os.path.join(config['region_data_store_dir'], f'all_data_df_{granularity}_{region_name}.csv')
    if not os.path.exists(fpath):
        raise FileNotFoundError(fpath)
    print(f"Reading data from {fpath}")
    df = pd.read_csv(fpath, index_col=0)
    
    return df

def scale_data(data: np.ndarray, scaler: Optional[BaseEstimator]=None) -> Tuple[np.ndarray, BaseEstimator]:
    """
    Scale the input data using a scaler.
    
    Parameters:
        data (np.ndarray): The input data to be scaled.
        scaler (Optional[BaseEstimator], optional): The scaler object to be used for scaling. 
            If None, a new scaler will be created. Defaults to None.
    
    Returns:
        Tuple[np.ndarray, BaseEstimator]: The scaled data and the scaler object.
    """
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        return scaler.fit_transform(data), scaler
    return scaler.transform(data)

def create_dataset(dataset: np.ndarray, look_back: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create the input and output datasets for training a time series model.
    
    Parameters:
        dataset (np.ndarray): The input dataset.
        look_back (int): The number of previous time steps to use as input features.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: The input and output datasets.
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)


def create_test_dataset(config: Dict, dataset: pd.DataFrame, train_df: Optional[pd.DataFrame]=None) -> np.ndarray:
    """
    Create a dataset for the sequence model from the input dataframe.
    Args:
        config (Dict): A dictionary containing configuration parameters.
        dataset (pd.DataFrame): The input dataframe.
        train_df (Optional[pd.DataFrame]): The train dataframe (default: None).
    Returns:
        np.ndarray: The created dataset for inference.
    Raises:
        ValueError: If the dataset does not have enough samples for the look_back period and train_df is None.
    """
    # Check if the dataset has enough samples for the look_back period
    if len(dataset) < config['look_back']:
        if train_df is None:
            print("train_df is None in `create_test_dataset`. Need train_df to accomodate the look_back...")
            raise ValueError
        
        # Calculate the number of samples needed from train_df
        required_samples = config['look_back'] - len(dataset)
        # Get the tail data from train_df
        tail_data = train_df[-required_samples:]
        # Concatenate the tail data with the dataset
        dataset = np.concatenate((tail_data, dataset), axis=0)
    
    # Create the dataset for inference
    dataX = []
    for i in range(len(dataset) - config['look_back'] + 1):
        a = dataset[i:(i + config['look_back']), :]
        dataX.append(a)
    
    return np.array(dataX)


def get_model(config: Dict, train_X: pd.DataFrame, train_Y: Optional[pd.DataFrame]=None, 
        loss: Optional[str]='mean_squared_error', optimizer: Optional['str']='adam', 
        metrics: Optional['str']=['mean_squared_error']) -> tf.keras.models.Sequential:
    """
    Creates and compiles a tf.keras sequential model.
    Parameters:
        config (Dict): A dictionary containing configuration parameters.
        train_X (pd.DataFrame): The input training data.
        train_Y (Optional[pd.DataFrame], optional): The target training data. 
                Defaults to None.
        loss (Optional[str], optional): The loss function to be used during training. 
                Defaults to 'mean_squared_error'.
        optimizer (Optional[str], optional): The optimizer to be used during training. 
                Defaults to 'adam'.
        metrics (Optional[str], optional): The evaluation metrics to be used during training. 
                Defaults to ['mean_squared_error'].
    Returns:
        tf.keras.models.Sequential: The compiled sequential model.
    """
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Input
    
    model = Sequential()
    model.add(Input(shape=(config['look_back'], train_X.shape[2])))
    model.add(LSTM(50, return_sequences=True, activation='sigmoid'))
    model.add(LSTM(50, return_sequences=False, activation='sigmoid'))
    model.add(BatchNormalization())
    model.add(Dense(10, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(train_X.shape[2]))
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    
    return model

def adjust_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adjusts the index of a DataFrame - reset and rename the current index with feature_name column.
    Assign a new numeric index for compatibility with other data pre-processing functions.
    
    Parameters:
    df (Dict): The DataFrame to be adjusted.
    Returns:
    Dict: The DataFrame with adjusted index.
    """
    
    df.reset_index(drop=False, inplace=True)
    df.rename(columns={'index': 'feature_name'}, inplace=True)
    new_index = np.arange(0, df.shape[0])
    df.set_index(new_index, inplace=True)
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in a DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: The DataFrame with missing values handled.
    """
    # drop the rows with nan values are more than 60%
    nan_cnt_per_row = df.isnull().sum(axis=1)
    df = df.loc[nan_cnt_per_row < 0.6 * df.shape[1]]

    date_cols = df.columns.tolist()[1:]
    
    # fill intermedite missing values using cubic interpolation
    df.loc[:, date_cols] = df[date_cols].interpolate(method='linear', axis=0, inplace=False)

    # fill remaining missing values using forward and backward fill
    df.loc[:, date_cols] = df[date_cols].ffill(axis=1, inplace=False)
    df.loc[:, date_cols] = df[date_cols].bfill(axis=1, inplace=False)
    
    return df

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the input dataframe by adjusting the index, handling missing values,
    dropping unwanted columns, and adjusting shapes for model compatibility.
    
    Args:
        df (Dict): The input dataframe.
    Returns:
        pd.DataFrame: The preprocessed dataframe.
    """
    
    # adjust index
    df = adjust_index(df)

    # fill missing values
    df = handle_missing_values(df)
    
    # drop unwanted columns
    # date_cols = df.columns.tolist()[1:]
    feature_names = df['feature_name']
    
    df = df.drop('feature_name', axis=1)
    
    
    # adjust shapes for model compatibility    
    df = df.T  # (features, time) -> (time, features)

    return df, feature_names

def save_model(config: Dict, model: tf.keras.models.Sequential, fold_idx: Optional[int]=None) -> None:
    """
    Save the trained model to a file.
    Args:
        config (Dict): A dictionary containing configuration parameters.
        model (tf.keras.models.Sequential): The trained model to be saved.
        fold_idx (Optional[int], optional): The fold index. Defaults to None.
    Returns:
        None
    """
    if not isinstance(model, tf.keras.models.Sequential):
        raise ValueError("The model should be an instance of tf.keras.models.Sequential class.")
    if not isinstance(fold_idx, (int, type(None))):
        raise ValueError("The fold index should be an integer or None.")
    if not isinstance(config, dict):
        raise ValueError("The config should be a dictionary.")

    if fold_idx is None:
        fold_idx = ''
    
    save_fname = f'model_{config['region_name']}_{config['granularity']}_fold_{fold_idx}'
    
    save_fpath = os.path.join(config['region_model_store_dir'], save_fname)
    extension = 'keras'
    
    model.save(save_fpath + '.' + extension)
  
def load_model(config: Dict, fold_idx: Optional[int]=None) -> tf.keras.models.Sequential:
    """
    Load a trained tf.keras model from a file.
    Args:
        config (Dict): A dictionary containing configuration parameters.
        fold_idx (int): The fold index of the model to load.
    Returns:
        tf.keras.models.Sequential: The loaded model.
    """
    if not isinstance(config, dict):
        raise ValueError("The config should be a dictionary.")
    if not isinstance(fold_idx, (int, type(None))):
        raise ValueError("The fold index should be an integer or None.")
    
    if fold_idx is None:
        fold_idx = ''
    save_fname = f'model_{config['region_name']}_{config['granularity']}_fold_{fold_idx}'
    save_fpath = os.path.join(config['region_model_store_dir'], save_fname)
    extension = 'keras'
    
    save_fpath = save_fpath + '.' + extension
    if not os.path.exists(save_fpath):
        raise FileNotFoundError(f"Model file {save_fpath} not found.")
    
    model = tf.keras.models.load_model(save_fpath)
    return model

def save_scaler(config: Dict, scaler: BaseEstimator, fold_idx: int) -> None:
    """
    Save the scikit-learn scaler object to a file with pickle.
    Parameters:
        config (Dict): A dictionary containing configuration parameters.
        scaler (BaseEstimator): The scaler object to be saved.
        fold_idx (int): The index of the fold.
    Returns:
        None
    """
    
    save_fname = f'scaler_{config['region_name']}_{config['granularity']}_fold_{fold_idx}'
    save_fpath = os.path.join(config['region_model_store_dir'], save_fname)
    extension = 'pkl'
    
    with open(save_fpath+'.'+extension, 'wb') as fp:
        pickle.dump(scaler, fp)
     
def load_scaler(config: Dict, fold_idx: int) -> BaseEstimator:
    """
    Load a scikit-learn scaler object from a saved file.
    Parameters:
        config (Dict): A dictionary containing configuration parameters.
        fold_idx (int): The index of the fold.
    Returns:
        BaseEstimator: The loaded scaler object.
    """
    save_fname = f'scaler_{config['region_name']}_{config['granularity']}_fold_{fold_idx}'
    save_fpath = os.path.join(config['region_model_store_dir'], save_fname)
    extension = 'pkl'
    
    with open(save_fpath+'.'+extension, 'rb') as fp:
        scaler = pickle.load(fp)
    
    return scaler


def get_fold_splits(config, df):
    tscv = TimeSeriesSplit(n_splits=config['n_folds'])
    splits = tscv.split(df)
    fold_split_dict = {}
    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        fold_split_dict[fold_idx] = (train_idx, val_idx)
    
    return fold_split_dict

