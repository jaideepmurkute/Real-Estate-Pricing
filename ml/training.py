
"""
    This file contains the code for training and evaluating the LSTM model for forecasting 
    the real estate prices.
    To be used for development and local model evaluation only. 

    __author__ = ""
    __email__ = ""
    __version__ = ""
"""

import os
from typing import Dict, Tuple, Optional, List, Any

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
import tensorflow as tf
from tqdm import tqdm

from .utils import *

# ---------------------------------------------------------------

def evaluate_model(model: tf.keras.models.Sequential, scaler: BaseEstimator, trainX: pd.DataFrame, 
                   trainY: pd.DataFrame, testX: pd.DataFrame, testY: pd.DataFrame, 
                   verbose: Optional[bool]=False) -> None:
    """
    Evaluate the performance of a machine learning model on training and testing data.
    Parameters:
        model (tf.keras.models.Sequential): The trained machine learning model.
        scaler (BaseEstimator): The scaler used to normalize the data.
        trainX (pd.DataFrame): The input features of the training data.
        trainY (pd.DataFrame): The target variable of the training data.
        testX (pd.DataFrame): The input features of the testing data.
        testY (pd.DataFrame): The target variable of the testing data.
    Returns:
        None
    """
    # Evaluate the model
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)

    # Inverse transform predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform(trainY)

    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform(testY)

    # Calculate root mean squared error
    trainScore = np.sqrt(np.mean((trainY - trainPredict) ** 2))
    testScore = np.sqrt(np.mean((testY - testPredict) ** 2))
    if verbose:
        print('Train Score: %.2f RMSE' % trainScore)
        print('Test Score: %.2f RMSE' % testScore)
    

class QuietCallback(tf.keras.callbacks.Callback):
    '''
        A callback class to suppress the output of the model training.
    '''
    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

def train_model(config: Dict[str, Any], df: pd.DataFrame, verbose: bool = False) -> None:
    """
    Train the machine learning model on the given dataframe.
    Parameters:
        config (Dict[str, Any]): The configuration dictionary.
        df (pd.DataFrame): The input dataframe.
        verbose (bool): Whether to print verbose output. Default is False.
    Returns:
        None
    """
    fold_split_dict = get_fold_splits(config, df)
    for fold_idx, (train_idx, val_idx) in fold_split_dict.items():
        if verbose: 
            print("Training fold: ", fold_idx)
        
        train_df, val_df = df.iloc[train_idx], df.iloc[val_idx]

        # Scale the data
        train_df, scaler = scale_data(train_df, scaler=None)
        val_df = scale_data(val_df, scaler=scaler)
        save_scaler(config, scaler, fold_idx)
        
        # create dataset
        # trainX: sequence of look_back value chunks 
        # trainY: next timestep following sequences in trainX
        trainX, trainY = create_dataset(train_df, config['look_back'])
        valX, valY = create_dataset(val_df, config['look_back'])

        # create model
        model = get_model(config, trainX)

        if verbose:
            print("BEFORE fitting the model performance: ")
            evaluate_model(model, scaler, trainX, trainY, valX, valY)

        # Training
        model.fit(trainX, trainY, epochs=config['n_epochs'], 
                    batch_size=config['train_batch_size'], verbose=0, 
                    callbacks=[QuietCallback()])
        
        # Persistence
        save_model(config, model, fold_idx)
        
        if verbose:
            print("AFTER fitting the model performance: ")
            evaluate_model(model, scaler, trainX, trainY, valX, valY)

        if verbose:
            print("Finished fold.....")
            print('-'*30)
            print('-'*30)

# @tf.function(reduce_retracing=True)
def predict_with_model(model, X):
    return model.predict(X)
    
def test_model(config, test_df, train_df=None):
    
    per_fold_preds = []
    for fold_idx in range(config['n_folds']):
        scaler = load_scaler(config, fold_idx)
        testX = scale_data(test_df, scaler=scaler)
        X = create_test_dataset(config, testX)
        
        model = load_model(config, fold_idx)
        
        # ------------------
        dataset = tf.data.Dataset.from_tensor_slices(X).batch(config['eval_batch_size']).prefetch(tf.data.AUTOTUNE)
        
        preds = []
        for batch in dataset:
            batch_preds = predict_with_model(model, batch)
            preds.append(batch_preds)
        
        preds = np.concatenate(preds, axis=0)
        # ------------------
        preds = scaler.inverse_transform(preds)
        per_fold_preds.append(preds)
    
    per_fold_preds = np.concatenate(per_fold_preds, axis=0)
    
    aggr_preds = np.average(per_fold_preds, axis=0)
    
    return aggr_preds


# ---------------------------------------------------------------

if __name__ == '__main__':
    config = {
        'choice': 1, 
        'region_name': 'New York, NY', 
        'granularity': 'month',
        'look_back': 6,   # these many timesteps to predict the next timestep; do consider granularity
        'n_folds': 3, 
        'n_epochs': 100,
        'train_batch_size': 64,
        'eval_batch_size': 64, 
        
        'data_types': ['sale_price', 'pct_sold_above_list', 'pct_sold_below_list', 'sale_to_list_ratio', 
                                    'market_heat_index', 'home_value_index', 'home_value_forecast'],
        'handle_outliers': False,
        'seed': 42, 
        
        'data_dir': os.path.join('..', 'data', 'zillow'), 
        'data_viz_dir': os.path.join('..', 'data_viz', 'zillow'),
        'feature_dir': os.path.join('..', 'features'),
        'region_data_store_dir': os.path.join('..', 'data', 'zillow', 'region_data_store'),
        'model_store_dir': os.path.join('model_store'),
    }


    housekeeping(config)

    if config['choice'] == 1:
        for region_name in tqdm(get_available_regions(config)):
            config['region_name'] = region_name
            housekeeping(config) # calling again to update paths.
            
            # ----------------------
            if os.path.exists(config['region_model_store_dir']) and \
                        len(os.listdir(config['region_model_store_dir'])) > 0:
                print(f"Model already exists for {config['region_name']}. Skipping....")
                continue
            
            # ----------------------
            
            df = get_region_data(config, 'month', config['region_name'])
            df, feature_names = preprocess_dataframe(df)
            train_model(config, df, verbose=False)  
    if config['choice'] == 2:
        train_df = get_region_data(config, 'month', config['region_name'])
        train_df, feature_names = preprocess_dataframe(train_df)
        
        # '2024-06-30' - string in the format 'YYYY-MM-DD'
        forecast_date = train_df.index[-1]  # can be passed by the user
        
        if forecast_date > train_df.index[-1]:
            forecast_date = train_df.index[-1]
        
        # take the slice from train_df that is needed for forecasting
        forecast_date_idx = train_df.index.get_loc(forecast_date)
        train_df = train_df[forecast_date_idx-config['look_back']+1:]
        
        aggr_preds = test_model(config, train_df)
        aggr_preds_df = pd.Series(aggr_preds, index=feature_names)
        
        # can use following: index for : 
        # 'Metro_median_sale_price_uc_sfrcondo_sm_sa_month'
        # 'Metro_mean_sale_price_uc_sfrcondo_sm_sa_month'
        print("prediction: ", aggr_preds_df['Metro_median_sale_price_uc_sfrcondo_sm_sa_month'])
    else:
        pass


