import os
import sys

import pandas as pd
import pytest
from unittest.mock import patch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api_helper import check_input, update_forecast_date


# ----------------- test check_input() -----------------
def test_check_input_valid():
    config = {'region_name': 'valid_region'}
    
    with patch('api_helper.get_available_regions') as mock_get_available_regions:
        mock_get_available_regions.return_value = ['valid_region', 'another_region']
        
        # This should not raise any exception
        check_input(config)
        mock_get_available_regions.assert_called_once_with(config)

def test_check_input_invalid():
    config = {'region_name': 'invalid_region'}
    
    with patch('api_helper.get_available_regions') as mock_get_available_regions:
        mock_get_available_regions.return_value = ['valid_region', 'another_region']
        
        # This should raise a ValueError
        with pytest.raises(ValueError) as excinfo:
            check_input(config)
        
        assert str(excinfo.value) == "Region name: invalid_region not found in available regions: ['valid_region', 'another_region']"
        mock_get_available_regions.assert_called_once_with(config)

# ----------------- test update_forecast_date -----------------


@pytest.fixture
def train_df():
    dates = pd.date_range(start='2022-01-01', end='2022-01-10')
    # remove the 5th date
    dates = dates.delete(4)
    data = {'value': range(len(dates))}
    df = pd.DataFrame(data, index=dates)
    # print(df.dtypes)
    # print(df)
    # print(df.index.dtype)
    
    return df

def test_forecast_date_none(train_df):
    result = update_forecast_date({}, train_df, None)
    assert result == train_df.index[-1], "Should return the last date in the train_df"

def test_forecast_date_less_than_first_date(train_df):
    result = update_forecast_date({}, train_df, '2021-12-31')
    assert result == train_df.index[0], "Should return the first date in the train_df"

def test_forecast_date_greater_than_last_date(train_df):
    result = update_forecast_date({}, train_df, '2022-01-11')
    assert result == train_df.index[-1], "Should return the last date in the train_df"

def test_forecast_date_not_present_in_train_df(train_df):
    result = update_forecast_date({}, train_df, '2022-01-05')
    print("result: ", result)
    assert result in [pd.Timestamp('2022-01-04'), 
                      pd.Timestamp('2022-01-06')], "Should return the nearest date in the train_df"
    
def test_forecast_date_present_in_train_df(train_df):
    result = update_forecast_date({}, train_df, '2022-01-03')
    result = pd.Timestamp(result)
    assert isinstance(result, pd.Timestamp), "Should return a pandas nearest value Timestamp object"    


