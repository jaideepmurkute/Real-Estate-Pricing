import pytest
from unittest.mock import patch, MagicMock
import sys
import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.api_requestor import (
    data_and_forecast_requestor,
    state_list_requestor,
    state_regions_requestor,
    feature_list_requestor,
    generate_prediction_config,
    add_paths,
    get_generatal_config
)

# Mocking the api_helper module
@patch('app.api_requestor.api_helper')
def test_data_and_forecast_requestor(mock_api_helper):
    # Arrange
    state = 'MI'
    region_name = 'Adrian, MI'
    feature = 'sale_price'
    granularity = 'month'
    look_back = 6

    mock_api_helper.housekeeping.return_value = None
    mock_api_helper.check_input.return_value = None
    mock_api_helper.predict.return_value = 12345

    # Act
    result = data_and_forecast_requestor(state, region_name, feature, granularity, look_back)

    # Assert
    assert result == 12345
    mock_api_helper.housekeeping.assert_called_once()
    mock_api_helper.check_input.assert_called_once()
    mock_api_helper.predict.assert_called_once()

@patch('app.api_requestor.api_helper')
def test_state_list_requestor(mock_api_helper):
    # Arrange
    mock_api_helper.get_states_list.return_value = ['MI', 'CA', 'NY']

    # Act
    result = state_list_requestor()

    # Assert
    assert result == ['MI', 'CA', 'NY']
    mock_api_helper.get_states_list.assert_called_once()

@patch('app.api_requestor.api_helper')
def test_state_regions_requestor(mock_api_helper):
    # Arrange
    state = 'MI'
    mock_api_helper.get_state_regions.return_value = ['Adrian', 'Detroit', 'Ann Arbor']

    # Act
    result = state_regions_requestor(state)

    # Assert
    assert result == ['Adrian', 'Detroit', 'Ann Arbor']
    mock_api_helper.get_state_regions.assert_called_once_with(get_generatal_config(), state)

@patch('app.api_requestor.api_helper')
def test_feature_list_requestor(mock_api_helper):
    # Arrange
    state = 'MI'
    region = 'Adrian'
    mock_api_helper.get_features_list.return_value = ['sale_price', 'rent_price']

    # Act
    result = feature_list_requestor(state, region)

    # Assert
    assert result == ['sale_price', 'rent_price']
    mock_api_helper.get_features_list.assert_called_once_with(get_generatal_config(), state, region)