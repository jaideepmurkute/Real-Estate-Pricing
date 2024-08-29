from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .database import engine, Base                                              
from .api_requestor import *
from .schemas import ForecastResponse # Pydantic validation models

# ----------------- Database Setup ----------------- #
# Create the database tables
Base.metadata.create_all(bind=engine)

# ----------------- FastAPI App ----------------- #
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- API Endpoints ----------------- #

@app.get("/")
def read_root():
    '''
        Index route handler with dummy message.
    '''
    return {"message": "Welcome to the Real Estate Price Search and Forecasting API"}


@app.get("/states", response_model=List[str])
def fetch_states():
    """
    Returns a list of supported states.

    Returns:
        List[str]: A list of supported states.
    """
    states_list = state_list_requestor()
    return states_list


@app.get("/regions", response_model=List[str])
def fetch_regions(state: str = Query(..., description="The state for which to fetch regions")):
    """
    Returns a list of regions for the selected state.

    Parameters:
    - state (str): The state for which to fetch regions.

    Returns:
    - List[str]: A list of regions for the selected state.
    """
    regions = state_regions_requestor(state) 
    return regions

@app.get("/features", response_model=List[str])
def fetch_features(state: str = Query(..., description="The state for which to fetch supported features"), 
                   region: str = Query(..., description="The region for which to fetch supported features")):
    '''
        Returns a list of supported features; given state and region.
    
    Parameters:
    - state (str): The state for which to fetch supported features.
    - region (str): The region for which to fetch supported features.
    
    Returns:
    - List[str]: A list of supported features.
    '''
    features_list = feature_list_requestor(state, region)
    return features_list
    
@app.get("/data", response_model=ForecastResponse)
async def fetch_data_forecast(state: str, region: str, feature: str):
    """
    Fetches historical data and forecast value for the next month based on the provided state, region, and feature.
    
    Parameters:
    - state (str): The state for which the data and forecast are requested.
    - region (str): The region within the state for which the data and forecast are requested.
    - feature (str): The feature for which the data and forecast are requested.
    
    Returns:
    - JSONResponse: A JSON response containing the historical data and forecast value.
        - historical (list): A list of dictionaries representing the historical data.
        - forecast (float): The forecast value for the next month.
    """
    forecast_value, historical_data = data_and_forecast_requestor(state, region, feature, 
                                                            granularity='month', look_back=6)
    
    historical_data = historical_data[::-1]  # reverse the data order to show the latest data first
    
    response = {
        "historical": historical_data.to_dict(orient='records'),
        "forecast": float(forecast_value)
    }
    
    return JSONResponse(content=response)