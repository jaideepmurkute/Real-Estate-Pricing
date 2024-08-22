'''
The entry point for our FastAPI application
'''

from fastapi import FastAPI, Query
from .routers import properties, search
from .database import engine, Base
from typing import List, Dict
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
from fastapi.responses import JSONResponse
from .api_requestor import *

# ----------------- Database Setup ----------------- #
# Create the database tables
Base.metadata.create_all(bind=engine)

# ----------------- FastAPI App ----------------- #
app = FastAPI()

# ----------------- Middleware ----------------- #
# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Routers ----------------- #
# Include routers
app.include_router(properties.router)
app.include_router(search.router)

# ----------------- Pydantic Models ----------------- #
class HistoricalData(BaseModel):
    Date: str
    Price: float

class ForecastResponse(BaseModel):
    historical: List[HistoricalData]
    forecast: float

# ----------------- API Endpoints ----------------- #
   
@app.get("/")
def read_root():
    return {"message": "Welcome to the Real Estate Price Search and Forecasting API"}

@app.get("/states", response_model=List[str])
def fetch_states():
    # Response: A list of supported states.
    states_list = state_list_requestor()
    return states_list

'''
Using FastAPI's Query helps you:
    Validate Input: Ensure that the input meets certain criteria (e.g., minimum length, pattern).
    Provide Metadata: Add descriptions, titles, and examples for better API documentation.
    Set Defaults: Define default values for query parameters.

FastAPI's Query Parameters:
    ... : indicates that the parameter is required
    description: parameter description which can be displayed in API documentation
    default: default value for the parameter
'''
@app.get("/regions", response_model=List[str])
def fetch_regions(state: str = Query(..., description="The state for which to fetch regions")):
    # Response: A list of regions for the selected state.
    regions = state_regions_requestor(state) 
    return regions

@app.get("/features", response_model=List[str])
def fetch_features(state: str = Query(..., description="The state for which to fetch supported features"), 
                   region: str = Query(..., description="The region for which to fetch supported features")):
    # Response: A list of supported features; given state and region.
    features_list = feature_list_requestor(state, region)
    # print("API main: features_list: ", features_list)
    return features_list

@app.get("/data", response_model=ForecastResponse)
async def fetch_data_forecast(state: str, region: str, feature: str):
    forecast_value, historical_data = data_and_forecast_requestor(state, region, feature, 
                                                            granularity='month', look_back=6)
    # if historical_data is None:
    #     # historical_data = pd.DataFrame(data=[['Unavailable', 'Unavailable']], columns=['Date', 'Price'])  # Create an empty DataFrame with the required columns
    #     historical_data = pd.DataFrame({'Date': [], 'Price': ['Unavailable']})
    # else:
    
    historical_data = historical_data[::-1]  # reverse the data order to show the latest data first
    
    
    response = {
        "historical": historical_data.to_dict(orient='records'),
        "forecast": float(forecast_value)
    }
    
    return JSONResponse(content=response)