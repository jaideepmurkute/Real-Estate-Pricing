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

# Create the database tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this to your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(properties.router)
app.include_router(search.router)


# class DataResponse(BaseModel):
#     data: List[int]
#     forecast: int

class HistoricalData(BaseModel):
    Date: str
    Price: float

class ForecastResponse(BaseModel):
    historical: List[HistoricalData]
    forecast: float

   
@app.get("/")
def read_root():
    return {"message": "Welcome to the Real Estate Price Search and Prediction API"}

@app.get("/states", response_model=List[str])
def fetch_states():
    # Return a list of states
    # return {"message": "Welcome to the Real Estate Price Search and Prediction API"}
    # print("fetch_states called...")
    # return ["NY", "CA", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI"]
    states_list = state_list_requestor()
    # print("states_list: ", states_list)
    return states_list

@app.get("/regions", response_model=List[str])
def fetch_regions(state: str = Query(..., description="The state for which to fetch regions")):
    # Response: List of regions for the selected state
    # return ["New York, NY", "Chicago, IL"]
    # print("fetch_regions called... wit state: ", state)
    
    # state_region_map = {
    #     "NY": ["New York, NY", "Buffalo, NY"],
    #     "CA": ["Los Angeles, CA", "San Francisco, CA"],
    #     "TX": ["Houston, TX", "Dallas, TX"],
    #     "FL": ["Miami, FL", "Orlando, FL"],
    #     "IL": ["Chicago, IL", "Springfield, IL"],
    #     "PA": ["Philadelphia, PA", "Pittsburgh, PA"],
    #     "OH": ["Columbus, OH", "Cleveland, OH"],
    #     "GA": ["Atlanta, GA", "Savannah, GA"],
    #     "NC": ["Charlotte, NC", "Raleigh, NC"],
    #     "MI": ["Detroit, MI", "Grand Rapids, MI"]
    # }
    # return state_region_map[state]
    
    regions = state_regions_requestor(state) 
    # print("regions: ", regions)
    return regions
    
# @app.get("/data", response_model=DataResponse)
# def fetch_data_forecast(
#     state: str = Query(..., description="The state for which to fetch data"),
#     region: str = Query(..., description="The region for which to fetch data"),
#     feature: str = Query(..., description="The feature for which to fetch forecast")
#     ):
#     # console.log("fetch_data_forecast called...")
#     print("fetch_data_forecast called...")
#     # Response: Data for the last 6 months and forecasted value for the feature
#     # return {"message": "Welcome to the Real Estate Price Search and Prediction API"}
    
#     return {"data": [100, 200, 300, 400, 500, 600], "forecast": 700}
    

# @app.get("/data")
# async def fetch_data_forecast(state: str, region: str, feature: str):
#     print("fetch_data_forecast called...")
#     data = pd.DataFrame({
#         'Date': ['2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01'],
#         'Price': [100, 200, 300, 400, 500, 600]
#     })
#     return JSONResponse(content=data.to_dict(orient='records'))


@app.get("/data", response_model=ForecastResponse)
async def fetch_data_forecast(state: str, region: str, feature: str):
    # print("fetch_data_forecast called...")
    
    # Historical data
    # historical_data = pd.DataFrame({
    #     'Date': ['2021-01-01', '2021-02-01', '2021-03-01', '2021-04-01', '2021-05-01', '2021-06-01'],
    #     'Price': [100, 200, 300, 400, 500, 600]
    # })
    
    # Forecasted value (example)
    # forecast_value = 700
    forecast_value, historical_data = data_and_forecast_requestor(region, 'month', 6)
    
    # flip the order of the historical data
    historical_data = historical_data[::-1]
    
    
    # print("forecast_value: ", forecast_value)
    # print("historical_data: ", historical_data)
    
    # print("type(forecast_value): ", type(forecast_value))
    # print("historical_data.dtypes: ", historical_data.dtypes)
    
    '''
    forecast_value:  401906.1
    
    historical_data:  feature_name        Date     Price
    0             2024-01-31  567935.0
    1             2024-02-29  570319.0
    2             2024-03-31  574555.0
    3             2024-04-30  580204.0
    4             2024-05-31  591263.0
    5             2024-06-30  603073.0
    '''
    response = {
        "historical": historical_data.to_dict(orient='records'),
        "forecast": float(forecast_value)
    }
    
    # print("type(response['forecast']): ", type(response['forecast']))
    # for item in response["historical"]:
    #     print("type(item['Price']): ", type(item['Price']))
    
    return JSONResponse(content=response)