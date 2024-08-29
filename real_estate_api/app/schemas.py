'''
    Define the Pydantic schemas for data validation.
    
    NOTE: Indiv
'''

from typing import List
from pydantic import BaseModel

class HistoricalData(BaseModel):
    Date: str
    Price: float

class ForecastResponse(BaseModel):
    historical: List[HistoricalData]
    forecast: float

class PropertyBase(BaseModel):
    address: str
    city: str
    state: str
    zip_code: str
    price: float
    bedrooms: int
    bathrooms: float
    sqft: int

class PropertyCreate(PropertyBase):
    pass

class Property(PropertyBase):
    id: int

    class Config:
        orm_mode = True