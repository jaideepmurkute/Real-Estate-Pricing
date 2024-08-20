'''
    Define the Pydantic schemas for data validation.
'''

from pydantic import BaseModel

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