'''
The entry point for our FastAPI application
'''

from fastapi import FastAPI
from .routers import properties, search
from .database import engine, Base

# Create the database tables
Base.metadata.create_all(bind=engine)

app = FastAPI()

# Include routers
app.include_router(properties.router)
app.include_router(search.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Real Estate Price Search and Prediction API"}