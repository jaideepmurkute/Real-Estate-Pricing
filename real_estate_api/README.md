```
real_estate_api/
│
├── app/
|   ├── ml_utils/
|   |   ├── api_helper.py    # Fetches requested data and generates forecasts from ML models
|   |   └── utils.py         # Utility functions for the api_helper.py code
│   ├── __init__.py
│   ├── main.py              # Sets up api routes and calls appropriate functions in api_requestor to complete actions
|   ├── api_requestor.py     # Makes calls to api_helper with required config dict; performs basic response processing, if needed
│   ├── models.py
│   ├── schemas.py           # Defines the Pydantic schemas for data validation
│   ├── crud.py
│   ├── database.py
│   └── routers/
│       ├── __init__.py
│       ├── data.py
│       └── prediction.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── alembic.ini
```
