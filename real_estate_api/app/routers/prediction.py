from fastapi import APIRouter

router = APIRouter()

@router.get("/predict/")
def predict_price(state: str, region: str):
    # Placeholder for ML model prediction
    return {"state": state, "region": region, "predicted_price": 500000}

