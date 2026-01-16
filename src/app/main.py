from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from app.model_loader import load_xgb_final

app = FastAPI()

class Request(BaseModel):
    ticker: str
    features: dict

@app.post("/predict")
def predict(req: Request):
    model, scaler, feature_names = load_xgb_final(req.ticker)
    x = [req.features[f] for f in feature_names]
    x = scaler.transform([x])
    pred = model.predict(x)[0]
    return {"prediction": int(pred)}
