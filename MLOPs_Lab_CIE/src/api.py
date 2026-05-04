from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("models/model.pkl")

class InputData(BaseModel):
    txn_amount: float = Field(..., ge=100, le=1000000)
    merchant_risk_score: float = Field(..., ge=0.1, le=1.0)
    is_international: int = Field(..., ge=0, le=1)
    gateway_load: float = Field(..., ge=0.1, le=1.0)

@app.get("/ping")
def ping():
    return {"status": "operational", "service": "PayFlow API"}

@app.post("/infer")
def infer(data: InputData):
    features = np.array([[ 
        data.txn_amount,
        data.merchant_risk_score,
        data.is_international,
        data.gateway_load
    ]])
    pred = model.predict(features)[0]
    return {"prediction": float(pred)}