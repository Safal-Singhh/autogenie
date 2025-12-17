from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import os

app = FastAPI(title="AutoGenie ML Inference API")

# ----------------------------------
# Safe absolute paths (Render-ready)
# ----------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ----------------------------------
# Input schema
# ----------------------------------

class TelemetryInput(BaseModel):
    engine_temp: float
    load: float
    wear: float
    rpm: float

# ----------------------------------
# Routes
# ----------------------------------

@app.get("/")
def root():
    return {"status": "AutoGenie API is running"}

@app.post("/predict")
def predict_risk(data: TelemetryInput):
    input_data = np.array([[ 
        data.engine_temp,
        data.load,
        data.wear,
        data.rpm
    ]])

    scaled = scaler.transform(input_data)

    prediction = model.predict(scaled)[0]
    probability = model.predict_proba(scaled)[0][1]

    return {
        "risk": "HIGH" if int(prediction) == 1 else "LOW",
        "probability": round(float(probability), 2)
    }
