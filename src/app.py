import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException

from src.request import PredictRequest

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")
SCALER_PATH = os.environ.get("SCALER_PATH", "models/scaler.joblib")
ENCODER_PATH = os.environ.get("ENCODER_PATH", "models/encoder.joblib")

app = FastAPI(title="Iris Classifier", version="0.1")

@app.on_event("startup")
def load_model():
    global MODEL, SCALER, ENCODER
    
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    MODEL = joblib.load(MODEL_PATH)
    
    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(f"Scaler file not found at {SCALER_PATH}")
    SCALER = joblib.load(SCALER_PATH)
    
    if not os.path.exists(ENCODER_PATH):
        raise RuntimeError(f"Encoder file not found at {ENCODER_PATH}")
    ENCODER = joblib.load(ENCODER_PATH)
    
    print(f"Loaded model with species classes: {ENCODER.classes_}")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: PredictRequest):
    try:
        x = np.array([[
            payload.sepal_length,
            payload.sepal_width,
            payload.petal_length,
            payload.petal_width
        ]], dtype=float)
        
        x_scaled = SCALER.transform(x)
        
        prediction_encoded = MODEL.predict(x_scaled)[0]
        
        species_name = ENCODER.inverse_transform([prediction_encoded])[0]
        
        probabilities = MODEL.predict_proba(x_scaled)[0]
        prob_dict = {
            species: float(prob) 
            for species, prob in zip(ENCODER.classes_, probabilities)
        }
        
        return {
            "prediction": species_name,
            "probabilities": prob_dict
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))