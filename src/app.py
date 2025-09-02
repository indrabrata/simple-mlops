import os

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException

from src.request import PredictRequest

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")

app = FastAPI(title="Iris Classifier", version="0.1")

@app.on_event("startup")
def load_model():
    global MODEL
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    MODEL = joblib.load(MODEL_PATH)


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

        preds = MODEL.predict(x)
        return {"prediction": str(preds[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
