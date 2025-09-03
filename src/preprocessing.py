import os

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_and_preprocess(filepath: str):
    """Load and preprocess iris data, saving encoders for later use"""
    df = pd.read_csv(filepath)
    
    X = df.drop(columns=["species"])
    y = df["species"]
    
    scaler = StandardScaler()
    encoder = LabelEncoder()
    
    X_scaled = scaler.fit_transform(X)
    y_encoded = encoder.fit_transform(y)
    
    
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    encoder_path = os.path.join(models_dir, "encoder.joblib")
    if not os.path.exists(scaler_path):
        joblib.dump(scaler, scaler_path)
    if not os.path.exists(encoder_path):
        joblib.dump(encoder, encoder_path)
    
    print(f"Species classes: {encoder.classes_}")
    print(f"Encoded mapping: {dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))}") # type: ignore
    
    return X_scaled, y_encoded