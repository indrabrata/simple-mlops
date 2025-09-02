import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

scaler = StandardScaler()
encoder = LabelEncoder()

def load_and_preprocess(filepath: str):
    df = pd.read_csv(filepath)

    X = df.drop(columns=["species"])
    y = df["species"]

    X_scaled = scaler.fit_transform(X)
    y_encoded = encoder.fit_transform(y)

    return X_scaled, y_encoded
