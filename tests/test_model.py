import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from src.preprocessing import load_and_preprocess


def test_model_accuracy():
    df = pd.read_csv("iris.csv")
    X, y = load_and_preprocess("iris.csv")
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = joblib.load("models/model.joblib")
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    assert acc > 0.7, f"Accuracy too low: {acc}"
