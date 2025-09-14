import os

import pandas as pd

from src import load_and_preprocess


def test_data_schema():
    data_path = os.getenv("DATA_PATH", "iris.csv")
    df = pd.read_csv(data_path)
    expected_cols = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    assert set(df.columns) == expected_cols, "Schema mismatch!"

def test_no_missing_values():
    data_path = os.getenv("DATA_PATH", "iris.csv")
    X, _= load_and_preprocess(data_path)
    X_df = pd.DataFrame(X)
    assert not X_df.isnull().any().any(), "Feature set contains missing values!"
