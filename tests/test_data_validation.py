import pandas as pd

from src.preprocessing import load_and_preprocess


def test_data_schema():
    df = pd.read_csv("iris.csv")
    expected_cols = {"sepal_length", "sepal_width", "petal_length", "petal_width", "species"}
    assert set(df.columns) == expected_cols, "Schema mismatch!"

def test_no_missing_values():
    X, _= load_and_preprocess("iris.csv")
    X_df = pd.DataFrame(X)
    assert not X_df.isnull().any().any(), "Feature set contains missing values!"
