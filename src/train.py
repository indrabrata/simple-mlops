import os

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from mlflow import sklearn as mlflow_sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from preprocessing import load_and_preprocess

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")

os.makedirs(MODEL_DIR, exist_ok=True)

X, y = load_and_preprocess("iris.csv")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)

with mlflow.start_run() as run:
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    try:
        roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
    except ValueError:
        roc_auc = np.nan

    mlflow.log_metric("accuracy", float(acc))
    mlflow.log_metric("precision_weighted", float(precision))
    mlflow.log_metric("recall_weighted", float(recall))
    mlflow.log_metric("f1_weighted", float(f1))
    mlflow.log_metric("roc_auc_weighted", float(roc_auc))

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=list(clf.classes_),
        yticklabels=list(clf.classes_),
        cmap="Blues"
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()

    mlflow_sklearn.log_model(clf, artifact_path="model", registered_model_name="iris-ml")
    joblib.dump(clf, MODEL_PATH)

    print(f"Run ID: {run.info.run_id}, Acc: {acc:.3f}, F1: {f1:.3f}")
