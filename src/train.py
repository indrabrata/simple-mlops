import logging
import os

import boto3
import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from mlflow import sklearn as mlflow_sklearn
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from .preprocessing import load_and_preprocess

MODEL_DIR = os.getenv("MODEL_DIR", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

data_path = os.getenv("DATA_PATH", "data/iris.csv")
X, y = load_and_preprocess(data_path)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

mlflow.set_tracking_uri(mlflow_uri)

try:
    mlflow.set_experiment("iris-classification")
except Exception as e:
    print(f"Could not set experiment: {e}")

print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

with mlflow.start_run() as run:
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    
    encoder = joblib.load(os.path.join(MODEL_DIR, "encoder.joblib"))
    class_names = encoder.classes_
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
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
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=class_names,
        yticklabels=class_names,
        cmap="Blues"
    )
    plt.title("Confusion Matrix")
    plt.ylabel("True Species")
    plt.xlabel("Predicted Species")
    plt.tight_layout()
    
    plot_path = "confusion_matrix.png"
    plt.savefig("confusion_matrix.png")
    try:
        mlflow.log_artifact(plot_path)
    except Exception as e:
        logging.warning(f"Could not log plot artifact: {e}")
    plt.close()
    
    signature = infer_signature(X_train, clf.predict(X_train))
    try:
        mlflow_sklearn.log_model(
            clf, 
            name="model",
            signature=signature,
            registered_model_name="iris-ml",
        )
        logging.info("MLflow model logged successfully")
    except Exception as e:
        logging.error(f"MLflow model logging failed: {e}")
        
    
    joblib.dump(clf, os.path.join(MODEL_DIR, "model.joblib"))
    
    mlflow.log_artifact(os.path.join(MODEL_DIR, "scaler.joblib"))
    mlflow.log_artifact(os.path.join(MODEL_DIR, "encoder.joblib"))
    
    logging.info(f"Run ID: {run.info.run_id}, Acc: {acc:.3f}, F1: {f1:.3f}")
    logging.info(f"Model saved to: {os.path.join(MODEL_DIR, "model.joblib")}")
    logging.info(f"Species classes: {class_names}")