import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(result):
    labels = [0,1,2]
    codes = result.get("class_codes", ["MEL","NV","BKL"])
    y_true = result["y_test"]
    y_pred = result["pred"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "confusion_matrix": cm,
        "class_names": codes,
        "report": classification_report(y_true, y_pred, labels=labels, target_names=codes, zero_division=0)
    }
