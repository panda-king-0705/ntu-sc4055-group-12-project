"""Shared helpers so every model uses the same data split and the same metrics.

    from common import load_data, train_val_split, evaluate
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"


def load_data(quantum=True):
    """Returns X_train, X_test, y_train, y_test.

    quantum=True  -> PCA features scaled to [0, pi] (VQC, QSVC)
    quantum=False -> raw PCA features (logistic regression, SVM)
    Both files have the same rows in the same order.
    """
    name = "cmapss_qml_4f_500.npz" if quantum else "cmapss_ml_4f_500.npz"
    d = np.load(DATA_DIR / name)
    return d["X_train"], d["X_test"], d["y_train"], d["y_test"]


def load_units():
    """Engine id of every row: unit_train, unit_test (same for both data files)."""
    d = np.load(DATA_DIR / "cmapss_qml_4f_500.npz")
    return d["unit_train"], d["unit_test"]


def train_val_split(X, y, val_size=0.2, seed=42):
    """Splits the training set into train/validation by engine, so no engine is in both.

    X, y must be the full training set from load_data(), in the same row order.
    Fixed seed -> same rows for everyone, and the same for the qml and ml files.
    """
    units = load_units()[0]
    assert len(units) == len(X), "pass the full X_train from load_data()"
    splitter = StratifiedGroupKFold(n_splits=round(1 / val_size), shuffle=True, random_state=seed)
    train_idx, val_idx = next(splitter.split(X, y, groups=units))
    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]


def evaluate(y_true, y_pred):
    """Label 1 = engine fails within 30 cycles, so a false negative is a missed failure."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_negative_rate": fn / (fn + tp) if fn + tp else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
