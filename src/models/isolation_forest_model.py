"""
Isolation Forest — unsupervised anomaly detector trained only on normal traffic.
"""
import numpy as np
from sklearn.ensemble import IsolationForest

from .. import config
from ..evaluation import evaluate_model


def train_isolation_forest(X_train, y_train, X_test, y_test):
    """Train and evaluate an Isolation Forest. Returns (model, pred, anomaly_score)."""
    print("Training Isolation Forest (unsupervised) …")

    if X_train.size == 0 or len(set(y_train)) <= 1:
        print("Skipping Isolation Forest training due to insufficient data or single class label.")
        return None, None, None

    normal_mask = y_train == 0
    if normal_mask.sum() == 0:
        print("No normal samples found for Isolation Forest training. Skipping.")
        return None, None, None

    iso = IsolationForest(
        n_estimators=200,
        contamination=y_train.mean(),
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    iso.fit(X_train[normal_mask])

    iso_raw = iso.predict(X_test)            # +1 = inlier, -1 = outlier
    pred = (iso_raw == -1).astype(int)
    score = -iso.score_samples(X_test)       # higher = more anomalous
    evaluate_model("Isolation Forest", y_test, pred, score)

    return iso, pred, score
