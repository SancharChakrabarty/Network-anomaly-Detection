"""
Random Forest classifier — supervised baseline.
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from .. import config
from ..evaluation import evaluate_model


def train_random_forest(X_train, y_train, X_test, y_test, feature_cols, results_dir=None):
    """Train and evaluate a Random Forest classifier. Returns (model, pred, prob)."""
    print("Training Random Forest …")

    if X_train.size == 0 or len(set(y_train)) <= 1:
        print("Skipping Random Forest training due to insufficient data or single class label.")
        return None, None, None

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
    )
    rf.fit(X_train, y_train)

    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    evaluate_model("Random Forest", y_test, rf_pred, rf_prob)

    # Feature importance plot
    feat_imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    plt.figure(figsize=(9, 4))
    feat_imp.head(12).plot(kind="bar", color="#378ADD")
    plt.title("Random Forest — top feature importances", fontweight="bold")
    plt.tight_layout()
    if results_dir:
        plt.savefig(f"{results_dir}/02_rf_feature_importance.png", dpi=150)
    plt.show()

    return rf, rf_pred, rf_prob
