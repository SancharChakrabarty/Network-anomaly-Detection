"""
XGBoost classifier — gradient-boosted trees with class imbalance handling.
"""
import xgboost as xgb
import tensorflow as tf

from .. import config
from ..evaluation import evaluate_model


def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate an XGBoost classifier. Returns (model, pred, prob)."""
    print("Training XGBoost …")

    if X_train.size == 0 or len(set(y_train)) <= 1:
        print("Skipping XGBoost training due to insufficient data or single class label.")
        return None, None, None

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    has_gpu = len(tf.config.list_physical_devices("GPU")) > 0

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=7,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
        device="cuda" if has_gpu else "cpu",
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    evaluate_model("XGBoost", y_test, pred, prob)

    return model, pred, prob
