"""
Load already-trained models from results/ and run predictions on REAL
flows sampled from your test set during training (saved in
inference_samples.pkl) — no retraining, no XML re-parsing, no guessed
feature values.

Run with:
    python test_inference.py

Note: this requires re-running `python -m src.main` at least once AFTER
the inference_samples.pkl save logic was added to main.py, so the sample
file exists in RESULTS_DIR.
"""
import os
import joblib
import numpy as np
from tensorflow.keras.models import load_model

from src import config
from src.inference import predict_flow

RESULTS_DIR = config.RESULTS_DIR


def load_saved_models():
    """Load whichever trained artifacts exist in RESULTS_DIR."""
    models = {}

    rf_path = os.path.join(RESULTS_DIR, "random_forest.pkl")
    if os.path.exists(rf_path):
        models["rf"] = joblib.load(rf_path)
        print(f"Loaded Random Forest from {rf_path}")

    xgb_path = os.path.join(RESULTS_DIR, "xgboost.pkl")
    if os.path.exists(xgb_path):
        models["xgboost"] = joblib.load(xgb_path)
        print(f"Loaded XGBoost from {xgb_path}")

    ae_path = os.path.join(RESULTS_DIR, "autoencoder.keras")
    if os.path.exists(ae_path):
        models["autoencoder"] = load_model(ae_path)
        print(f"Loaded Autoencoder from {ae_path}")

    cnn_path = os.path.join(RESULTS_DIR, "cnn_model.keras")
    if os.path.exists(cnn_path):
        models["cnn"] = load_model(cnn_path)
        print(f"Loaded CNN from {cnn_path}")

    scaler_path = os.path.join(RESULTS_DIR, "scaler.pkl")
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    if scaler:
        print(f"Loaded scaler from {scaler_path}")

    return models, scaler


if __name__ == "__main__":
    models, scaler = load_saved_models()

    if scaler is None:
        raise SystemExit("scaler.pkl not found — run the training pipeline first.")

    samples_path = os.path.join(RESULTS_DIR, "inference_samples.pkl")
    if not os.path.exists(samples_path):
        raise SystemExit(
            "inference_samples.pkl not found.\n"
            "This file is created by the updated src/main.py. "
            "Re-run `python -m src.main` once to generate it, then rerun this script."
        )

    saved = joblib.load(samples_path)
    feature_cols = saved["feature_cols"]
    X_sample_raw = saved["X_sample_raw"]       # real, UNSCALED feature rows
    y_sample_true = saved["y_sample_true"]     # ground-truth labels (0=Normal, 1=Attack)
    ae_threshold = saved["autoencoder_threshold"]

    print(f"\nTesting on {len(X_sample_raw)} REAL flows sampled from your test set.\n")

    correct = {name: 0 for name in models}
    total = len(X_sample_raw)

    for i in range(total):
        flow_dict = {col: float(X_sample_raw[i, j]) for j, col in enumerate(feature_cols)}
        true_label = "ATTACK" if y_sample_true[i] == 1 else "Normal"

        print(f"--- Flow {i+1}/{total}  (true label: {true_label}) ---")
        for model_name in ["xgboost", "rf", "cnn", "autoencoder"]:
            if model_name not in models:
                continue
            thresh = ae_threshold if model_name == "autoencoder" else 0.5
            result = predict_flow(flow_dict, feature_cols, scaler, models, model=model_name, threshold=thresh)
            print(f"  [{model_name}] {result}")
            if result.get("prediction") == true_label:
                correct[model_name] += 1
        print()

    print("=== Accuracy on this sample ===")
    for name, c in correct.items():
        print(f"  {name}: {c}/{total} correct")
