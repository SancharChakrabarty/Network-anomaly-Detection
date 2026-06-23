"""
Real-time single-flow inference helper. Wraps a trained model + scaler so
new flows can be classified without re-running the whole pipeline.
"""
import numpy as np


def predict_flow(flow_dict: dict, feature_cols, scaler, models: dict, model="xgboost", threshold=None) -> dict:
    """
    Classify a single network flow.

    Parameters
    ----------
    flow_dict    : dict  — feature values keyed by feature_cols names
    feature_cols : list  — ordered list of feature names the models expect
    scaler       : fitted sklearn scaler (e.g. RobustScaler)
    models       : dict  — {"rf": rf_model, "xgboost": xgb_model, "autoencoder": ae_model, "cnn": cnn_model}
    model        : str   — which model to use
    threshold    : float — decision threshold. Defaults to 0.5 for classifiers.
                   For autoencoder, pass the fitted reconstruction-error
                   threshold (e.g. saved as 'autoencoder_threshold') — 0.5
                   is meaningless for reconstruction-error scores.

    Returns
    -------
    dict with 'prediction', 'confidence', 'model' (or 'error')
    """
    if threshold is None:
        threshold = 0.5
    if scaler is None:
        return {"error": "Scaler not initialized. Cannot predict without a trained scaler.", "model": model}

    chosen = models.get(model)
    if chosen is None:
        return {"error": f"Model '{model}' not trained. Cannot predict.", "model": model}

    if not feature_cols:
        return {"error": "feature_cols not defined. Cannot predict.", "model": model}

    x = np.array([[flow_dict.get(c, 0) for c in feature_cols]], dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    x_s = scaler.transform(x)

    if model == "rf":
        prob = chosen.predict_proba(x_s)[0, 1]
    elif model == "xgboost":
        prob = chosen.predict_proba(x_s)[0, 1]
    elif model == "autoencoder":
        recon = chosen.predict(x_s, verbose=0)
        prob = float(np.mean(np.square(x_s - recon)))
    elif model == "cnn":
        x_cnn = x_s.reshape(x_s.shape[0], x_s.shape[1], 1)
        prob = float(chosen.predict(x_cnn, verbose=0).flatten()[0])
    else:
        raise ValueError(f"Unknown model: {model}")

    label = "ATTACK" if prob > threshold else "Normal"
    return {"prediction": label, "confidence": round(float(prob), 4), "model": model}
