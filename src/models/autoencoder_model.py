"""
Deep Autoencoder — unsupervised anomaly detector via reconstruction error.
Trained only on normal traffic; flows with high reconstruction error are
flagged as anomalies using a dynamic (95th percentile) threshold.
"""
import numpy as np
from tensorflow.keras import layers, models

from ..evaluation import evaluate_model


def build_autoencoder(input_dim):
    """Build a bottleneck autoencoder: input -> 12 -> 6 -> 12 -> input."""
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(12, activation="relu")(input_layer)
    encoded = layers.Dense(6, activation="relu")(encoded)  # bottleneck
    decoded = layers.Dense(12, activation="relu")(encoded)
    output_layer = layers.Dense(input_dim, activation="sigmoid")(decoded)

    autoencoder = models.Model(input_layer, output_layer)
    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder


def train_autoencoder(X_train, y_train, X_test, y_test, percentile=95):
    """Train an autoencoder on normal traffic and threshold reconstruction error.

    Returns (model, pred, reconstruction_error).
    """
    print("Training Deep Autoencoder...")

    X_train_normal = X_train[y_train == 0]

    ae = build_autoencoder(X_train.shape[1])
    ae.fit(
        X_train_normal,
        X_train_normal,
        epochs=50,
        batch_size=128,
        shuffle=True,
        validation_split=0.1,
        verbose=0,
    )

    reconstructions = ae.predict(X_test)
    mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

    # Dynamic threshold: flag the top (100 - percentile)% highest-error flows
    threshold = np.percentile(mse, percentile)
    pred = (mse > threshold).astype(int)

    evaluate_model("Autoencoder", y_test, pred, mse)

    return ae, pred, mse
