"""
1D Convolutional Neural Network — treats engineered features as a 1D
"sequence" to learn local patterns between adjacent feature values.
"""
import numpy as np
from tensorflow.keras import layers, models, callbacks

from ..evaluation import evaluate_model


def build_1d_cnn(input_shape):
    """Build a small 1D-CNN classifier."""
    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv1D(32, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.MaxPooling1D(pool_size=2),
            layers.Conv1D(64, kernel_size=3, activation="relu", padding="same"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dense(64, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_1d_cnn(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train and evaluate a 1D-CNN classifier. Returns (model, pred, prob)."""
    print("Training 1D-CNN …")

    if X_train.size == 0 or len(set(y_train)) <= 1:
        print("Skipping CNN training due to insufficient data.")
        return None, None, None

    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val_cnn = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model = build_1d_cnn((X_train_cnn.shape[1], 1))

    counts = np.bincount(y_train)
    weight_for_0 = (1 / counts[0]) * (len(y_train) / 2.0)
    weight_for_1 = (1 / counts[1]) * (len(y_train) / 2.0)
    class_weights = {0: weight_for_0, 1: weight_for_1}

    model.fit(
        X_train_cnn,
        y_train,
        epochs=30,
        batch_size=256,
        validation_data=(X_val_cnn, y_val),
        class_weight=class_weights,
        callbacks=[
            callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=3, factor=0.5),
        ],
        verbose=1,
    )

    prob = model.predict(X_test_cnn).flatten()
    pred = (prob > 0.5).astype(int)
    evaluate_model("CNN", y_test, pred, prob)

    return model, pred, prob
