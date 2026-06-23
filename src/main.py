"""
Network Anomaly Detection — main pipeline.

Run with:
    python -m src.main

Or, inside a Colab cell:
    from src import main
    main.run()
"""
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from . import config
from .data_loader import load_all_xml
from .feature_engineering import (
    detect_label_col,
    engineer_features,
    select_features,
    build_feature_matrix,
)
from .evaluation import results_log
from .models.random_forest_model import train_random_forest
from .models.xgboost_model import train_xgboost
from .models.isolation_forest_model import train_isolation_forest
from .models.autoencoder_model import train_autoencoder
from .models.cnn_model import train_1d_cnn
from . import visualize


def run():
    # 1. Load -------------------------------------------------------------
    raw_df = load_all_xml(config.DATASET_PATH)
    label_col = detect_label_col(raw_df)
    print(f"\nLabel column: '{label_col}'")
    print(raw_df[label_col].value_counts())
    visualize.plot_label_distribution(raw_df, label_col, config.RESULTS_DIR)

    # 2. Feature engineering ----------------------------------------------
    df = engineer_features(raw_df, label_col)
    feature_cols = select_features(df)
    print(f"Features used ({len(feature_cols)}): {feature_cols}")

    X, y = build_feature_matrix(df, feature_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=config.VAL_SIZE / (1 - config.TEST_SIZE),
        random_state=config.RANDOM_STATE, stratify=y_train,
    )
    print(f"Train : {X_train.shape}  |  Val : {X_val.shape}  |  Test : {X_test.shape}")

    # 3. Scaling ------------------------------------------------------------
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # 4. Train models ---------------------------------------------------------
    rf, rf_pred, rf_prob = train_random_forest(
        X_train_s, y_train, X_test_s, y_test, feature_cols, config.RESULTS_DIR
    )
    xgb_model, xgb_pred, xgb_prob = train_xgboost(
        X_train_s, y_train, X_val_s, y_val, X_test_s, y_test
    )
    iso, iso_pred, iso_score = train_isolation_forest(
        X_train_s, y_train, X_test_s, y_test
    )
    ae, ae_pred, ae_err = train_autoencoder(X_train_s, y_train, X_test_s, y_test)
    ae_threshold = float(np.percentile(ae_err, 95)) if ae_err is not None else 0.5
    cnn_model, cnn_pred, cnn_prob = train_1d_cnn(
        X_train_s, y_train, X_val_s, y_val, X_test_s, y_test
    )

    # 5. Visual comparisons ---------------------------------------------------
    model_results = [
        ("Random Forest", rf_prob, rf_pred),
        ("XGBoost", xgb_prob, xgb_pred),
        ("Isolation Forest", iso_score, iso_pred),
        ("Autoencoder", ae_err, ae_pred),
        ("CNN", cnn_prob, cnn_pred),
    ]
    visualize.plot_roc_pr_curves(y_test, model_results, config.RESULTS_DIR)
    visualize.plot_confusion_matrices(y_test, model_results[:4], config.RESULTS_DIR)
    summary = visualize.plot_model_comparison(results_log, config.RESULTS_DIR)
    visualize.plot_score_distributions(
        y_test,
        [
            ("Random Forest score", rf_prob),
            ("XGBoost score", xgb_prob),
            ("Isolation Forest score", iso_score),
            ("Autoencoder recon error", ae_err),
        ],
        config.RESULTS_DIR,
    )
    visualize.plot_protocol_breakdown(df, config.RESULTS_DIR)

    # 6. Save artifacts ------------------------------------------------------
    if rf is not None:
        joblib.dump(rf, f"{config.RESULTS_DIR}/random_forest.pkl")
    if xgb_model is not None:
        joblib.dump(xgb_model, f"{config.RESULTS_DIR}/xgboost.pkl")
    if iso is not None:
        joblib.dump(iso, f"{config.RESULTS_DIR}/isolation_forest.pkl")
    if ae is not None:
        ae.save(f"{config.RESULTS_DIR}/autoencoder.keras")
    if cnn_model is not None:
        cnn_model.save(f"{config.RESULTS_DIR}/cnn_model.keras")
    joblib.dump(scaler, f"{config.RESULTS_DIR}/scaler.pkl")

    # Save a handful of REAL test rows (pre-scaling, raw feature values) plus
    # their true labels and the autoencoder's fitted threshold, so future
    # inference testing uses genuine flows instead of guessed numbers.
    sample_idx = np.random.RandomState(config.RANDOM_STATE).choice(
        len(X_test), size=min(20, len(X_test)), replace=False
    )
    joblib.dump(
        {
            "feature_cols": feature_cols,
            "X_sample_raw": X_test[sample_idx],
            "y_sample_true": y_test[sample_idx],
            "autoencoder_threshold": ae_threshold,
        },
        f"{config.RESULTS_DIR}/inference_samples.pkl",
    )

    print(f"\nAll artifacts saved to {config.RESULTS_DIR}/")
    if os.path.exists(config.RESULTS_DIR):
        for f in sorted(os.listdir(config.RESULTS_DIR)):
            size = os.path.getsize(os.path.join(config.RESULTS_DIR, f))
            print(f"  {f:40s}  {size / 1024:>8.1f} KB")

    # 7. Final summary --------------------------------------------------------
    print("\n" + "=" * 55)
    print("  NETWORK ANOMALY DETECTION — COMPLETE")
    print("=" * 55)
    if summary is not None and not summary.empty:
        best = summary.iloc[0]
        print(f"  Dataset   : {len(df):,} flows | {int(df['is_attack'].sum()):,} attacks")
        print(f"  Features  : {len(feature_cols)}")
        print(f"  Best model: {best['Model']}")
        print(f"    F1 (attack) : {float(best['f1_attack']):.4f}")
        print(f"    ROC-AUC     : {float(best['roc_auc']):.4f}")
        print(f"    Accuracy    : {float(best['accuracy']):.4f}")
    print(f"\n  Plots saved to : {config.RESULTS_DIR}/")
    print("=" * 55)

    return {
        "df": df,
        "feature_cols": feature_cols,
        "scaler": scaler,
        "models": {"rf": rf, "xgboost": xgb_model, "autoencoder": ae, "cnn": cnn_model, "isolation_forest": iso},
        "test_data": (X_test_s, y_test),
        "summary": summary,
    }


if __name__ == "__main__":
    run()
