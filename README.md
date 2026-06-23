# Network Anomaly Detection

Detecting malicious network flows in the **ISCX IDS 2012** dataset using five
complementary machine learning approaches — two supervised, two unsupervised,
and one deep learning sequence model.

## Models

| Model | Type | Notes |
|---|---|---|
| Random Forest | Supervised | Balanced class weights, feature importance plot |
| XGBoost | Supervised | `scale_pos_weight` for imbalance, GPU-aware |
| Isolation Forest | Unsupervised | Trained only on normal traffic |
| Deep Autoencoder | Unsupervised | Reconstruction-error threshold (95th percentile) |
| 1D-CNN | Supervised (deep learning) | Treats features as a 1D sequence |

Best result: **Random Forest — 99.43% accuracy, ROC-AUC 0.9993** (on the
held-out test split; see `results/06_model_comparison.png` after a run).

## Project structure

```
network-anomaly-detection/
├── src/
│   ├── config.py              # paths, random state, feature list
│   ├── data_loader.py         # ISCX XML parsing
│   ├── feature_engineering.py # derived features, encoding, label
│   ├── evaluation.py          # classification report + confusion matrix helpers
│   ├── visualize.py           # EDA / ROC / PR / comparison plots
│   ├── inference.py           # single-flow real-time prediction
│   ├── main.py                # orchestrates the full pipeline
│   └── models/
│       ├── random_forest_model.py
│       ├── xgboost_model.py
│       ├── isolation_forest_model.py
│       ├── autoencoder_model.py
│       └── cnn_model.py
├── notebooks/
│   └── run_colab.ipynb        # thin Colab wrapper around src/
├── results/                   # generated plots + saved models (gitignored)
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/<your-username>/network-anomaly-detection.git
cd network-anomaly-detection
pip install -r requirements.txt
```

Download the [ISCX IDS 2012](https://www.unb.ca/cic/datasets/ids.html) XML
files and point `DATASET_PATH` at the folder containing them:

```bash
export DATASET_PATH=/path/to/iscxids2012-master
export RESULTS_DIR=./results
```

## Run

```bash
python -m src.main
```

This loads and parses the XML flows, engineers features, trains all five
models, generates evaluation plots into `RESULTS_DIR`, and saves trained
models (`.pkl` / `.keras`) plus the fitted scaler.

### In Google Colab

Open `notebooks/run_colab.ipynb` — it mounts Drive, clones this repo, installs
dependencies, and calls `src.main.run()`.

## Real-time inference

```python
from src.inference import predict_flow

result = predict_flow(
    flow_dict=my_flow_features,
    feature_cols=feature_cols,
    scaler=scaler,
    models={"xgboost": xgb_model, "rf": rf_model, "autoencoder": ae_model},
    model="xgboost",
)
# {'prediction': 'ATTACK', 'confidence': 0.97, 'model': 'xgboost'}
```

## Dataset

[ISCX IDS 2012](https://www.unb.ca/cic/datasets/ids.html), Canadian Institute
for Cybersecurity. Flows are labelled `Normal` or `Attack`; this project frames
detection as a binary classification problem on flow-level statistics
(byte/packet counts, ratios, protocol, duration).
