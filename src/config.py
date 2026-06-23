"""
Central configuration for the Network Anomaly Detection project.
Edit DATASET_PATH and RESULTS_DIR to match your environment.
"""
import os
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: saves plots without opening windows

# --- Paths -------------------------------------------------------------
# When running in Google Colab, mount Drive first:
#   from google.colab import drive
#   drive.mount('/content/drive')
DATASET_PATH = os.environ.get(
    "DATASET_PATH", "/content/drive/MyDrive/iscxids2012-master"
)
RESULTS_DIR = os.environ.get(
    "RESULTS_DIR", "/content/drive/MyDrive/Results"
)

os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Reproducibility / splits -------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# --- Feature engineering -------------------------------------------------
NUMERIC_COLS = [
    "totalSourceBytes", "totalDestinationBytes",
    "totalDestinationPackets", "totalSourcePackets",
    "sourcePort", "destinationPort",
]

CATEGORICAL_COLS = ["protocolName", "direction"]

FEATURE_COLS_TEMPLATE = [
    "totalSourceBytes", "totalDestinationBytes",
    "totalSourcePackets", "totalDestinationPackets",
    "bytes_ratio", "packet_ratio",
    "bytes_per_packet_src", "bytes_per_packet_dst",
    "total_bytes", "total_packets",
    "bytes_per_sec", "duration_sec",
    "protocolName_enc", "direction_enc",
    "is_well_known_port",
]
