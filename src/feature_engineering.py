"""
Feature engineering for ISCX network flow records.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from . import config


def detect_label_col(df: pd.DataFrame) -> str:
    """Find the label/tag column in the raw dataframe."""
    if "Tag" in df.columns:
        return "Tag"
    if "tag" in df.columns:
        return "tag"
    raise KeyError(
        f"Label column not found. Available columns: {df.columns.tolist()}"
    )


def engineer_features(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """Add derived numeric features, encode categoricals, and build the binary label."""
    # Drop heavy unused columns (raw payload text) before doing anything else —
    # these are not used as features and can be huge in memory.
    drop_cols = [
        "sourcePayloadAsBase64", "destinationPayloadAsBase64",
        "sourcePayloadAsUTF", "destinationPayloadAsUTF",
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Normalise label (modify in place — avoid a full deep copy on large data)
    df["label"] = df[label_col].str.strip().str.lower()
    df["is_attack"] = (~df["label"].str.contains("normal")).astype(int)

    # Coerce numerics
    for col in config.NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Derived features
    df["bytes_ratio"] = (
        df["totalSourceBytes"] / (df["totalDestinationBytes"].replace(0, 1))
    )
    df["packet_ratio"] = (
        df["totalSourcePackets"] / (df["totalDestinationPackets"].replace(0, 1))
    )
    df["bytes_per_packet_src"] = (
        df["totalSourceBytes"] / (df["totalSourcePackets"].replace(0, 1))
    )
    df["bytes_per_packet_dst"] = (
        df["totalDestinationBytes"] / (df["totalDestinationPackets"].replace(0, 1))
    )
    df["total_bytes"] = df["totalSourceBytes"] + df["totalDestinationBytes"]
    df["total_packets"] = df["totalSourcePackets"] + df["totalDestinationPackets"]

    # Duration (seconds) from timestamps
    for ts_col in ["startDateTime", "stopDateTime"]:
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")

    if "startDateTime" in df.columns and "stopDateTime" in df.columns:
        df["duration_sec"] = (
            (df["stopDateTime"] - df["startDateTime"])
            .dt.total_seconds()
            .fillna(0)
            .clip(lower=0)
        )
        df["bytes_per_sec"] = df["total_bytes"] / (df["duration_sec"] + 1)

    # Encode categoricals
    le = LabelEncoder()
    for col in config.CATEGORICAL_COLS:
        if col in df.columns:
            df[col + "_enc"] = le.fit_transform(df[col].fillna("unknown").astype(str))

    # Well-known port flag
    if "destinationPort" in df.columns:
        df["dest_port_num"] = pd.to_numeric(df["destinationPort"], errors="coerce").fillna(0)
        df["is_well_known_port"] = (df["dest_port_num"] < 1024).astype(int)

    return df


def select_features(df: pd.DataFrame) -> list:
    """Return the subset of the feature template that's actually present in df."""
    return [c for c in config.FEATURE_COLS_TEMPLATE if c in df.columns]


def build_feature_matrix(df: pd.DataFrame, feature_cols: list):
    """Convert selected feature columns + label into numpy arrays, cleaning inf/nan."""
    X = df[feature_cols].values.astype(np.float32)
    y = df["is_attack"].values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    return X, y
