"""
Evaluation utilities shared across all models.
"""
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)

# Populated by evaluate_model(); keyed by model name -> metrics dict
results_log = {}


def evaluate_model(name, y_true, y_pred, y_prob=None):
    """Print a classification report and store summary metrics in results_log."""
    if len(y_true) == 0:
        print(f"\nNo data to evaluate for {name}.")
        return None

    report = classification_report(
        y_true, y_pred, target_names=["Normal", "Attack"], output_dict=True
    )
    auc = (
        roc_auc_score(y_true, y_prob)
        if y_prob is not None and len(np.unique(y_true)) > 1
        else None
    )
    ap = (
        average_precision_score(y_true, y_prob)
        if y_prob is not None and len(np.unique(y_true)) > 1
        else None
    )

    print(f"\n{'=' * 55}")
    print(f"  {name}")
    print(f"{'=' * 55}")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Attack"]))
    if auc:
        print(f"  ROC-AUC  : {auc:.4f}")
        print(f"  Avg Prec : {ap:.4f}")

    results_log[name] = {
        "f1_attack": report["Attack"]["f1-score"],
        "precision": report["Attack"]["precision"],
        "recall": report["Attack"]["recall"],
        "accuracy": report["accuracy"],
        "roc_auc": auc,
        "avg_prec": ap,
    }
    return report


def plot_confusion(ax, y_true, y_pred, title):
    """Draw a confusion matrix heatmap onto the given matplotlib axis."""
    if len(y_true) == 0:
        ax.set_title(f"{title} (No data)")
        return
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"],
        ax=ax,
    )
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
