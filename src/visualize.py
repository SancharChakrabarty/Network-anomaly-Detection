"""
Plotting utilities: EDA charts, ROC/PR curves, confusion matrices,
model comparison bars, score distributions, protocol breakdowns.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

from .evaluation import plot_confusion


def plot_label_distribution(raw_df, label_col, results_dir=None):
    """Bar + pie chart of normal vs. attack flow counts."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    vc = raw_df[label_col].value_counts()
    axes[0].bar(
        vc.index,
        vc.values,
        color=["#1D9E75" if "Normal" in str(k) else "#E24B4A" for k in vc.index],
    )
    axes[0].set_title("Flow label distribution", fontweight="bold")
    axes[0].set_ylabel("Count")
    for bar, v in zip(axes[0].patches, vc.values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
            f"{v:,}", ha="center", fontsize=10,
        )

    attack_pct = vc[vc.index.str.lower().str.contains("attack")].sum() / vc.sum() * 100
    axes[1].pie(
        [100 - attack_pct, attack_pct],
        labels=["Normal", "Attack"],
        colors=["#1D9E75", "#E24B4A"],
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops={"linewidth": 1.5, "edgecolor": "white"},
    )
    axes[1].set_title("Attack ratio", fontweight="bold")
    plt.tight_layout()
    if results_dir:
        plt.savefig(f"{results_dir}/01_label_distribution.png", dpi=150)
    plt.show()


def plot_roc_pr_curves(y_test, model_results, results_dir=None):
    """model_results: list of (name, prob, pred) tuples."""
    if y_test.size == 0 or len(np.unique(y_test)) <= 1:
        print("Skipping ROC/PR curves due to insufficient test data.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#378ADD", "#1D9E75", "#E24B4A", "#BA7517", "#7B68EE"]

    for (name, prob, _), color in zip(model_results, colors):
        if prob is not None and prob.size > 0:
            fpr, tpr, _ = roc_curve(y_test, prob)
            auc = roc_auc_score(y_test, prob)
            axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, lw=2)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves", fontweight="bold")
    axes[0].legend(fontsize=9)

    for (name, prob, _), color in zip(model_results, colors):
        if prob is not None and prob.size > 0:
            prec, rec, _ = precision_recall_curve(y_test, prob)
            ap = average_precision_score(y_test, prob)
            axes[1].plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=color, lw=2)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves", fontweight="bold")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    if results_dir:
        plt.savefig(f"{results_dir}/04_roc_pr_curves.png", dpi=150)
    plt.show()


def plot_confusion_matrices(y_test, model_results, results_dir=None):
    """model_results: list of (name, prob, pred) tuples."""
    if y_test.size == 0 or len(np.unique(y_test)) <= 1:
        print("Skipping confusion matrices due to insufficient test data.")
        return

    fig, axes = plt.subplots(1, len(model_results), figsize=(4.5 * len(model_results), 4))
    if len(model_results) == 1:
        axes = [axes]
    for ax, (name, _, pred) in zip(axes, model_results):
        plot_confusion(ax, y_test, pred, name)
    plt.tight_layout()
    if results_dir:
        plt.savefig(f"{results_dir}/05_confusion_matrices.png", dpi=150)
    plt.show()


def plot_model_comparison(results_log, results_dir=None):
    """Bar chart comparing accuracy / f1 / roc_auc / avg_prec across models."""
    if not results_log:
        print("Skipping model comparison as no models were trained or evaluated.")
        return None

    summary = pd.DataFrame(results_log).T.reset_index().rename(columns={"index": "Model"})
    summary = summary.sort_values("f1_attack", ascending=False)
    print("\n=== Final Model Comparison ===")
    print(summary.to_string(index=False))

    fig, ax = plt.subplots(figsize=(10, 4))
    metrics = ["accuracy", "f1_attack", "roc_auc", "avg_prec"]
    x = np.arange(len(summary))
    width = 0.2
    pal = ["#378ADD", "#1D9E75", "#E24B4A", "#BA7517"]
    for i, (metric, color) in enumerate(zip(metrics, pal)):
        ax.bar(x + i * width, summary[metric].astype(float).fillna(0), width, label=metric, color=color, alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(summary["Model"], fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison — key metrics", fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    if results_dir:
        plt.savefig(f"{results_dir}/06_model_comparison.png", dpi=150)
    plt.show()

    return summary


def plot_score_distributions(y_test, model_results, results_dir=None):
    """model_results: list of (name, prob) tuples (max 4)."""
    if y_test.size == 0 or len(np.unique(y_test)) <= 1:
        print("Skipping score distribution plots due to insufficient test data.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    flat_axes = axes.flatten()
    for (title, scores), ax in zip(model_results, flat_axes):
        if scores is not None and scores.size > 0:
            ax.hist(scores[y_test == 0], bins=80, alpha=0.6, color="#1D9E75", label="Normal", density=True)
            ax.hist(scores[y_test == 1], bins=80, alpha=0.6, color="#E24B4A", label="Attack", density=True)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Score")
            ax.set_ylabel("Density")
            ax.legend(fontsize=9)
        else:
            ax.set_title(f"{title} (No data)")
    plt.suptitle("Anomaly score distributions", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    if results_dir:
        plt.savefig(f"{results_dir}/07_score_distributions.png", dpi=150)
    plt.show()


def plot_protocol_breakdown(df, results_dir=None):
    """Attack count and attack rate per protocol."""
    if "protocolName" not in df.columns or df.empty:
        print("Skipping protocol breakdown: 'protocolName' column missing or df empty.")
        return

    proto_attack = (
        df.groupby("protocolName")["is_attack"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "attacks", "count": "total"})
    )
    proto_attack["attack_rate"] = proto_attack["attacks"] / proto_attack["total"]
    proto_attack = proto_attack.sort_values("attacks", ascending=False).head(10)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    proto_attack["attacks"].plot(kind="bar", ax=axes[0], color="#E24B4A", alpha=0.85)
    axes[0].set_title("Attacks by protocol", fontweight="bold")
    axes[0].set_ylabel("Attack count")
    axes[0].tick_params(axis="x", rotation=45)

    proto_attack["attack_rate"].plot(kind="bar", ax=axes[1], color="#378ADD", alpha=0.85)
    axes[1].set_title("Attack rate by protocol", fontweight="bold")
    axes[1].set_ylabel("Fraction of flows")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    if results_dir:
        plt.savefig(f"{results_dir}/08_protocol_analysis.png", dpi=150)
    plt.show()
