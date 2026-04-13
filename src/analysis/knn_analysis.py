from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


def save_knn_tuning_curves(output_dir: Path, feature_modes: Iterable[str], file_name: str = "knn_tuning_curves.png") -> Path:
    """Save validation macro-F1 and accuracy curves for each feature mode."""
    feature_modes = list(feature_modes)
    if not feature_modes:
        raise ValueError("feature_modes is empty")

    fig, axes = plt.subplots(1, len(feature_modes), figsize=(5 * len(feature_modes), 4), squeeze=False)

    for i, mode in enumerate(feature_modes):
        tune_path = output_dir / mode / "validation_tuning.csv"
        tune_df = pd.read_csv(tune_path).sort_values("k")

        axes[0, i].plot(tune_df["k"], tune_df["val_f1_macro"], marker="o", label="Val Macro-F1")
        axes[0, i].plot(tune_df["k"], tune_df["val_accuracy"], marker="s", label="Val Accuracy", alpha=0.8)
        axes[0, i].set_title(f"KNN Tuning ({mode})")
        axes[0, i].set_xlabel("k")
        axes[0, i].set_ylabel("score")
        axes[0, i].set_xticks(sorted(tune_df["k"].unique()))
        axes[0, i].legend()

    plt.tight_layout()
    path = output_dir / file_name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_knn_confusion_matrices(
    output_dir: Path,
    summary_df: pd.DataFrame,
    class_a: int,
    class_b: int,
    file_name: str = "knn_confusion_matrices.png",
) -> Path:
    """Save confusion-matrix heatmaps from a KNN summary table."""
    if summary_df.empty:
        raise ValueError("summary_df is empty")

    fig, axes = plt.subplots(1, len(summary_df), figsize=(5 * len(summary_df), 4), squeeze=False)

    for i, (_, row) in enumerate(summary_df.iterrows()):
        cm = np.asarray(row["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[0, i],
            xticklabels=[class_a, class_b],
            yticklabels=[class_a, class_b],
        )
        axes[0, i].set_title(f"{row['feature_mode']} | F1={row['test_f1_macro']:.4f}")
        axes[0, i].set_xlabel("Predicted")
        axes[0, i].set_ylabel("True")

    plt.tight_layout()
    path = output_dir / file_name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_knn_feature_comparison(output_dir: Path, summary_df: pd.DataFrame, file_name: str = "knn_feature_comparison.png") -> Path:
    """Save a bar chart comparing test macro-F1 and accuracy across feature modes."""
    if summary_df.empty:
        raise ValueError("summary_df is empty")

    ordered = summary_df.sort_values(by=["test_f1_macro", "test_accuracy"], ascending=[False, False]).copy()
    plot_df = ordered[["feature_mode", "test_f1_macro", "test_accuracy"]].set_index("feature_mode")

    fig, ax = plt.subplots(figsize=(8, 4))
    plot_df.plot(kind="bar", ax=ax)
    ax.set_title("KNN Feature Comparison")
    ax.set_xlabel("Feature mode")
    ax.set_ylabel("Score")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="lower right")
    plt.tight_layout()

    path = output_dir / file_name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_knn_tuning_curves(output_dir: Path, feature_modes: Iterable[str]) -> None:
    """Plot validation macro-F1 and accuracy curves for each feature mode."""
    feature_modes = list(feature_modes)
    if not feature_modes:
        return

    fig, axes = plt.subplots(1, len(feature_modes), figsize=(5 * len(feature_modes), 4), squeeze=False)

    for i, mode in enumerate(feature_modes):
        tune_path = output_dir / mode / "validation_tuning.csv"
        tune_df = pd.read_csv(tune_path).sort_values("k")

        axes[0, i].plot(tune_df["k"], tune_df["val_f1_macro"], marker="o", label="Val Macro-F1")
        axes[0, i].plot(tune_df["k"], tune_df["val_accuracy"], marker="s", label="Val Accuracy", alpha=0.8)
        axes[0, i].set_title(f"KNN Tuning ({mode})")
        axes[0, i].set_xlabel("k")
        axes[0, i].set_ylabel("score")
        axes[0, i].set_xticks(sorted(tune_df["k"].unique()))
        axes[0, i].legend()

    plt.tight_layout()
    plt.show()


def plot_knn_confusion_matrices(summary_df: pd.DataFrame, class_a: int, class_b: int) -> None:
    """Plot confusion matrices from a KNN summary table."""
    if summary_df.empty:
        return

    fig, axes = plt.subplots(1, len(summary_df), figsize=(5 * len(summary_df), 4), squeeze=False)

    for i, (_, row) in enumerate(summary_df.iterrows()):
        cm = np.asarray(row["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[0, i],
            xticklabels=[class_a, class_b],
            yticklabels=[class_a, class_b],
        )
        axes[0, i].set_title(f"{row['feature_mode']} | F1={row['test_f1_macro']:.4f}")
        axes[0, i].set_xlabel("Predicted")
        axes[0, i].set_ylabel("True")

    plt.tight_layout()
    plt.show()
