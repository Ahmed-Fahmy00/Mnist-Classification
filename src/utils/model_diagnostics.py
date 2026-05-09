"""Reusable model diagnostics for multiclass classification.

This module is intentionally model-agnostic. It works with any estimator that
implements ``fit(X, y)`` and ``predict(X)``. It is suitable for custom models
such as the project KNN, logistic regression, and Naive Bayes implementations.
"""

from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np

from src.utils.evaluation import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

ModelFactory = Callable[[], Any]

def _as_array(x: Any) -> np.ndarray:
    return np.asarray(x)

def _subset_training_data(
    x: np.ndarray,
    y: np.ndarray,
    train_size: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw a stratified subset using only NumPy."""
    if train_size >= len(x):
        return x, y

    rng = np.random.default_rng(random_state)
    labels, counts = np.unique(y, return_counts=True)

    if len(labels) == 1:
        indices = rng.choice(len(x), size=train_size, replace=False)
        return x[indices], y[indices]

    class_indices = {label: np.where(y == label)[0] for label in labels}
    quotas = {}

    for label, count in zip(labels, counts):
        quota = int(round(train_size * count / len(y)))
        quota = max(1, quota)
        quotas[label] = min(quota, len(class_indices[label]))

    total = sum(quotas.values())

    while total < train_size:
        room = [label for label in labels if quotas[label] < len(class_indices[label])]
        if not room:
            break
        label = max(room, key=lambda lab: len(class_indices[lab]) - quotas[lab])
        quotas[label] += 1
        total += 1

    while total > train_size:
        reducible = [label for label in labels if quotas[label] > 1]
        if not reducible:
            break
        label = max(reducible, key=lambda lab: quotas[lab])
        quotas[label] -= 1
        total -= 1

    chosen = []
    for label in labels:
        selected = rng.choice(class_indices[label], size=quotas[label], replace=False)
        chosen.extend(selected.tolist())

    chosen = np.asarray(chosen)
    rng.shuffle(chosen)
    return x[chosen], y[chosen]

def evaluate_model(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    average: str = "macro",
    predict_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Fit a model once and evaluate it on train and validation data."""
    fit_kwargs = fit_kwargs or {}
    predict_kwargs = predict_kwargs or {}

    model.fit(x_train, y_train, **fit_kwargs)

    y_train_pred = model.predict(x_train, **predict_kwargs)
    y_val_pred = model.predict(x_val, **predict_kwargs)

    results = {
        "train_accuracy": accuracy_score(y_train, y_train_pred),
        "val_accuracy": accuracy_score(y_val, y_val_pred),
        "train_precision": precision_score(y_train, y_train_pred, average=average),
        "val_precision": precision_score(y_val, y_val_pred, average=average),
        "train_recall": recall_score(y_train, y_train_pred, average=average),
        "val_recall": recall_score(y_val, y_val_pred, average=average),
        "train_f1": f1_score(y_train, y_train_pred, average=average),
        "val_f1": f1_score(y_val, y_val_pred, average=average),
        "val_confusion_matrix": confusion_matrix(y_val, y_val_pred),
        "val_report": classification_report(y_val, y_val_pred),
        "train_predictions": y_train_pred,
        "val_predictions": y_val_pred,
    }

    return results

def diagnose_bias_variance(
    train_scores: Sequence[float],
    val_scores: Sequence[float],
    *,
    low_score_threshold: float = 0.70,
    gap_threshold: float = 0.08,
) -> Dict[str, Any]:
    """Return a heuristic overfitting / underfitting diagnosis.

    The diagnosis is intentionally simple and transparent:
    - low train and validation scores suggest underfitting
    - a large train/validation gap suggests overfitting
    - otherwise the model is considered reasonably balanced
    """
    train_scores = np.asarray(train_scores, dtype=float)
    val_scores = np.asarray(val_scores, dtype=float)

    final_train = float(train_scores[-1])
    final_val = float(val_scores[-1])
    gap = final_train - final_val

    if final_train < low_score_threshold and final_val < low_score_threshold:
        label = "underfitting"
        reason = (
            "Both training and validation performance remain low, so the model is "
            "not learning enough structure from the data."
        )
    elif gap >= gap_threshold and final_train >= final_val:
        label = "overfitting"
        reason = (
            "Training performance is notably higher than validation performance, "
            "which indicates poor generalization."
        )
    else:
        label = "well-balanced"
        reason = (
            "Training and validation curves are reasonably close, so the model "
            "appears to generalize acceptably."
        )

    return {
        "label": label,
        "reason": reason,
        "final_train_score": final_train,
        "final_val_score": final_val,
        "generalization_gap": gap,
    }

def build_learning_curve(
    model_factory: ModelFactory,
    x_train: Any,
    y_train: Any,
    x_val: Any,
    y_val: Any,
    *,
    train_sizes: Optional[Sequence[float]] = None,
    random_state: int = 42,
    fit_kwargs: Optional[Dict[str, Any]] = None,
    average: str = "macro",
    predict_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute learning curves for any classifier with fit/predict methods."""
    x_train = _as_array(x_train)
    y_train = _as_array(y_train)
    x_val = _as_array(x_val)
    y_val = _as_array(y_val)
    fit_kwargs = fit_kwargs or {}
    predict_kwargs = predict_kwargs or {}

    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 10)

    n_train = len(x_train)
    sizes_abs = []
    train_accuracy = []
    val_accuracy = []
    train_f1 = []
    val_f1 = []

    for i, size in enumerate(train_sizes):
        if 0 < size <= 1:
            subset_size = int(max(1, round(size * n_train)))
        else:
            subset_size = int(size)

        subset_size = min(max(subset_size, len(np.unique(y_train))), n_train)
        x_subset, y_subset = _subset_training_data(
            x_train,
            y_train,
            subset_size,
            random_state + i,
        )

        model = model_factory()
        model.fit(x_subset, y_subset, **fit_kwargs)

        y_train_pred = model.predict(x_subset, **predict_kwargs)
        y_val_pred = model.predict(x_val, **predict_kwargs)

        sizes_abs.append(len(x_subset))
        train_accuracy.append(accuracy_score(y_subset, y_train_pred))
        val_accuracy.append(accuracy_score(y_val, y_val_pred))
        train_f1.append(f1_score(y_subset, y_train_pred, average=average))
        val_f1.append(f1_score(y_val, y_val_pred, average=average))

    diagnosis = diagnose_bias_variance(train_accuracy, val_accuracy)

    return {
        "train_sizes": np.array(sizes_abs),
        "train_accuracy": np.array(train_accuracy),
        "val_accuracy": np.array(val_accuracy),
        "train_f1": np.array(train_f1),
        "val_f1": np.array(val_f1),
        "diagnosis": diagnosis,
    }

def plot_learning_curve(
    curve: Dict[str, Any],
    *,
    metric: str = "accuracy",
    title: Optional[str] = None,
):
    """Plot a learning curve returned by ``build_learning_curve``."""
    if metric not in {"accuracy", "f1"}:
        raise ValueError("metric must be 'accuracy' or 'f1'")

    x = curve["train_sizes"]
    train_scores = curve[f"train_{metric}"]
    val_scores = curve[f"val_{metric}"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, train_scores, "o-", linewidth=2, label=f"Train {metric.upper()}")
    ax.plot(x, val_scores, "o-", linewidth=2, label=f"Validation {metric.upper()}")
    ax.set_xlabel("Training set size")
    ax.set_ylabel(metric.upper())
    ax.set_title(title or f"Learning Curve ({metric.upper()})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    return fig, ax

def print_diagnosis(curve: Dict[str, Any]) -> None:
    """Print a short textual summary of the learning-curve diagnosis."""
    diag = curve["diagnosis"]
    print("Learning-curve diagnosis:")
    print(f"  label: {diag['label']}")
    print(f"  gap:   {diag['generalization_gap']:.4f}")
    print(f"  train: {diag['final_train_score']:.4f}")
    print(f"  val:   {diag['final_val_score']:.4f}")
    print(f"  note:  {diag['reason']}")
