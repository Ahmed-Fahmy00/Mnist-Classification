import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.neighbors import KNeighborsClassifier


def evaluate_knn_model(
    model: KNeighborsClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    labels: List[int],
) -> Dict[str, object]:
    """Return standard classification metrics for a fitted KNN model."""
    test_pred = model.predict(x_test)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, test_pred, average="macro", zero_division=0
    )
    acc = accuracy_score(y_test, test_pred)
    cm = confusion_matrix(y_test, test_pred, labels=labels)

    return {
        "test_accuracy": float(acc),
        "test_precision_macro": float(precision),
        "test_recall_macro": float(recall),
        "test_f1_macro": float(f1),
        "confusion_matrix": cm.tolist(),
    }


def save_knn_feature_outputs(
    output_dir: Path,
    feature_mode: str,
    class_a: int,
    class_b: int,
    best_k: int,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    tuning_table: pd.DataFrame,
    metrics: Dict[str, object],
    extra_info: Dict[str, float],
) -> Dict[str, object]:
    """Persist per-feature tuning + metrics artifacts for KNN experiments."""
    feature_output = output_dir / feature_mode
    feature_output.mkdir(parents=True, exist_ok=True)

    tuning_table.to_csv(feature_output / "validation_tuning.csv", index=False)

    metrics_output = {
        "feature_mode": feature_mode,
        "best_k": int(best_k),
        "class_a": class_a,
        "class_b": class_b,
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        **extra_info,
        **metrics,
    }

    with open(feature_output / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_output, f, indent=2)

    cm = np.array(metrics["confusion_matrix"])
    cm_df = pd.DataFrame(cm, index=[class_a, class_b], columns=[class_a, class_b])
    cm_df.to_csv(feature_output / "confusion_matrix.csv")

    return metrics_output
