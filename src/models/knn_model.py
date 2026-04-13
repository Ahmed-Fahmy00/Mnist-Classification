from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier

from src.tests.knn_evaluation import evaluate_knn_model, save_knn_feature_outputs
from src.features.mnist_features import build_features


def tune_k(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    k_values: List[int],
) -> Tuple[int, pd.DataFrame]:
    """Select best k based on validation macro F1."""
    records = []

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        val_pred = model.predict(x_val)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, val_pred, average="macro", zero_division=0
        )
        accuracy = accuracy_score(y_val, val_pred)

        records.append(
            {
                "k": k,
                "val_accuracy": accuracy,
                "val_precision_macro": precision,
                "val_recall_macro": recall,
                "val_f1_macro": f1,
            }
        )

    table = pd.DataFrame(records).sort_values(
        by=["val_f1_macro", "val_accuracy", "k"], ascending=[False, False, True]
    )
    best_k = int(table.iloc[0]["k"])
    return best_k, table


def evaluate_model(
    model: KNeighborsClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    labels: List[int],
) -> Dict[str, object]:
    """Compatibility wrapper; use src.tests.knn_evaluation for new code."""
    return evaluate_knn_model(model=model, x_test=x_test, y_test=y_test, labels=labels)


def run_single_feature_experiment(
    feature_mode: str,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    class_a: int,
    class_b: int,
    k_values: List[int],
    pca_components: float,
    random_state: int,
    output_dir: Optional[Path] = None,
    include_timing: bool = False,
) -> Dict[str, object]:
    t0 = perf_counter()
    feat_train, feat_val, feat_test, info = build_features(
        mode=feature_mode,
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        pca_components=pca_components,
        random_state=random_state,
    )
    t_features = perf_counter()

    best_k, tuning_table = tune_k(
        x_train=feat_train,
        y_train=y_train,
        x_val=feat_val,
        y_val=y_val,
        k_values=k_values,
    )
    t_tune = perf_counter()

    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(feat_train, y_train)
    metrics = evaluate_knn_model(model, feat_test, y_test, labels=[class_a, class_b])
    t_eval = perf_counter()

    metrics_output = {
        "feature_mode": feature_mode,
        "best_k": best_k,
        "class_a": class_a,
        "class_b": class_b,
        "train_samples": int(len(y_train)),
        "val_samples": int(len(y_val)),
        "test_samples": int(len(y_test)),
        **info,
        **metrics,
    }

    if include_timing:
        metrics_output["timing_seconds"] = {
            "feature_build": float(t_features - t0),
            "tuning": float(t_tune - t_features),
            "fit_eval": float(t_eval - t_tune),
            "total": float(t_eval - t0),
        }

    if output_dir is not None:
        saved_output = save_knn_feature_outputs(
            output_dir=output_dir,
            feature_mode=feature_mode,
            class_a=class_a,
            class_b=class_b,
            best_k=best_k,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            tuning_table=tuning_table,
            metrics=metrics,
            extra_info=info,
        )
        if include_timing:
            saved_output["timing_seconds"] = metrics_output["timing_seconds"]
        return saved_output

    return metrics_output
