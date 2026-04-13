import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from src.analysis.knn_analysis import (
    save_knn_confusion_matrices,
    save_knn_feature_comparison,
    save_knn_tuning_curves,
)
from src.features.mnist_features import (
    class_distribution,
    load_mnist,
    normalize_pixels,
    select_binary_classes,
    split_data,
)
from src.models.knn_model import run_single_feature_experiment


def _resolve_path(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (base / p)


def _load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_failure_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for _, row in summary_df.iterrows():
        cm = row["confusion_matrix"]
        tn, fp = cm[0]
        fn, tp = cm[1]
        total = tn + fp + fn + tp
        records.append(
            {
                "feature_mode": row["feature_mode"],
                "best_k": row["best_k"],
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "total_errors": int(fp + fn),
                "error_rate": float((fp + fn) / total if total else 0.0),
            }
        )
    return pd.DataFrame(records).sort_values(by=["total_errors", "error_rate"], ascending=[True, True])


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    project_root = None
    for candidate in [config_path.parent.resolve(), *config_path.parent.resolve().parents]:
        if (candidate / "data" / "mnist.npz").exists():
            project_root = candidate
            break
    if project_root is None:
        raise RuntimeError("Could not find project root containing data/mnist.npz")
    cfg = _load_config(config_path)

    random_state = int(cfg["random_state"])
    class_a = int(cfg["class_a"])
    class_b = int(cfg["class_b"])
    test_size = float(cfg["test_size"])
    val_size = float(cfg["val_size"])
    features = [str(v).strip().lower() for v in cfg["features"]]
    k_values = [int(v) for v in cfg["k_values"]]
    pca_components = float(cfg["pca_components"])

    data_path = _resolve_path(project_root, str(cfg["data_path"]))
    output_dir = _resolve_path(project_root, str(cfg["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = perf_counter()
    x_all, y_all = load_mnist(str(data_path))
    x_binary, y_binary = select_binary_classes(x_all, y_all, class_a, class_b)

    x_train, x_val, x_test, y_train, y_val, y_test = split_data(
        x_binary,
        y_binary,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )

    x_train = normalize_pixels(x_train)
    x_val = normalize_pixels(x_val)
    x_test = normalize_pixels(x_test)
    t_prep = perf_counter()

    summary_rows: List[Dict[str, Any]] = []
    timing_rows: List[Dict[str, Any]] = [
        {
            "stage": "data_preprocessing",
            "feature_mode": "all",
            "seconds": float(t_prep - t_start),
        }
    ]

    for mode in features:
        result = run_single_feature_experiment(
            feature_mode=mode,
            x_train=x_train,
            x_val=x_val,
            x_test=x_test,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            class_a=class_a,
            class_b=class_b,
            k_values=k_values,
            pca_components=pca_components,
            random_state=random_state,
            output_dir=output_dir,
            include_timing=True,
        )
        summary_rows.append(result)

        timing = result.get("timing_seconds", {})
        for stage in ["feature_build", "tuning", "fit_eval", "total"]:
            timing_rows.append(
                {
                    "stage": stage,
                    "feature_mode": mode,
                    "seconds": float(timing.get(stage, 0.0)),
                }
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["test_f1_macro", "test_accuracy"],
        ascending=[False, False],
    )

    summary_df.to_csv(output_dir / "summary_knn.csv", index=False)

    best_knn = summary_df.iloc[0].to_dict()
    for key, value in list(best_knn.items()):
        if np.isscalar(value) and pd.isna(value):
            best_knn[key] = None

    with open(output_dir / "best_knn.json", "w", encoding="utf-8") as f:
        json.dump(best_knn, f, indent=2)

    benchmark_df = pd.DataFrame(timing_rows)
    benchmark_df.to_csv(output_dir / "benchmark_timing_breakdown.csv", index=False)

    failure_df = _build_failure_summary(summary_df)
    failure_df.to_csv(output_dir / "failure_analysis.csv", index=False)

    counts_all = class_distribution(y_binary)
    counts_train = class_distribution(y_train)
    counts_val = class_distribution(y_val)
    counts_test = class_distribution(y_test)

    run_info = {
        "class_distribution_all": counts_all,
        "class_distribution_train": counts_train,
        "class_distribution_val": counts_val,
        "class_distribution_test": counts_test,
        "feature_modes": features,
        "k_values": k_values,
        "test_size": test_size,
        "val_size": val_size,
        "data_path": str(data_path),
        "random_state": random_state,
    }
    with open(output_dir / "run_info.json", "w", encoding="utf-8") as f:
        json.dump(run_info, f, indent=2)

    tuning_plot = save_knn_tuning_curves(output_dir=output_dir, feature_modes=features)
    cm_plot = save_knn_confusion_matrices(
        output_dir=output_dir,
        summary_df=summary_df,
        class_a=class_a,
        class_b=class_b,
    )
    comparison_plot = save_knn_feature_comparison(output_dir=output_dir, summary_df=summary_df)

    artifacts = {
        "summary": "summary_knn.csv",
        "best": "best_knn.json",
        "timing": "benchmark_timing_breakdown.csv",
        "failure": "failure_analysis.csv",
        "run_info": "run_info.json",
        "plots": [
            tuning_plot.name,
            cm_plot.name,
            comparison_plot.name,
        ],
    }
    with open(output_dir / "artifacts_manifest.json", "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run config-driven Phase 1 KNN benchmark pipeline")
    parser.add_argument(
        "--config",
        default="src/pipelines/configs/phase1_knn_config.json",
        help="Path to JSON config file",
    )
    args = parser.parse_args()

    output_dir = run(Path(args.config))
    print(f"Saved KNN benchmark outputs to: {output_dir}")


if __name__ == "__main__":
    main()
