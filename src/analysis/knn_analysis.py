from typing import Iterable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

def plot_knn_tuning_curves(
	tuning_tables: Mapping[str, pd.DataFrame],
	feature_modes: Iterable[str],
	score_columns: Optional[Iterable[str]] = None,
):
	"""Plot k-wise tuning curves from in-memory tables and return (fig, axes)."""
	feature_modes = list(feature_modes)
	if not feature_modes:
		raise ValueError("feature_modes is empty")

	if score_columns is None:
		score_columns = ["val_f1_macro", "val_accuracy"]
	score_columns = list(score_columns)

	fig, axes = plt.subplots(1, len(feature_modes), figsize=(5 * len(feature_modes), 4), squeeze=False)

	for i, mode in enumerate(feature_modes):
		if mode not in tuning_tables:
			raise KeyError(f"Missing tuning table for feature mode '{mode}'")

		tune_df = tuning_tables[mode].sort_values("k")
		for col in score_columns:
			if col not in tune_df.columns:
				continue
			label = col.replace("_", " ").title()
			axes[0, i].plot(tune_df["k"], tune_df[col], marker="o", label=label)

		axes[0, i].set_title(f"KNN Tuning ({mode})")
		axes[0, i].set_xlabel("k")
		axes[0, i].set_ylabel("score")
		axes[0, i].set_xticks(sorted(tune_df["k"].unique()))
		axes[0, i].set_ylim(0.0, 1.02)
		axes[0, i].grid(alpha=0.25)
		axes[0, i].legend(fontsize=8)

	plt.tight_layout()
	return fig, axes

def plot_knn_confusion_matrices(
	summary_df: pd.DataFrame,
	class_a: int,
	class_b: int,
):
	"""Plot confusion matrix heatmaps for each row in summary_df and return (fig, axes)."""
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
	return fig, axes

def plot_knn_feature_comparison(
	summary_df: pd.DataFrame,
	metric_columns: Optional[Iterable[str]] = None,
):
	"""Plot a bar chart comparing feature modes over selected metrics."""
	if summary_df.empty:
		raise ValueError("summary_df is empty")

	if metric_columns is None:
		metric_columns = ["test_f1_macro", "test_accuracy"]
	metric_columns = [col for col in metric_columns if col in summary_df.columns]
	if not metric_columns:
		raise ValueError("No valid metric columns found in summary_df")

	ordered = summary_df.sort_values(by=metric_columns, ascending=[False] * len(metric_columns)).copy()
	plot_df = ordered[["feature_mode", *metric_columns]].set_index("feature_mode")

	fig, ax = plt.subplots(figsize=(8, 4))
	plot_df.plot(kind="bar", ax=ax)
	ax.set_title("KNN Feature Comparison")
	ax.set_xlabel("Feature mode")
	ax.set_ylabel("Score")
	ax.set_ylim(0.0, 1.0)
	ax.legend(loc="lower right")
	plt.tight_layout()
	return fig, ax

def summarize_knn_results(
	experiment_results: Sequence[dict],
	sort_by: str = "test_f1_macro",
	ascending: bool = False,
):
	"""Convert experiment dict outputs into a compact, sorted summary table."""
	if len(experiment_results) == 0:
		return pd.DataFrame()

	summary_rows = []
	for row in experiment_results:
		kept = {
			"feature_mode": row.get("feature_mode"),
			"best_k": row.get("best_k"),
			"test_accuracy": row.get("test_accuracy"),
			"test_precision_macro": row.get("test_precision_macro"),
			"test_recall_macro": row.get("test_recall_macro"),
			"test_f1_macro": row.get("test_f1_macro"),
			"confusion_matrix": row.get("confusion_matrix"),
			"train_samples": row.get("train_samples"),
			"val_samples": row.get("val_samples"),
			"test_samples": row.get("test_samples"),
		}
		summary_rows.append(kept)

	summary_df = pd.DataFrame(summary_rows)
	if sort_by in summary_df.columns:
		summary_df = summary_df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

	return summary_df

def show_top_config(
	summary_df: pd.DataFrame,
	metric: str = "test_f1_macro",
	):
	"""Return the best configuration row (as dict) by a metric (descending)."""
	if summary_df.empty:
		raise ValueError("summary_df is empty")
	if metric not in summary_df.columns:
		raise KeyError(f"Metric '{metric}' is not in summary_df")

	best_row = summary_df.sort_values(by=[metric], ascending=[False]).iloc[0]
	return best_row.to_dict()

