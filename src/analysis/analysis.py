from typing import Iterable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid")

def _as_dataframe(rows):
	"""Convert sequence rows to DataFrame safely."""
	if len(rows) == 0:
		return pd.DataFrame()
	return pd.DataFrame(list(rows))

def plot_tuning_curves(
	tuning_tables,
	group_keys,
	x_col = "k",
	score_columns: Optional[Iterable[str]] = None,
	title_prefix = "Tuning",
	x_label: Optional[str] = None,
	y_label = "score",
):
	"""Plot x-wise tuning curves from in-memory tables and return (fig, axes)."""
	group_keys = list(group_keys)
	if not group_keys:
		raise ValueError("group_keys is empty")

	if score_columns is None:
		score_columns = ["val_f1_macro", "val_accuracy"]
	score_columns = list(score_columns)

	fig, axes = plt.subplots(1, len(group_keys), figsize=(5 * len(group_keys), 4), squeeze=False)

	for i, key in enumerate(group_keys):
		if key not in tuning_tables:
			raise KeyError(f"Missing tuning table for key '{key}'")

		tune_df = tuning_tables[key]
		if x_col not in tune_df.columns:
			raise KeyError(f"Missing x column '{x_col}' in tuning table for key '{key}'")
		tune_df = tune_df.sort_values(x_col)

		for col in score_columns:
			if col not in tune_df.columns:
				continue
			label = col.replace("_", " ").title()
			axes[0, i].plot(tune_df[x_col], tune_df[col], marker="o", label=label)

		x_vals = tune_df[x_col].dropna().unique().tolist()
		if len(x_vals) > 0:
			axes[0, i].set_xticks(sorted(x_vals))

		axes[0, i].set_title(f"{title_prefix} ({key})")
		axes[0, i].set_xlabel(x_col if x_label is None else x_label)
		axes[0, i].set_ylabel(y_label)
		axes[0, i].set_ylim(0.0, 1.02)
		axes[0, i].grid(alpha=0.25)
		axes[0, i].legend(fontsize=8)

	plt.tight_layout()
	return fig, axes

def plot_confusion_matrices(
	summary_df: pd.DataFrame,
	labels: Optional[Sequence[object]] = None,
	confusion_col: str = "confusion_matrix",
	group_col: str = "feature_mode",
	title_metric: Optional[str] = "f1_macro",
	title_prefix: Optional[str] = None,
):
	"""Plot confusion matrix heatmaps for each row in summary_df and return (fig, axes)."""
	if summary_df.empty:
		raise ValueError("summary_df is empty")
	if confusion_col not in summary_df.columns:
		raise KeyError(f"Missing column '{confusion_col}'")

	fig, axes = plt.subplots(1, len(summary_df), figsize=(5 * len(summary_df), 4), squeeze=False)

	for i, (_, row) in enumerate(summary_df.iterrows()):
		cm = np.asarray(row[confusion_col])
		if cm.ndim != 2:
			raise ValueError(f"Confusion matrix at row {i} is not 2D")

		if labels is None:
			axis_labels = list(range(cm.shape[0]))
		else:
			axis_labels = list(labels)

		fmt = "d" if np.issubdtype(cm.dtype, np.integer) else ".2f"
		sns.heatmap(
			cm,
			annot=True,
			fmt=fmt,
			cmap="Blues",
			cbar=False,
			ax=axes[0, i],
			xticklabels=axis_labels,
			yticklabels=axis_labels,
		)

		group_val = row.get(group_col, f"row_{i}")
		if title_metric and title_metric in row.index and pd.notna(row[title_metric]):
			metric_name = title_metric.replace("_", " ").upper()
			base_title = f"{group_val} | {metric_name}={float(row[title_metric]):.4f}"
		else:
			base_title = str(group_val)

		if title_prefix:
			axes[0, i].set_title(f"{title_prefix} ({base_title})")
		else:
			axes[0, i].set_title(base_title)

		axes[0, i].set_xlabel("Predicted")
		axes[0, i].set_ylabel("True")

	plt.tight_layout()
	return fig, axes

def plot_metric_comparison(
	summary_df: pd.DataFrame,
	group_col: str = "feature_mode",
	metric_columns: Optional[Iterable[str]] = None,
	title: str = "Model Comparison",
	y_label: str = "Score",
):
	"""Plot a bar chart comparing groups over selected metrics."""
	if summary_df.empty:
		raise ValueError("summary_df is empty")
	if group_col not in summary_df.columns:
		raise KeyError(f"Missing group column '{group_col}'")

	if metric_columns is None:
		metric_columns = ["f1_macro", "accuracy"]
	metric_columns = [col for col in metric_columns if col in summary_df.columns]
	if not metric_columns:
		raise ValueError("No valid metric columns found in summary_df")

	ordered = summary_df.sort_values(by=metric_columns, ascending=[False] * len(metric_columns)).copy()
	plot_df = ordered[[group_col, *metric_columns]].set_index(group_col)

	fig, ax = plt.subplots(figsize=(8, 4))
	plot_df.plot(kind="bar", ax=ax)
	ax.set_title(title)
	ax.set_xlabel(group_col.replace("_", " ").title())
	ax.set_ylabel(y_label)
	ax.set_ylim(0.0, 1.0)
	ax.legend(loc="lower right")
	plt.tight_layout()
	return fig, ax

def summarize_results(
	experiment_results: Sequence[dict],
	fields: Optional[Iterable[str]] = None,
	sort_by: Optional[str] = None,
	ascending: bool = False,
):
	"""Convert experiment outputs into a compact, sorted summary table."""
	summary_df = _as_dataframe(experiment_results)
	if summary_df.empty:
		return summary_df

	if fields is not None:
		selected = [col for col in fields if col in summary_df.columns]
		if not selected:
			raise ValueError("None of the requested fields exist in experiment_results")
		summary_df = summary_df[selected].copy()

	if sort_by is not None and sort_by in summary_df.columns:
		summary_df = summary_df.sort_values(by=sort_by, ascending=ascending).reset_index(drop=True)

	return summary_df

def show_top_config(
	summary_df: pd.DataFrame,
	metric: str = "f1_macro",
	):
	"""Return the best configuration row (as dict) by a metric (descending)."""
	if summary_df.empty:
		raise ValueError("summary_df is empty")
	# Allow common metric synonyms (e.g. 'f1_macro' vs 'f1_score') for backward compatibility
	if metric not in summary_df.columns:
		synonyms = {
			'f1_macro': ['f1_score', 'f1_macro', 'f1'],
			'f1_score': ['f1_score', 'f1_macro', 'f1'],
			'accuracy': ['accuracy', 'acc'],
		}
		found = None
		if metric in synonyms:
			for alt in synonyms[metric]:
				if alt in summary_df.columns:
					found = alt
					break
		else:
			# Try to find any column that contains the metric string
			for col in summary_df.columns:
				if metric in col:
					found = col
					break

		if found is None:
			raise KeyError(f"Metric '{metric}' is not in summary_df. Available metrics: {list(summary_df.columns)}")
		metric = found

	best_row = summary_df.sort_values(by=[metric], ascending=[False]).iloc[0]
	return best_row.to_dict()

def plot_roc_curves(
    summary_df: pd.DataFrame,
    y_test: np.ndarray,
    proba_col: str = "test_probas",
    pos_label: int = 1,
    group_col: str = "feature_mode",
    title: str = "ROC Curves",
):
    """Plot ROC curves for each row in summary_df and return (fig, ax)."""
    from src.analysis.evaluation import roc_curve_binary, auc_trapezoid
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for _, row in summary_df.iterrows():
        probas = np.asarray(row[proba_col])
        # If probas is 2D (num_samples, num_classes), take the column for pos_label
        if probas.ndim == 2:
            # Assuming labels are 0, 1 etc.
            scores = probas[:, pos_label]
        else:
            scores = probas
            
        fpr, tpr, _ = roc_curve_binary(y_test, scores, pos_label=pos_label)
        roc_auc = auc_trapezoid(fpr, tpr)
        
        label = f"{row[group_col]} (AUC = {roc_auc:.4f})"
        ax.plot(fpr, tpr, lw=2, label=label)
        
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    plt.tight_layout()
    return fig, ax

def plot_precision_recall_curves(
    summary_df: pd.DataFrame,
    y_test: np.ndarray,
    proba_col: str = "test_probas",
    pos_label: int = 1,
    group_col: str = "feature_mode",
    title: str = "Precision-Recall Curves",
):
    """Plot Precision-Recall curves for each row in summary_df and return (fig, ax)."""
    from src.analysis.evaluation import precision_recall_curve_binary, average_precision_binary
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for _, row in summary_df.iterrows():
        probas = np.asarray(row[proba_col])
        if probas.ndim == 2:
            scores = probas[:, pos_label]
        else:
            scores = probas
            
        precision, recall, _ = precision_recall_curve_binary(y_test, scores, pos_label=pos_label)
        ap = average_precision_binary(y_test, scores, pos_label=pos_label)
        
        label = f"{row[group_col]} (AP = {ap:.4f})"
        ax.plot(recall, precision, lw=2, label=label)
        
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc="lower left")
    plt.tight_layout()
    return fig, ax

def plot_probability_distribution(
    y_true: np.ndarray,
    y_probas: np.ndarray,
    pos_label: int = 1,
    title: str = "Predicted Probability Distribution",
):
    """
    Plot the distribution of predicted probabilities for each class.
    Useful for seeing how 'confident' the model is.
    """
    if y_probas.ndim == 2:
        scores = y_probas[:, pos_label]
    else:
        scores = y_probas
        
    fig, ax = plt.subplots(figsize=(10, 5))
    
    sns.histplot(scores[y_true == pos_label], color="blue", label=f"Class {pos_label}", kde=True, ax=ax, alpha=0.5)
    
    # Identify other class
    other_classes = np.unique(y_true)
    other_classes = other_classes[other_classes != pos_label]
    if len(other_classes) > 0:
        other_label = other_classes[0]
        sns.histplot(scores[y_true == other_label], color="red", label=f"Class {other_label}", kde=True, ax=ax, alpha=0.5)
        
    ax.set_title(title)
    ax.set_xlabel(f"Predicted Probability of Class {pos_label}")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    return fig, ax

def plot_pca_2d(X_pca, y, title="PCA - 2D Cluster Visualization"):
    """
    Plot the first two components of PCA to visualize class separation.
    """
    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y, palette='viridis', alpha=0.7)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True, alpha=0.3)
    plt.legend(title='Class')
    plt.show()
