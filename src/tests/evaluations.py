import numpy as np
from typing import Dict, List, Optional, Tuple

def confusion_matrix_from_predictions(
	y_true: np.ndarray,
	y_pred: np.ndarray,
	labels: List[int],
) -> np.ndarray:
	"""Build a confusion matrix from true and predicted labels."""
	label_to_idx = {label: idx for idx, label in enumerate(labels)}
	cm = np.zeros((len(labels), len(labels)), dtype=np.int64)
	for t, p in zip(y_true, y_pred):
		cm[label_to_idx[int(t)], label_to_idx[int(p)]] += 1
	return cm

def macro_precision_recall_f1(cm: np.ndarray) -> Tuple[float, float, float]:
	"""Compute macro precision, recall, and F1 from a confusion matrix."""
	precisions = []
	recalls = []
	f1_scores = []

	for i in range(cm.shape[0]):
		tp = float(cm[i, i])
		fp = float(np.sum(cm[:, i]) - tp)
		fn = float(np.sum(cm[i, :]) - tp)

		precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
		recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
		f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

		precisions.append(precision)
		recalls.append(recall)
		f1_scores.append(f1)

	return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1_scores))

def accuracy_from_confusion(cm: np.ndarray) -> float:
	"""Compute accuracy from a confusion matrix."""
	total = float(np.sum(cm))
	return float(np.trace(cm) / total) if total > 0 else 0.0

def roc_curve_binary(
	y_true: np.ndarray,
	y_score: np.ndarray,
	pos_label: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Compute FPR, TPR, and thresholds for binary ROC."""
	y_bin = (y_true == pos_label).astype(np.int32)
	order = np.argsort(-y_score)
	y_bin = y_bin[order]
	y_score_sorted = y_score[order]

	p_total = max(int(np.sum(y_bin)), 1)
	n_total = max(int(y_bin.shape[0] - np.sum(y_bin)), 1)

	tps = np.cumsum(y_bin)
	fps = np.cumsum(1 - y_bin)

	distinct_idx = np.where(np.diff(y_score_sorted))[0]
	threshold_idx = np.r_[distinct_idx, y_bin.shape[0] - 1]

	tpr = np.r_[0.0, tps[threshold_idx] / p_total, 1.0]
	fpr = np.r_[0.0, fps[threshold_idx] / n_total, 1.0]
	thresholds = np.r_[np.inf, y_score_sorted[threshold_idx], -np.inf]
	return fpr.astype(float), tpr.astype(float), thresholds.astype(float)

def precision_recall_curve_binary(
	y_true: np.ndarray,
	y_score: np.ndarray,
	pos_label: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Compute precision, recall, and thresholds for binary PR curve."""
	y_bin = (y_true == pos_label).astype(np.int32)
	order = np.argsort(-y_score)
	y_bin = y_bin[order]
	y_score_sorted = y_score[order]

	p_total = max(int(np.sum(y_bin)), 1)

	tps = np.cumsum(y_bin)
	fps = np.cumsum(1 - y_bin)

	distinct_idx = np.where(np.diff(y_score_sorted))[0]
	threshold_idx = np.r_[distinct_idx, y_bin.shape[0] - 1]

	precision = tps[threshold_idx] / np.maximum(tps[threshold_idx] + fps[threshold_idx], 1)
	recall = tps[threshold_idx] / p_total

	precision = np.r_[1.0, precision]
	recall = np.r_[0.0, recall]
	thresholds = y_score_sorted[threshold_idx]
	return precision.astype(float), recall.astype(float), thresholds.astype(float)

def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
	"""Compute area under curve using trapezoidal integration."""
	order = np.argsort(x)
	x_sorted = x[order]
	y_sorted = y[order]
	return float(np.trapz(y_sorted, x_sorted))

def average_precision_binary(
	y_true: np.ndarray,
	y_score: np.ndarray,
	pos_label: int,
) -> float:
	"""Compute average precision for binary classification."""
	precision, recall, _ = precision_recall_curve_binary(y_true, y_score, pos_label=pos_label)
	return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))

def _extract_positive_scores(
	model,
	x_test: np.ndarray,
	labels: List[int],
	pos_label: int,
) -> Optional[np.ndarray]:
	"""Extract positive-class scores from model outputs when available."""
	if not hasattr(model, "predict_proba"):
		return None

	proba = model.predict_proba(x_test)
	proba = np.asarray(proba)

	if proba.ndim == 1:
		return proba.astype(float)
	if proba.ndim != 2:
		return None

	if hasattr(model, "classes_"):
		classes = list(np.asarray(model.classes_).tolist())
	elif hasattr(model, "classes"):
		classes = list(np.asarray(model.classes).tolist())
	else:
		classes = labels

	if pos_label in classes:
		pos_idx = classes.index(pos_label)
	elif len(classes) == 2:
		pos_idx = 1
	else:
		return None

	if pos_idx < 0 or pos_idx >= proba.shape[1]:
		return None

	return proba[:, pos_idx].astype(float)

def evaluate_model(
	model,
	x_test: np.ndarray,
	y_test: np.ndarray,
	labels: List[int],
	pos_label: Optional[int] = None,
	include_prob_metrics: bool = True,
) -> Dict[str, object]:
	"""Compute classification metrics and optionally probability-based metrics."""
	test_pred = model.predict(x_test)

	cm = confusion_matrix_from_predictions(y_test, test_pred, labels=labels)
	precision, recall, f1 = macro_precision_recall_f1(cm)
	acc = accuracy_from_confusion(cm)

	result: Dict[str, object] = {
		"accuracy": float(acc),
		"precision_macro": float(precision),
		"recall_macro": float(recall),
		"f1_macro": float(f1),
		"confusion_matrix": cm.tolist(),
	}

	if not include_prob_metrics:
		return result

	if pos_label is None and len(labels) == 2:
		pos_label = labels[1]
	if pos_label is None:
		return result

	y_score = _extract_positive_scores(model, x_test, labels=labels, pos_label=pos_label)
	if y_score is None:
		return result

	fpr, tpr, _ = roc_curve_binary(y_test, y_score, pos_label=pos_label)
	precision_curve, recall_curve, _ = precision_recall_curve_binary(y_test, y_score, pos_label=pos_label)

	result.update(
		{
			"roc_auc": float(auc_trapezoid(fpr, tpr)),
			"avg_precision": float(average_precision_binary(y_test, y_score, pos_label=pos_label)),
			"fpr": fpr.tolist(),
			"tpr": tpr.tolist(),
			"precision_curve": precision_curve.tolist(),
			"recall_curve": recall_curve.tolist(),
		}
	)

	return result

