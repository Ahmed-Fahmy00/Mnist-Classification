from time import perf_counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.mnist_features import build_features
from src.tests.evaluations import (
	accuracy_from_confusion,
	confusion_matrix_from_predictions,
	evaluate_model as evaluate_generic_model,
	macro_precision_recall_f1,
)

class CustomKNNClassifier:
	"""Simple NumPy KNN classifier (uniform voting, Euclidean distance)."""

	def __init__(self, n_neighbors: int = 5):
		self.n_neighbors = int(n_neighbors)
		self.x_train: Optional[np.ndarray] = None
		self.y_train: Optional[np.ndarray] = None
		self.classes_: Optional[np.ndarray] = None
		self.classes: Optional[np.ndarray] = None


	def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "CustomKNNClassifier":
		self.x_train = np.asarray(x_train, dtype=np.float32)
		self.y_train = np.asarray(y_train)
		self.classes_ = np.unique(self.y_train)
		self.classes = self.classes_

		if self.x_train.ndim != 2:
			raise ValueError("x_train must be a 2D array")
		if self.y_train.ndim != 1:
			raise ValueError("y_train must be a 1D array")
		if self.x_train.shape[0] != self.y_train.shape[0]:
			raise ValueError("x_train and y_train must contain the same number of samples")
		if self.n_neighbors > self.x_train.shape[0]:
			raise ValueError("n_neighbors cannot exceed number of training samples")

		return self

	def _pairwise_distances(self, x_batch: np.ndarray) -> np.ndarray:
		if self.x_train is None:
			raise RuntimeError("Model must be fitted before prediction")

		x_batch = np.asarray(x_batch, dtype=np.float32)
		if x_batch.ndim != 2:
			raise ValueError("x_batch must be a 2D array")
		if x_batch.shape[1] != self.x_train.shape[1]:
			raise ValueError("x_batch feature count must match training feature count")

		batch_sq = np.sum(x_batch * x_batch, axis=1, keepdims=True)
		train_sq = np.sum(self.x_train * self.x_train, axis=1)[None, :]
		d2 = batch_sq + train_sq - 2.0 * (x_batch @ self.x_train.T)
		d2 = np.maximum(d2, 0.0)
		return np.sqrt(d2)

	def _vote_counts(self, neighbor_labels: np.ndarray) -> np.ndarray:
		if self.classes_ is None:
			raise RuntimeError("Model must be fitted before prediction")

		counts = np.zeros((neighbor_labels.shape[0], self.classes_.shape[0]), dtype=np.int32)
		for i in range(neighbor_labels.shape[0]):
			for j, cls in enumerate(self.classes_):
				counts[i, j] = int(np.sum(neighbor_labels[i] == cls))
		return counts

	def predict(self, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
		if self.y_train is None or self.classes_ is None:
			raise RuntimeError("Model must be fitted before prediction")
		if batch_size < 1:
			raise ValueError("batch_size must be >= 1")

		x = np.asarray(x, dtype=np.float32)
		if x.ndim != 2:
			raise ValueError("x must be a 2D array")

		preds = []
		for start in range(0, x.shape[0], batch_size):
			batch = x[start : start + batch_size]
			dists = self._pairwise_distances(batch)
			nn_idx = np.argpartition(dists, kth=self.n_neighbors - 1, axis=1)[:, : self.n_neighbors]
			neighbor_labels = self.y_train[nn_idx]
			counts = self._vote_counts(neighbor_labels)
			pred_idx = np.argmax(counts, axis=1)
			preds.append(self.classes_[pred_idx])

		return np.concatenate(preds, axis=0)

	def predict_proba(self, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
		if self.y_train is None or self.classes_ is None:
			raise RuntimeError("Model must be fitted before prediction")
		if batch_size < 1:
			raise ValueError("batch_size must be >= 1")

		x = np.asarray(x, dtype=np.float32)
		if x.ndim != 2:
			raise ValueError("x must be a 2D array")

		probas = []
		for start in range(0, x.shape[0], batch_size):
			batch = x[start : start + batch_size]
			dists = self._pairwise_distances(batch)
			nn_idx = np.argpartition(dists, kth=self.n_neighbors - 1, axis=1)[:, : self.n_neighbors]
			neighbor_labels = self.y_train[nn_idx]
			counts = self._vote_counts(neighbor_labels).astype(np.float32)
			probas.append(counts / float(self.n_neighbors))

		return np.concatenate(probas, axis=0)

def tune_k(
	x_train: np.ndarray,
	y_train: np.ndarray,
	x_val: np.ndarray,
	y_val: np.ndarray,
	k_values: List[int],
) -> Tuple[int, pd.DataFrame]:
	"""Select best k based on validation macro-F1 (ties: accuracy then smaller k)."""
	if len(k_values) == 0:
		raise ValueError("k_values must not be empty")

	records = []
	labels = [int(v) for v in np.unique(np.concatenate([y_train, y_val]))]

	for k in k_values:
		model = CustomKNNClassifier(n_neighbors=int(k))
		model.fit(x_train, y_train)
		val_pred = model.predict(x_val)

		cm = confusion_matrix_from_predictions(y_val, val_pred, labels=labels)
		precision, recall, f1 = macro_precision_recall_f1(cm)
		accuracy = accuracy_from_confusion(cm)

		records.append(
			{
				"k": int(k),
				"val_accuracy": float(accuracy),
				"val_precision_macro": float(precision),
				"val_recall_macro": float(recall),
				"val_f1_macro": float(f1),
			}
		)

	table = pd.DataFrame(records).sort_values(
		by=["val_f1_macro", "val_accuracy", "k"], ascending=[False, False, True]
	)
	best_k = int(table.iloc[0]["k"])
	return best_k, table

def evaluate_model(
	model: CustomKNNClassifier,
	x_test: np.ndarray,
	y_test: np.ndarray,
	labels: List[int],
) -> Dict[str, object]:
	"""Evaluate KNN and return KNN-focused key names for notebook reporting."""
	metrics = evaluate_generic_model(
		model=model,
		x_test=x_test,
		y_test=y_test,
		labels=labels,
		pos_label=labels[1] if len(labels) == 2 else None,
		include_prob_metrics=False,
	)

	return {
		"test_accuracy": float(metrics["accuracy"]),
		"test_precision_macro": float(metrics["precision_macro"]),
		"test_recall_macro": float(metrics["recall_macro"]),
		"test_f1_macro": float(metrics["f1_macro"]),
		"confusion_matrix": metrics["confusion_matrix"],
	}

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
	include_timing: bool = False,
) -> Dict[str, object]:
	"""Run one KNN experiment for one feature mode and return in-memory outputs only."""
	t0 = perf_counter()
	feat_train, feat_val, feat_test = build_features(
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

	model = CustomKNNClassifier(n_neighbors=best_k)
	model.fit(feat_train, y_train)
	metrics = evaluate_model(model, feat_test, y_test, labels=[class_a, class_b])
	t_eval = perf_counter()

	result: Dict[str, object] = {
		"feature_mode": feature_mode,
		"pca_components": float(pca_components),
		"random_state": int(random_state),
		"best_k": int(best_k),
		"class_a": int(class_a),
		"class_b": int(class_b),
		"train_samples": int(len(y_train)),
		"val_samples": int(len(y_val)),
		"test_samples": int(len(y_test)),
		**metrics,
		"tuning_table": tuning_table,
	}

	if include_timing:
		result["timing_seconds"] = {
			"feature_build": float(t_features - t0),
			"tuning": float(t_tune - t_features),
			"fit_eval": float(t_eval - t_tune),
			"total": float(t_eval - t0),
		}

	return result

