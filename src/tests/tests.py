import numpy as np

from src.features.mnist_features import build_features
from src.tests.evaluations import evaluate_model
from src.models.knn_model import CustomKNNClassifier, tune_k


def _check(condition: bool, name: str, details: str = ""):
	"""Build one structured check entry."""
	return {
		"name": name,
		"passed": bool(condition),
		"details": details,
	}

def run_common_model_checks(
	model,
	x_train,
	y_train,
	x_eval,
	y_eval,
	labels,
	pos_label=None,
	fit_model: bool = True,
):
	"""Run generic fit/predict/metric checks for a model."""
	if pos_label is None and len(labels) == 2:
		pos_label = labels[1]

	checks = []

	if fit_model:
		model.fit(x_train, y_train)

	y_pred = model.predict(x_eval)
	checks.append(
		_check(
			y_pred.shape[0] == x_eval.shape[0],
			"prediction_length_matches_input",
			f"pred={y_pred.shape[0]}, expected={x_eval.shape[0]}",
		)
	)
	checks.append(
		_check(
			set(np.unique(y_pred)).issubset(set(labels)),
			"predicted_labels_within_allowed_set",
			f"pred_labels={sorted(np.unique(y_pred).tolist())}, allowed={sorted(labels)}",
		)
	)

	metrics = evaluate_model(
		model=model,
		x_test=x_eval,
		y_test=y_eval,
		labels=labels,
		pos_label=pos_label,
	)

	required = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "confusion_matrix"]
	for key in required:
		checks.append(_check(key in metrics, f"metric_key_exists::{key}"))

	checks.append(_check(0.0 <= float(metrics.get("accuracy", -1.0)) <= 1.0, "accuracy_in_range_0_1"))
	checks.append(
		_check(
			0.0 <= float(metrics.get("precision_macro", -1.0)) <= 1.0,
			"precision_macro_in_range_0_1",
		)
	)
	checks.append(
		_check(
			0.0 <= float(metrics.get("recall_macro", -1.0)) <= 1.0,
			"recall_macro_in_range_0_1",
		)
	)
	checks.append(_check(0.0 <= float(metrics.get("f1_macro", -1.0)) <= 1.0, "f1_macro_in_range_0_1"))

	cm = np.array(metrics.get("confusion_matrix", []))
	checks.append(
		_check(
			cm.shape == (len(labels), len(labels)),
			"confusion_matrix_shape_matches_labels",
			f"cm_shape={cm.shape}, expected={(len(labels), len(labels))}",
		)
	)

	return {
		"all_passed": all(item["passed"] for item in checks),
		"checks": checks,
		"metrics": metrics,
	}

def run_smoke_checks_with_factory(model_factory):
	"""Run a tiny synthetic-data smoke test using a model factory."""
	x_train = np.array(
		[
			[0.0, 0.0],
			[0.0, 1.0],
			[1.0, 0.0],
			[1.0, 1.0],
			[0.1, 0.2],
			[0.8, 0.9],
		],
		dtype=np.float32,
	)
	y_train = np.array([0, 0, 1, 1, 0, 1], dtype=np.int32)

	x_eval = np.array(
		[
			[0.0, 0.1],
			[0.2, 0.1],
			[0.9, 0.8],
			[1.0, 0.9],
		],
		dtype=np.float32,
	)
	y_eval = np.array([0, 0, 1, 1], dtype=np.int32)

	labels = [0, 1]
	model = model_factory()

	return run_common_model_checks(
		model=model,
		x_train=x_train,
		y_train=y_train,
		x_eval=x_eval,
		y_eval=y_eval,
		labels=labels,
		pos_label=1,
		fit_model=True,
	)

def run_feature_pipeline_checks(
	x_train,
	x_val,
	x_test,
	y_train,
	y_val,
	y_test,
	feature_mode: str,
	pca_components: float,
	random_state: int,
):
	"""Validate feature-building pipeline outputs and basic invariants."""
	checks = []

	try:
		feat_train, feat_val, feat_test = build_features(
			mode=feature_mode,
			x_train=x_train,
			x_val=x_val,
			x_test=x_test,
		)
	except TypeError:
		feat_train, feat_val, feat_test = build_features(
			mode=feature_mode,
			x_train=x_train,
			x_val=x_val,
			x_test=x_test,
			pca_components=pca_components,
			random_state=random_state,
		)

	checks.append(
		_check(
			feat_train.shape[0] == y_train.shape[0],
			"train_rows_match_labels",
			f"feat_train={feat_train.shape[0]}, y_train={y_train.shape[0]}",
		)
	)
	checks.append(
		_check(
			feat_val.shape[0] == y_val.shape[0],
			"val_rows_match_labels",
			f"feat_val={feat_val.shape[0]}, y_val={y_val.shape[0]}",
		)
	)
	checks.append(
		_check(
			feat_test.shape[0] == y_test.shape[0],
			"test_rows_match_labels",
			f"feat_test={feat_test.shape[0]}, y_test={y_test.shape[0]}",
		)
	)

	checks.append(_check(feat_train.ndim == 2, "train_features_are_2d", f"ndim={feat_train.ndim}"))
	checks.append(_check(feat_val.ndim == 2, "val_features_are_2d", f"ndim={feat_val.ndim}"))
	checks.append(_check(feat_test.ndim == 2, "test_features_are_2d", f"ndim={feat_test.ndim}"))

	checks.append(
		_check(
			feat_train.shape[1] == feat_val.shape[1] == feat_test.shape[1],
			"feature_dim_consistent_across_splits",
			f"dims={(feat_train.shape[1], feat_val.shape[1], feat_test.shape[1])}",
		)
	)

	checks.append(_check(np.all(np.isfinite(feat_train)), "train_features_finite"))
	checks.append(_check(np.all(np.isfinite(feat_val)), "val_features_finite"))
	checks.append(_check(np.all(np.isfinite(feat_test)), "test_features_finite"))

	return {
		"all_passed": all(item["passed"] for item in checks),
		"checks": checks,
		"feature_shapes": {
			"train": tuple(feat_train.shape),
			"val": tuple(feat_val.shape),
			"test": tuple(feat_test.shape),
		},
	}

def run_knn_specific_checks(
	x_train,
	y_train,
	x_eval,
	y_eval,
	labels,
	k_values,
):
	"""Validate KNN-specific behavior such as k tuning and predictions."""
	checks = []

	best_k, tuning_table = tune_k(
		x_train=x_train,
		y_train=y_train,
		x_val=x_eval,
		y_val=y_eval,
		k_values=[int(k) for k in k_values],
	)

	checks.append(_check(best_k in list(k_values), "best_k_in_candidate_list", f"best_k={best_k}"))
	checks.append(_check(len(tuning_table) == len(k_values), "tuning_rows_match_k_values_count"))

	required_cols = {
		"k",
		"val_accuracy",
		"val_precision_macro",
		"val_recall_macro",
		"val_f1_macro",
	}
	checks.append(
		_check(
			required_cols.issubset(set(tuning_table.columns.tolist())),
			"tuning_table_has_required_columns",
			f"columns={tuning_table.columns.tolist()}",
		)
	)

	model = CustomKNNClassifier(n_neighbors=int(best_k))
	model.fit(x_train, y_train)
	y_pred = model.predict(x_eval)
	checks.append(_check(y_pred.shape[0] == x_eval.shape[0], "knn_prediction_length_matches"))
	checks.append(
		_check(
			set(np.unique(y_pred)).issubset(set(labels)),
			"knn_predictions_within_labels",
			f"pred_labels={sorted(np.unique(y_pred).tolist())}, labels={sorted(labels)}",
		)
	)

	return {
		"all_passed": all(item["passed"] for item in checks),
		"checks": checks,
		"best_k": int(best_k),
		"tuning_table": tuning_table,
	}

def run_experiment_output_checks(result: dict, expected_labels_count: int = 2):
	"""Validate expected keys and value ranges in experiment output dict."""
	checks = []

	required = {
		"feature_mode",
		"best_k",
		"class_a",
		"class_b",
		"train_samples",
		"val_samples",
		"test_samples",
		"confusion_matrix",
	}
	for key in sorted(required):
		checks.append(_check(key in result, f"result_key_exists::{key}"))

	metric_keys = ["test_accuracy", "test_precision_macro", "test_recall_macro", "test_f1_macro"]
	fallback_metric_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

	has_knn_keys = all(k in result for k in metric_keys)
	has_generic_keys = all(k in result for k in fallback_metric_keys)
	checks.append(_check(has_knn_keys or has_generic_keys, "result_has_metric_keys"))

	keys_to_check = metric_keys if has_knn_keys else fallback_metric_keys
	for key in keys_to_check:
		value = float(result.get(key, -1.0))
		checks.append(_check(0.0 <= value <= 1.0, f"{key}_in_range_0_1", f"value={value}"))

	cm = np.asarray(result.get("confusion_matrix", []))
	checks.append(
		_check(
			cm.shape == (expected_labels_count, expected_labels_count),
			"confusion_matrix_shape_matches_expected_labels",
			f"cm_shape={cm.shape}",
		)
	)

	checks.append(_check(int(result.get("best_k", 0)) > 0, "best_k_positive"))
	checks.append(_check(int(result.get("train_samples", 0)) > 0, "train_samples_positive"))
	checks.append(_check(int(result.get("val_samples", 0)) > 0, "val_samples_positive"))
	checks.append(_check(int(result.get("test_samples", 0)) > 0, "test_samples_positive"))

	return {
		"all_passed": all(item["passed"] for item in checks),
		"checks": checks,
	}

