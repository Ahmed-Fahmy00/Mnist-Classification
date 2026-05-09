"""Generic bias-variance and model-complexity analysis utilities.
The functions in this module work with any classifier that exposes:
- ``fit(X, y)``
- ``predict(X)``
Use a model factory so each experiment gets a fresh model instance.
"""
import numpy as np
import pandas as pd

def accuracy_metric(y_true, y_pred):
    """Default classification metric."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))

def make_train_subset_indices(y, train_size, seed=42, stratified=True):
    """Create subset indices for a requested train size.

    ``train_size`` can be a float in (0, 1] or an integer sample count.
    Stratified sampling keeps class proportions stable across learning curves.
    """
    y = np.asarray(y)
    n_samples = len(y)

    if isinstance(train_size, float):
        if not 0 < train_size <= 1:
            raise ValueError("float train_size must be in (0, 1]")
        requested = max(1, int(round(train_size * n_samples)))
    else:
        requested = int(train_size)

    if requested < 1 or requested > n_samples:
        raise ValueError("integer train_size must be between 1 and len(y)")

    rng = np.random.default_rng(seed)

    if not stratified:
        return rng.permutation(n_samples)[:requested]

    selected = []
    labels, counts = np.unique(y, return_counts=True)
    raw_allocations = requested * counts / n_samples
    allocations = np.floor(raw_allocations).astype(int)

    remainder = requested - np.sum(allocations)
    if remainder > 0:
        fractions = raw_allocations - allocations
        for idx in np.argsort(fractions)[::-1][:remainder]:
            allocations[idx] += 1

    for label, allocation in zip(labels, allocations):
        label_indices = np.where(y == label)[0]
        allocation = min(allocation, len(label_indices))
        if allocation > 0:
            selected.extend(rng.choice(label_indices, size=allocation, replace=False))

    selected = np.asarray(selected)
    rng.shuffle(selected)
    return selected

def fit_model(model, X_train, y_train, fit_kwargs=None):
    """Fit a model and return it, regardless of whether fit returns self."""
    fit_kwargs = fit_kwargs or {}
    fitted = model.fit(X_train, y_train, **fit_kwargs)
    return model if fitted is None else fitted

def evaluate_model(
    model,
    X,
    y,
    metric_fn=accuracy_metric,
    transform_y_fn=None,
    predict_kwargs=None,
):
    """Evaluate a fitted model with optional true-label transformation."""
    predict_kwargs = predict_kwargs or {}
    y_eval = transform_y_fn(y) if transform_y_fn is not None else y
    y_pred = model.predict(X, **predict_kwargs)
    return metric_fn(y_eval, y_pred)

def learning_curve(
    model_factory,
    X_train,
    y_train,
    X_val,
    y_val,
    train_sizes=(0.1, 0.25, 0.5, 0.75, 1.0),
    seeds=(42,),
    metric_fn=accuracy_metric,
    transform_y_fn=None,
    fit_kwargs=None,
    predict_kwargs=None,
    stratified=True,
    model_name="model",
):
    """Run a generic bias-variance learning curve experiment.

    Parameters
    ----------
    model_factory : callable
        Callable with no required arguments that returns a fresh model.
    transform_y_fn : callable or None
        Use this when a model predicts transformed labels. For example,
        binary one-vs-rest logistic regression can use:
        ``lambda y: (y == 3).astype(int)``.

    Returns
    -------
    pandas.DataFrame
        Columns: model, train_size, seed, train_score, val_score, gap.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)

    rows = []
    for train_size in train_sizes:
        for seed in seeds:
            subset_idx = make_train_subset_indices(
                y_train,
                train_size=train_size,
                seed=seed,
                stratified=stratified,
            )

            model = model_factory()
            model = fit_model(
                model,
                X_train[subset_idx],
                y_train[subset_idx],
                fit_kwargs=fit_kwargs,
            )

            train_score = evaluate_model(
                model,
                X_train[subset_idx],
                y_train[subset_idx],
                metric_fn=metric_fn,
                transform_y_fn=transform_y_fn,
                predict_kwargs=predict_kwargs,
            )
            val_score = evaluate_model(
                model,
                X_val,
                y_val,
                metric_fn=metric_fn,
                transform_y_fn=transform_y_fn,
                predict_kwargs=predict_kwargs,
            )

            rows.append(
                {
                    "model": model_name,
                    "train_size": len(subset_idx),
                    "train_size_requested": train_size,
                    "seed": seed,
                    "train_score": train_score,
                    "val_score": val_score,
                    "gap": train_score - val_score,
                }
            )

    return pd.DataFrame(rows)

def complexity_curve(
    model_factory,
    param_grid,
    X_train,
    y_train,
    X_val,
    y_val,
    metric_fn=accuracy_metric,
    transform_y_fn=None,
    fit_kwargs=None,
    predict_kwargs=None,
    model_name="model",
):
    """Evaluate train/validation scores across model complexity settings.

    ``model_factory`` is called as ``model_factory(**params)`` for each
    dictionary in ``param_grid``.
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_val = np.asarray(X_val)
    y_val = np.asarray(y_val)

    rows = []
    for params in param_grid:
        model = model_factory(**params)
        model = fit_model(model, X_train, y_train, fit_kwargs=fit_kwargs)

        train_score = evaluate_model(
            model,
            X_train,
            y_train,
            metric_fn=metric_fn,
            transform_y_fn=transform_y_fn,
            predict_kwargs=predict_kwargs,
        )
        val_score = evaluate_model(
            model,
            X_val,
            y_val,
            metric_fn=metric_fn,
            transform_y_fn=transform_y_fn,
            predict_kwargs=predict_kwargs,
        )

        row = {
            "model": model_name,
            "train_score": train_score,
            "val_score": val_score,
            "gap": train_score - val_score,
        }
        row.update(params)
        rows.append(row)

    return pd.DataFrame(rows)

def summarize_bias_variance(
    results,
    train_score_col="train_score",
    val_score_col="val_score",
    gap_col="gap",
    target_score=0.90,
    gap_threshold=0.05,
):
    """Add a simple bias/variance diagnosis to a results DataFrame."""
    summary = results.copy()

    def diagnose(row):
        train_score = row[train_score_col]
        val_score = row[val_score_col]
        gap = row[gap_col]

        if train_score < target_score and val_score < target_score:
            return "high_bias"
        if gap > gap_threshold and train_score >= target_score:
            return "high_variance"
        return "balanced"

    summary["diagnosis"] = summary.apply(diagnose, axis=1)
    return summary

