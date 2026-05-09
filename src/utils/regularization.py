"""Reusable L1/L2 regularization helpers.

These functions are model-agnostic and work for both binary logistic
regression and multiclass softmax/logistic regression. They assume the model
weights are either:

- shape (n_features + 1,)
- shape (n_features + 1, 1)
- shape (n_features + 1, n_classes)

By default, the last row is treated as the bias term and is not regularized.
"""

import numpy as np


VALID_PENALTIES = {None, "none", "l1", "l2"}


def validate_regularization(penalty=None, lambda_=0.0):
    """Validate regularization settings."""
    if penalty not in VALID_PENALTIES:
        raise ValueError("penalty must be one of None, 'none', 'l1', or 'l2'")
    if lambda_ < 0:
        raise ValueError("lambda_ must be non-negative")


def _regularized_view(weights, regularize_bias=False):
    """Return the part of weights that should be regularized."""
    weights = np.asarray(weights)
    if regularize_bias:
        return weights
    if weights.ndim == 0:
        return weights
    return weights[:-1]


def regularization_loss(weights, penalty=None, lambda_=0.0, regularize_bias=False):
    """Compute the L1/L2 regularization loss term.

    Parameters
    ----------
    weights : np.ndarray
        Model parameters.
    penalty : {None, 'none', 'l1', 'l2'}
        Regularization type.
    lambda_ : float
        Regularization strength.
    regularize_bias : bool
        If False, the last weight row is excluded.
    """
    validate_regularization(penalty, lambda_)

    if penalty in [None, "none"] or lambda_ == 0:
        return 0.0

    w_reg = _regularized_view(weights, regularize_bias)

    if penalty == "l1":
        return lambda_ * np.sum(np.abs(w_reg))
    if penalty == "l2":
        return 0.5 * lambda_ * np.sum(w_reg ** 2)

    raise ValueError("Unsupported penalty")


def regularization_gradient(weights, penalty=None, lambda_=0.0, regularize_bias=False):
    """Compute the L1/L2 regularization gradient.

    The returned array has the same shape as ``weights``. If ``regularize_bias``
    is False, the last row is zero.
    """
    validate_regularization(penalty, lambda_)

    weights = np.asarray(weights)
    gradient = np.zeros_like(weights, dtype=float)

    if penalty in [None, "none"] or lambda_ == 0:
        return gradient

    if regularize_bias:
        target = gradient
        source = weights
    else:
        target = gradient[:-1]
        source = weights[:-1]

    if penalty == "l1":
        target[...] = lambda_ * np.sign(source)
    elif penalty == "l2":
        target[...] = lambda_ * source
    else:
        raise ValueError("Unsupported penalty")

    return gradient


def regularization_terms(weights, penalty=None, lambda_=0.0, regularize_bias=False):
    """Return both regularization loss and gradient.

    This is the simplest helper to use inside a training loop:

    ``reg_loss, reg_gradient = regularization_terms(weights, "l2", 0.001)``
    """
    loss = regularization_loss(
        weights=weights,
        penalty=penalty,
        lambda_=lambda_,
        regularize_bias=regularize_bias,
    )
    gradient = regularization_gradient(
        weights=weights,
        penalty=penalty,
        lambda_=lambda_,
        regularize_bias=regularize_bias,
    )
    return loss, gradient


def add_regularization_to_gradient(
    data_gradient,
    weights,
    penalty=None,
    lambda_=0.0,
    regularize_bias=False,
):
    """Add the regularization gradient to an existing data gradient."""
    return np.asarray(data_gradient) + regularization_gradient(
        weights=weights,
        penalty=penalty,
        lambda_=lambda_,
        regularize_bias=regularize_bias,
    )
