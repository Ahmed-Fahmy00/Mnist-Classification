import numpy as np


import numpy as np
from logistic_regression import LogisticRegressionDigit
from data_loader import load_mnist


# Example Usage
# model = LogisticRegressionDigit(
#     model_digit=3,
#     batch_size=32,
#     learning_rate=0.01,
#     validation_patience=5,
#     max_epochs=100,
#     verbose=True
# )

# x_train, y_train, x_test, y_test = load_mnist()
# model.fit(x_train, y_train)

# # Make predictions
# y_pred = model.predict(x_test)                  
# y_proba = model.predict_proba(x_test)           

# print(f"Predicted labels (first 10): {y_pred[:10]}")
# print(f"Predicted probabilities (first 10): {y_proba[:10]}")

# accuracy = model.score(x_test, y_test)
# print(f"Accuracy: {accuracy:.4f}")



class LogisticRegressionDigit:
    """
    A binary logistic regression classifier for digit recognition.

    Trained to detect a single target digit using mini-batch
    gradient descent with class-balanced sample weighting and early stopping.

    Parameters
    ----------
    model_digit : int
        The digit (0–9) this classifier is trained to detect.
    batch_size : int, optional (default=32)
        Number of samples per mini-batch.
    learning_rate : float, optional (default=0.01)
        Step size for gradient descent.
    validation_patience : int, optional (default=5)
        Number of non-improving epochs before early stopping triggers.
    epsilon : float, optional (default=1e-4)
        Minimum improvement in validation loss to reset the patience counter.
    max_epochs : int, optional (default=100)
        Hard cap on the number of training epochs.
    verbose : bool, optional (default=True)
        Whether to print epoch-level loss during training.

    Attributes
    ----------
    weights_ : np.ndarray of shape (n_features + 1, 1)
        Learned weight vector (including bias). Available after calling fit().
    is_fitted_ : bool
        True once fit() has been called successfully.

    Examples
    --------
    >>> clf = LogisticRegressionDigit(model_digit=3)
    >>> clf.fit(x_train, y_train)
    >>> clf.predict(x_test)          # binary labels  (0 or 1)
    >>> clf.predict_proba(x_test)    # probability of being digit 3
    >>> clf.score(x_test, y_test)    # accuracy
    """

    def __init__(
        self,
        model_digit,
        batch_size=32,
        learning_rate=0.01,
        validation_patience=5,
        epsilon=1e-4,
        max_epochs=100,
        verbose=True,
    ):
        self.model_digit = model_digit
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.validation_patience = validation_patience
        self.epsilon = epsilon
        self.max_epochs = max_epochs
        self.verbose = verbose

        self.weights_ = None
        self.is_fitted_ = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _add_bias(self, x_flat):
        """Append a column of ones (bias term) to a 2-D feature matrix."""
        return np.hstack([x_flat, np.ones((x_flat.shape[0], 1))])

    def _preprocess(self, x):
        x = x.astype(float) / 255.0  
        return self._add_bias(x.reshape(x.shape[0], -1))

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError(
                "This classifier has not been fitted yet. Call fit() first."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, x_train, y_train, x_val=None, y_val=None):
        """
        Train the classifier.

        Parameters
        ----------
        x_train : np.ndarray, shape (N, ...)
            Training images (any shape; they will be flattened internally).
        y_train : np.ndarray, shape (N,)
            Integer labels for the training set.
        x_val : np.ndarray or None, shape (M, ...), optional
            Validation images. If None, 15 % of x_train is used automatically.
        y_val : np.ndarray or None, shape (M,), optional
            Integer labels for the validation set.

        Returns
        -------
        self
            The fitted classifier (enables method chaining).
        """
        # ── Optional automatic train/val split ──────────────────────────
        if x_val is None or y_val is None:
            val_size = max(1, int(0.15 * len(x_train)))
            x_val, y_val = x_train[-val_size:], y_train[-val_size:]
            x_train, y_train = x_train[:-val_size], y_train[:-val_size]

        # ── Binarise labels ─────────────────────────────────────────────
        y_tr = (y_train == self.model_digit).astype(float).reshape(-1, 1)
        y_v = (y_val == self.model_digit).astype(float).reshape(-1, 1)

        # ── Feature engineering ─────────────────────────────────────────
        x_tr_b = self._preprocess(x_train)   # (N, F+1)
        x_v_b = self._preprocess(x_val)      # (M, F+1)

        # ── Weight init ──────────────────────────────────────────────────
        w        = np.random.uniform(low=-0.01, high=0.01, size=(x_tr_b.shape[1], 1))
        best_v_loss = float('inf')
        best_w   = w.copy()
        counter  = 0
        epoch    = 0

        # ── Class-balanced sample weights ────────────────────────────────
        N     = x_tr_b.shape[0]
        n_pos = np.sum(y_tr == 1)
        n_neg = N - n_pos

        # Note: dividing by 2 is essential to keep the overall average weight at 1,
        # hence keeping the same learning rate scale as unweighted logistic regression
        w_pos = N / (2 * n_pos)
        w_neg = N / (2 * n_neg)

        # ── Training loop ────────────────────────────────────────────────
        while epoch < self.max_epochs:
            # Since we are doing Mini-batch gradient descent,
            # we need to shuffle the data to prevent the model from learning the order of the data.
            indices    = np.random.permutation(N)
            x_shuffled = x_tr_b[indices]
            y_shuffled = y_tr[indices]

            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)

                xi_batch = x_shuffled[start:end]   # (B, F+1)
                yi_batch = y_shuffled[start:end]   # (B, 1)

                z_batch  = np.dot(xi_batch, w)                          # (B, 1)
                y_hat    = self._sigmoid(z_batch)                       # (B, 1)

                sample_weights = yi_batch * w_pos + (1 - yi_batch) * w_neg
                error  = (y_hat - y_shuffled[start:end]) * sample_weights  # (B, 1)
                de_dw  = np.dot(xi_batch.T, error) / xi_batch.shape[0]     # (F+1, 1)
                w     -= self.learning_rate * de_dw

            z_v     = np.dot(x_v_b, w)
            y_hat_v = self._sigmoid(z_v)  
            y_hat_v = np.clip(y_hat_v, 1e-15, 1 - 1e-15)  # Cap to avoid log(0) errors

            v_loss = -np.mean(y_v * np.log(y_hat_v) + (1 - y_v) * np.log(1 - y_hat_v))

            if v_loss < best_v_loss - self.epsilon:
                best_v_loss = v_loss
                best_w      = w.copy()
                counter     = 0
            else:
                counter += 1

            if counter == self.validation_patience:
                self.weights_   = best_w
                self.is_fitted_ = True
                return self

        if self.verbose:
            print(f"Epoch: {epoch}, Loss:  {v_loss}")

        epoch += 1

        if epoch == self.max_epochs:
            if self.verbose:
                print("Reached maximum epochs. Stopping training...")
            self.weights_   = best_w
            self.is_fitted_ = True
            return self

    def predict_proba(self, x):
        self._check_fitted()

        x_bias = self._preprocess(x)

        z     = np.dot(x_bias, self.weights_)
        y_hat = self._sigmoid(z)

        return y_hat.reshape(-1)

    def predict(self, x, threshold=0.5):
        self._check_fitted()

        y_proba = self.predict_proba(x)
        y_pred  = (y_proba >= threshold).astype(int)

        return y_pred

    def score(self, x, y, threshold=0.5):
        self._check_fitted()

        y_true_bin = (y == self.model_digit).astype(int)
        y_pred     = self.predict(x, threshold=threshold)
        accuracy   = np.mean(y_pred == y_true_bin)

        return accuracy

    def __repr__(self):
        return (
            f"LogisticRegressionDigit("
            f"model_digit={self.model_digit}, "
            f"lr={self.learning_rate}, "
            f"batch_size={self.batch_size}, "
            f"patience={self.validation_patience})"
        )