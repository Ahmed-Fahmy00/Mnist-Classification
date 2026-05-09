import numpy as np
from skimage.feature import hog


# ── helpers ───────────────────────────────────────────────────────────────────

def _softmax(z):
    z_stable = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def _to_one_hot(y, num_classes):
    N = y.shape[0]
    Y = np.zeros((N, num_classes))
    Y[np.arange(N), y.astype(int)] = 1.0
    return Y


def _l2_gradient(W, lambda_reg):
    grad = lambda_reg * W.copy()
    grad[-1, :] = 0.0   # don't regularise the bias row
    return grad


def _l2_penalty(W, lambda_reg):
    return (lambda_reg / 2.0) * np.sum(W[:-1, :] ** 2)


def _train_val_split(X, y, val_fraction, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(X.shape[0])
    cut = int(X.shape[0] * (1 - val_fraction))
    return X[idx[:cut]], y[idx[:cut]], X[idx[cut:]], y[idx[cut:]]


def _hog_features(x, orientations, pixels_per_cell, cells_per_block):
    return np.array([
        hog(img,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block)
        for img in x
    ])


# ── main class ────────────────────────────────────────────────────────────────

class SoftmaxClassifier:
    """
    Softmax (multinomial logistic) regression with:
      - optional HOG preprocessing
      - mini-batch SGD
      - L2 regularisation
      - early stopping on a held-out validation split

    Usage
    -----
    clf = SoftmaxClassifier(learning_rate=0.1, lambda_reg=1e-5)
    clf.fit(x_train_images, y_train)          # raw 2-D images OR pre-computed features

    preds = clf.predict(x_test_images)        # integer class labels
    proba = clf.predict_proba(x_test_images)  # (N, num_classes) probabilities
    metrics = clf.evaluate(x_test_images, y_test)
    """

    def __init__(
        self,
        num_classes: int = 10,
        batch_size: int = 64,
        learning_rate: float = 0.1,
        lambda_reg: float = 1e-4,
        max_epochs: int = 200,
        val_fraction: float = 0.1,
        validation_patience: int = 7,
        epsilon: float = 1e-4,
        seed: int = 42,
        # HOG preprocessing (set use_hog=False if you pass pre-computed features)
        use_hog: bool = True,
        hog_orientations: int = 9,
        hog_pixels_per_cell: tuple = (4, 4),
        hog_cells_per_block: tuple = (2, 2),
        verbose: bool = True,
    ):
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lambda_reg = lambda_reg
        self.max_epochs = max_epochs
        self.val_fraction = val_fraction
        self.validation_patience = validation_patience
        self.epsilon = epsilon
        self.seed = seed
        self.use_hog = use_hog
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.hog_cells_per_block = hog_cells_per_block
        self.verbose = verbose

        # set after fit()
        self.W_ = None
        self.mean_ = None
        self.std_ = None
        self.is_fitted_ = False

    # ── internal preprocessing ────────────────────────────────────────────────

    def _preprocess(self, X, fit: bool = False):
        """HOG → flatten → standardise → add bias column."""
        if self.use_hog:
            X = _hog_features(
                X,
                self.hog_orientations,
                self.hog_pixels_per_cell,
                self.hog_cells_per_block,
            )

        X = X.reshape(X.shape[0], -1).astype(float)

        if fit:
            self.mean_ = np.mean(X, axis=0, keepdims=True)
            self.std_ = np.std(X, axis=0, keepdims=True) + 1e-8

        X = (X - self.mean_) / self.std_
        X = np.hstack([X, np.ones((X.shape[0], 1))])   # bias column
        return X

    # ── public API ────────────────────────────────────────────────────────────

    def fit(self, X_raw, y):
        """
        Train the classifier.

        Parameters
        ----------
        X_raw : array-like, shape (N, H, W) for images or (N, D) for features
        y     : integer labels, shape (N,)
        """
        y = np.asarray(y)

        # split before preprocessing so val stats don't leak
        X_tr_raw, y_tr, X_val_raw, y_val = _train_val_split(
            X_raw, y, self.val_fraction, self.seed
        )

        X_tr = self._preprocess(X_tr_raw, fit=True)
        X_val = self._preprocess(X_val_raw, fit=False)

        if self.verbose:
            print(f"Feature dimension: {X_tr.shape[1] - 1}  "
                  f"(train {X_tr.shape[0]}, val {X_val.shape[0]})")

        D = X_tr.shape[1]
        Y_tr = _to_one_hot(y_tr, self.num_classes)
        Y_val = _to_one_hot(y_val, self.num_classes)

        rng = np.random.default_rng(self.seed)
        W = rng.uniform(-0.01, 0.01, size=(D, self.num_classes))

        best_val_loss = float("inf")
        best_W = W.copy()
        patience_ctr = 0
        N = X_tr.shape[0]

        for epoch in range(self.max_epochs):
            idx = rng.permutation(N)
            X_sh, Y_sh = X_tr[idx], Y_tr[idx]

            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                X_b, Y_b = X_sh[start:end], Y_sh[start:end]

                Y_hat = _softmax(X_b @ W)
                grad = X_b.T @ (Y_hat - Y_b) / X_b.shape[0]
                grad += _l2_gradient(W, self.lambda_reg)
                W -= self.learning_rate * grad

            # validation loss
            Y_hat_val = _softmax(X_val @ W)
            Y_hat_val = np.clip(Y_hat_val, 1e-15, 1 - 1e-15)
            val_loss = -np.sum(Y_val * np.log(Y_hat_val)) / Y_val.shape[0]
            val_loss += _l2_penalty(W, self.lambda_reg)

            if self.verbose:
                print(f"Epoch {epoch:>4}  val_loss={val_loss:.6f}")

            if val_loss < best_val_loss - self.epsilon:
                best_val_loss = val_loss
                best_W = W.copy()
                patience_ctr = 0
            else:
                patience_ctr += 1

            if patience_ctr >= self.validation_patience:
                if self.verbose:
                    print(f"Early stopping at epoch {epoch + 1}.")
                break
        else:
            if self.verbose:
                print("Reached maximum epochs.")

        self.W_ = best_W
        self.is_fitted_ = True
        return self

    def predict_proba(self, X_raw):
        """Return class probabilities, shape (N, num_classes)."""
        self._check_fitted()
        X = self._preprocess(X_raw, fit=False)
        return _softmax(X @ self.W_)

    def predict(self, X_raw):
        """Return predicted integer class labels, shape (N,)."""
        return np.argmax(self.predict_proba(X_raw), axis=1)

    def evaluate(self, X_raw, y_true):
        """
        Compute accuracy, per-class precision / recall / F1, and confusion matrix.

        Returns
        -------
        dict with keys: accuracy, precision, recall, f1, confusion
        """
        self._check_fitted()
        preds = self.predict(X_raw)
        y_true = np.asarray(y_true).astype(int)
        C = self.num_classes

        cm = np.zeros((C, C), dtype=int)
        for t, p in zip(y_true, preds):
            cm[t, p] += 1

        accuracy = np.trace(cm) / len(y_true)
        precision = np.zeros(C)
        recall = np.zeros(C)
        f1 = np.zeros(C)

        for c in range(C):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            denom = precision[c] + recall[c]
            f1[c] = 2 * precision[c] * recall[c] / denom if denom > 0 else 0.0

        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "confusion": cm,
        }

    # ── internals ─────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Call fit() before predict() / evaluate().")