import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=5):
        """Initialize K-Nearest Neighbors classifier."""
        self.k = k
        self.X_train = None
        self.y_train = None
        self.classes_ = None
        self._x_train_sq = None

    def fit(self, X_train, y_train):
        """Store training data."""
        # Use float32 to cut memory footprint roughly in half vs float64.
        self.X_train = np.asarray(X_train, dtype=np.float32, order='C')
        self.y_train = np.asarray(y_train)
        self.classes_ = np.unique(y_train)
        self._x_train_sq = np.sum(self.X_train**2, axis=1)
    
    def compute_distances(self, X):
        """
        Compute vectorized Euclidean distances between X and training data.
        Formula: d² = ||X||² + ||Y||² - 2X·Y
        """
        X = np.asarray(X, dtype=np.float32, order='C')
        dists = -2 * np.dot(X, self.X_train.T)
        dists += np.sum(X**2, axis=1, keepdims=True)
        dists += self._x_train_sq
        return np.sqrt(np.maximum(dists, 0))

    def predict(self, X, dists=None, batch_size=100):
        """
        Predict class labels for samples in X.
        
        Args:
            X: Features (required if dists is None)
            dists: Precomputed distance matrix (optional, skips computation)
            batch_size: Process in batches to save memory
        """
        predictions = np.zeros(len(X) if dists is None else len(dists), dtype=self.y_train.dtype)
        
        if dists is None:
            # Compute distances with batching
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                X_batch = X[start:end]
                dists_batch = self.compute_distances(X_batch)
                
                for i in range(len(X_batch)):
                    k_idx = np.argpartition(dists_batch[i], min(self.k, len(dists_batch[i])-1))[:self.k]
                    predictions[start + i] = Counter(self.y_train[k_idx]).most_common(1)[0][0]
        else:
            # Use precomputed distances (no batching needed)
            for i in range(len(dists)):
                k_idx = np.argpartition(dists[i], min(self.k, len(dists[i])-1))[:self.k]
                predictions[i] = Counter(self.y_train[k_idx]).most_common(1)[0][0]
        
        return predictions

    def predict_proba(self, X, dists=None, batch_size=100):
        """
        Predict class probabilities.
        
        Args:
            X: Features (required if dists is None)
            dists: Precomputed distance matrix (optional, skips computation)
            batch_size: Process in batches to save memory
        """
        n_samples = len(X) if dists is None else len(dists)
        probas = np.zeros((n_samples, len(self.classes_)), dtype=float)
        label_to_idx = {label: i for i, label in enumerate(self.classes_)}
        
        if dists is None:
            # Compute distances with batching
            for start in range(0, len(X), batch_size):
                end = min(start + batch_size, len(X))
                X_batch = X[start:end]
                dists_batch = self.compute_distances(X_batch)
                
                for i in range(len(X_batch)):
                    k_idx = np.argpartition(dists_batch[i], min(self.k, len(dists_batch[i])-1))[:self.k]
                    for label, count in Counter(self.y_train[k_idx]).items():
                        if label in label_to_idx:
                            probas[start + i, label_to_idx[label]] = count / self.k
        else:
            # Use precomputed distances
            for i in range(len(dists)):
                k_idx = np.argpartition(dists[i], min(self.k, len(dists[i])-1))[:self.k]
                for label, count in Counter(self.y_train[k_idx]).items():
                    if label in label_to_idx:
                        probas[i, label_to_idx[label]] = count / self.k
        
        return probas
