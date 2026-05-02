import numpy as np
from collections import Counter
from src.features.evaluation import accuracy_score

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


def grid_search_knn(k_values, X_train, y_train, cv=5, max_train_eval_samples=500):
    """
    Memory-safe grid search for KNN.
    Avoids allocating very large train/train distance matrices that can exceed RAM.
    
    Args:
        k_values: List of K values to test.
        X_train: Training features.
        y_train: Training labels.
        cv: Number of cross-validation folds.
        max_train_eval_samples: Max number of training samples used to estimate train accuracy.
        
    Returns:
        list: A list of dictionaries containing K and metrics.
        dict: The best K value based on validation accuracy.
    """

    def validate_data(X, y):
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Mismatch in number of samples: X has {X.shape[0]}, y has {y.shape[0]}")
        if X.ndim != 2:
            raise ValueError(f"Features X must be a 2D array, got {X.ndim}D")
        if y.ndim != 1:
            raise ValueError(f"Labels y must be a 1D array, got {y.ndim}D")
        if not np.all(np.isfinite(X)):
            raise ValueError("Features X contain non-finite values (NaN or Inf)")
    
    validate_data(X_train, y_train)
    fold_size = len(X_train) // cv
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    
    # Store results for each K
    k_val_accs = {k: [] for k in k_values}
    k_train_accs = {k: [] for k in k_values}
    
    for fold_idx in range(cv):
        val_idx = indices[fold_idx * fold_size : (fold_idx + 1) * fold_size]
        train_idx = np.concatenate([indices[:fold_idx * fold_size], indices[(fold_idx + 1) * fold_size:]])
        
        X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
        X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]
        
        # Train a KNN model (K doesn't matter for training, only for predict)
        model = KNNClassifier(k=k_values[0])
        model.fit(X_train_fold, y_train_fold)
        
        # Estimating train accuracy on a bounded subset avoids N_train x N_train blowups.
        if len(X_train_fold) > max_train_eval_samples:
            eval_idx = np.random.choice(len(X_train_fold), size=max_train_eval_samples, replace=False)
            X_train_eval = X_train_fold[eval_idx]
            y_train_eval = y_train_fold[eval_idx]
        else:
            X_train_eval = X_train_fold
            y_train_eval = y_train_fold
        
        # Test all K values without materializing huge global distance matrices.
        for k in k_values:
            model.k = k
            y_val_pred = model.predict(X_val_fold)
            k_val_accs[k].append(accuracy_score(y_val_fold, y_val_pred))
            
            y_train_pred = model.predict(X_train_eval)
            k_train_accs[k].append(accuracy_score(y_train_eval, y_train_pred))
    
    # Aggregate results
    results = []
    best_k = None
    best_val_accuracy = -1
    
    for k in k_values:
        val_acc = np.mean(k_val_accs[k])
        train_acc = np.mean(k_train_accs[k])
        
        results.append({
            'params': {'k': k},
            'val_accuracy': val_acc,
            'train_accuracy': train_acc
        })
        
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_k = k
    
    return results, {'k': best_k}
