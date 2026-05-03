import numpy as np
from src.utils.evaluation import accuracy_score, precision_score, recall_score,  f1_score, confusion_matrix, per_class_accuracy

class KNNClassifier:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Store training data."""
        self.X_train = np.asarray(X_train)
        self.y_train = np.asarray(y_train)
        self.classes = np.unique(y_train)

    def compute_distances(self, X):
        """vectorized Euclidean distance calculation."""
        X = np.asarray(X)
        # Vectorized formula: d² = ||X||² + ||Y||² - 2X·Y
        x_sq = np.sum(X**2, axis=1, keepdims=True)
        train_sq = np.sum(self.X_train**2, axis=1)
        dot = -2 * np.dot(X, self.X_train.T)
        return np.sqrt(np.maximum(x_sq + train_sq + dot, 0))

    def predict(self, X):
        """Predict labels using fast distance calculation and manual voting."""
        dists = self.compute_distances(X)
        predictions = []
        for d in dists:
            k_indices = np.argsort(d)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            votes = {}
            for label in k_nearest_labels:
                votes[label] = votes.get(label, 0) + 1
            predictions.append(max(votes, key=votes.get))
            
        return np.array(predictions)

    def predict_proba(self, X):
        """Predict probabilities using fast distance calculation."""
        dists = self.compute_distances(X)
        probabilities = []
        for d in dists:
            k_indices = np.argsort(d)[:self.k]
            k_nearest_labels = self.y_train[k_indices]
            
            counts = {}
            for label in k_nearest_labels:
                counts[label] = counts.get(label, 0) + 1
            
            prob = [counts.get(c, 0) / self.k for c in self.classes]
            probabilities.append(prob)
            
        return np.array(probabilities)

def grid_search_knn(k_values, X, y, cv=5):
    """Optimized grid search for K (computes distances only once per fold)."""
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    fold_size = len(X) // cv
    
    results = []
    best_k, best_val_acc = k_values[0], -1
    
    for k in k_values:
        val_accs, train_accs = [], []
        for i in range(cv):
            val_idx = indices[i * fold_size : (i + 1) * fold_size]
            train_idx = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
            
            model = KNNClassifier(k=k)
            model.fit(X[train_idx], y[train_idx])
            
            # Validation accuracy
            val_preds = model.predict(X[val_idx])
            val_accs.append(accuracy_score(y[val_idx], val_preds))
            
            # Train accuracy (limited to save time)
            train_eval_idx = train_idx[:500] 
            train_preds = model.predict(X[train_eval_idx])
            train_accs.append(accuracy_score(y[train_eval_idx], train_preds))
        
        mean_val_acc = np.mean(val_accs)
        mean_train_acc = np.mean(train_accs)
        
        results.append({
            'params': {'k': k},
            'val_accuracy': mean_val_acc,
            'train_accuracy': mean_train_acc
        })
        
        if mean_val_acc > best_val_acc:
            best_val_acc, best_k = mean_val_acc, k
            
    return results, {'k': best_k}
