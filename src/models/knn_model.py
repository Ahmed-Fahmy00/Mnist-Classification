import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=5):
        """
        Initialize the K-Nearest Neighbors classifier.
        
        Args:
            k (int): Number of neighbors to use.
        """
        self.k = k
        self.X_train = None
        self.y_train = None
        self.classes_ = None

    def train(self, X_train, y_train):
        """
        Train the KNN model (store the training data).
        
        Args:
            X_train (np.ndarray): Training data features.
            y_train (np.ndarray): Training data labels.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.classes_ = np.unique(y_train)

    def compute_distances(self, X):
        """
        Compute the Euclidean distance between test data and training data.
        Uses vectorized operations for efficiency: (X - Y)^2 = X^2 + Y^2 - 2XY
        
        Args:
            X (np.ndarray): Test data features.
            
        Returns:
            np.ndarray: Matrix of distances of shape (num_test, num_train)
        """
        # X shape: (num_test, num_features)
        # self.X_train shape: (num_train, num_features)
        
        dists = -2 * np.dot(X, self.X_train.T)
        dists += np.sum(X**2, axis=1)[:, np.newaxis]
        dists += np.sum(self.X_train**2, axis=1)[np.newaxis, :]
        
        # Ensure no negative values due to floating point inaccuracies
        dists = np.maximum(dists, 0)
        return np.sqrt(dists)

    def predict(self, X, batch_size=500):
        """
        Predict the class labels for the provided data using batching to save RAM.
        
        Args:
            X (np.ndarray): Test data features.
            batch_size (int): Number of samples to process at a time.
            
        Returns:
            np.ndarray: Predicted class labels.
        """
        num_test = X.shape[0]
        y_pred = np.zeros(num_test, dtype=self.y_train.dtype)
        
        for start_idx in range(0, num_test, batch_size):
            end_idx = min(start_idx + batch_size, num_test)
            X_batch = X[start_idx:end_idx]
            
            # Compute distances for this batch only
            dists_batch = self.compute_distances(X_batch)
            
            for i in range(end_idx - start_idx):
                # Find indices of the k nearest neighbors. 
                if self.k < len(dists_batch[i]):
                    closest_y_indices = np.argpartition(dists_batch[i], self.k)[:self.k]
                else:
                    closest_y_indices = np.argsort(dists_batch[i])[:self.k]
                
                # Get the labels of the k nearest neighbors
                closest_y = self.y_train[closest_y_indices]
                
                # Find the most common label
                most_common = Counter(closest_y).most_common(1)
                y_pred[start_idx + i] = most_common[0][0]
                
        return y_pred

    def predict_proba(self, X, batch_size=500):
        """
        Predict class probabilities for the provided data.
        
        Args:
            X (np.ndarray): Test data features.
            batch_size (int): Number of samples to process at a time.
            
        Returns:
            np.ndarray: Probabilities of shape (num_test, num_classes).
        """
        if self.classes_ is None:
            raise RuntimeError("Model must be trained before predicting.")
            
        num_test = X.shape[0]
        num_classes = len(self.classes_)
        probas = np.zeros((num_test, num_classes), dtype=float)
        
        # Create a mapping from class label to index
        label_to_idx = {label: i for i, label in enumerate(self.classes_)}
        
        for start_idx in range(0, num_test, batch_size):
            end_idx = min(start_idx + batch_size, num_test)
            X_batch = X[start_idx:end_idx]
            
            dists_batch = self.compute_distances(X_batch)
            
            for i in range(end_idx - start_idx):
                if self.k < len(dists_batch[i]):
                    closest_y_indices = np.argpartition(dists_batch[i], self.k)[:self.k]
                else:
                    closest_y_indices = np.argsort(dists_batch[i])[:self.k]
                
                closest_y = self.y_train[closest_y_indices]
                
                # Count frequencies of each class among the k neighbors
                counts = Counter(closest_y)
                for label, count in counts.items():
                    if label in label_to_idx:
                        idx = label_to_idx[label]
                        probas[start_idx + i, idx] = count / self.k
                        
        return probas


class BaggedKNNClassifier:
    def __init__(self, n_estimators=5, k=5, max_samples=0.8, random_state=42):
        """
        Initialize the Bagged KNN classifier (Ensemble method).
        
        Args:
            n_estimators (int): Number of KNN models in the ensemble.
            k (int): Number of neighbors to use for each base KNN model.
            max_samples (float): The fraction of training data to use for each base model.
            random_state (int): Seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.k = k
        self.max_samples = max_samples
        self.random_state = random_state
        self.models = []

    def train(self, X_train, y_train):
        """
        Train the ensemble by giving each KNN model a random subset of the training data.
        """
        self.classes_ = np.unique(y_train)
        self.models = []
        n_samples = int(len(X_train) * self.max_samples)
        rng = np.random.default_rng(self.random_state)
        
        for _ in range(self.n_estimators):
            # Bootstrap sampling (with replacement)
            indices = rng.choice(len(X_train), size=n_samples, replace=True)
            X_subset = X_train[indices]
            y_subset = y_train[indices]
            
            # Train a new KNN model on this subset
            model = KNNClassifier(k=self.k)
            model.train(X_subset, y_subset)
            self.models.append(model)

    def predict(self, X, batch_size=500):
        """
        Predict the class labels by majority vote across all models in the ensemble.
        """
        num_test = X.shape[0]
        # Collect predictions from all models
        all_preds = np.zeros((self.n_estimators, num_test), dtype=int)
        
        for idx, model in enumerate(self.models):
            print(f"  Bagged Model {idx+1}/{self.n_estimators} predicting...")
            all_preds[idx] = model.predict(X, batch_size=batch_size)
            
        # Majority voting
        y_pred = np.zeros(num_test, dtype=int)
        for i in range(num_test):
            votes = all_preds[:, i]
            most_common = Counter(votes).most_common(1)
            y_pred[i] = most_common[0][0]
            
        return y_pred

    def predict_proba(self, X, batch_size=500):
        """
        Predict class probabilities by averaging probabilities across all models.
        """
        if not hasattr(self, 'classes_') or self.classes_ is None:
            raise RuntimeError("Model must be trained before predicting.")
            
        num_test = X.shape[0]
        num_classes = len(self.classes_)
        
        # Accumulate probabilities
        avg_probas = np.zeros((num_test, num_classes), dtype=float)
        
        for idx, model in enumerate(self.models):
            print(f"  Bagged Model {idx+1}/{self.n_estimators} calculating probabilities...")
            avg_probas += model.predict_proba(X, batch_size=batch_size)
            
        avg_probas /= self.n_estimators
        return avg_probas
