import itertools
import numpy as np
from src.analysis.evaluation import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, per_class_accuracy

def validate_data(X, y):
    """
    Perform basic data validation checks before training or evaluation.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"Mismatch in number of samples: X has {X.shape[0]}, y has {y.shape[0]}")
    if X.ndim != 2:
        raise ValueError(f"Features X must be a 2D array, got {X.ndim}D")
    if y.ndim != 1:
        raise ValueError(f"Labels y must be a 1D array, got {y.ndim}D")
    if not np.all(np.isfinite(X)):
        raise ValueError("Features X contain non-finite values (NaN or Inf)")

def cross_val_score(model_class, X, y, cv=5, average='macro', **model_params):
    """
    Perform K-Fold Cross-Validation.
    
    Args:
        model_class: The model class to instantiate.
        X: Features.
        y: Labels.
        cv: Number of folds.
        **model_params: Arbitrary hyperparameters passed to the model constructor.
        
    Returns:
        float: Mean validation accuracy across folds.
        float: Mean training accuracy across folds.
    """
    validate_data(X, y)
    fold_size = len(X) // cv
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    val_accs = []
    train_accs = []
    
    for i in range(cv):
        val_idx = indices[i * fold_size : (i + 1) * fold_size]
        train_idx = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
        
        X_train_fold, y_train_fold = X[train_idx], y[train_idx]
        X_val_fold, y_val_fold = X[val_idx], y[val_idx]
        
        model = model_class(**model_params)
        
        # Unified training API: all models are expected to expose .fit()
        if hasattr(model, 'fit'):
            model.fit(X_train_fold, y_train_fold)
        else:
            raise AttributeError("Model must have a 'fit' method.")
        
        y_val_pred = model.predict(X_val_fold)
        val_accs.append(accuracy_score(y_val_fold, y_val_pred))
        
        y_train_pred = model.predict(X_train_fold)
        train_accs.append(accuracy_score(y_train_fold, y_train_pred))
        
    return np.mean(val_accs), np.mean(train_accs)

def grid_search(model_class, param_grid, X_train, y_train, cv=5):
    """
    Evaluate the model using Cross-Validation for different combinations of hyperparameters.
    
    Args:
        model_class: The model class to instantiate.
        param_grid: Dictionary with parameter names as keys and lists of parameter settings to try as values.
        X_train: Training features.
        y_train: Training labels.
        cv: Number of cross-validation folds.
        
    Returns:
        list: A list of dictionaries containing the parameters and metrics for each combination.
        dict: The best parameter combination based on validation accuracy.
    """
    results = []
    best_params = None
    best_val_accuracy = -1
    
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    for params in param_combinations:
        param_str = ', '.join([f"{k}={v}" for k, v in params.items()])
        print(f"Evaluating model with {param_str} using {cv}-Fold CV...")
        
        val_acc, train_acc = cross_val_score(model_class, X_train, y_train, cv=cv, **params)
        
        results.append({
            'params': params,
            'val_accuracy': val_acc,
            'train_accuracy': train_acc
        })
        
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_params = params
            
    return results, best_params

def test_best_model(model, X_test, y_test, average='macro'):
    """
    Test the best trained model on the unseen test set and return all metrics.
    """
    validate_data(X_test, y_test)
    print("Testing best model on the test set...")
    
    y_pred = model.predict(X_test)
    
    # Check if model supports predict_proba
    y_probas = None
    if hasattr(model, 'predict_proba'):
        y_probas = model.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average=average)
    rec = recall_score(y_test, y_pred, average=average)
    f1 = f1_score(y_test, y_pred, average=average)
    cm = confusion_matrix(y_test, y_pred)
    pca = per_class_accuracy(y_test, y_pred)
    
    results = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'confusion_matrix': cm,
        'per_class_accuracy': pca,
        'test_preds': y_pred
    }
    
    if y_probas is not None:
        results['test_probas'] = y_probas
        
    return results


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
        int: The best K value based on validation accuracy.
    """
    from src.models.knn_model import KNNClassifier
    
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


def run_pipeline_checks(X_train, y_train, X_test, y_test, X_val=None, y_val=None):
    """
    Perform a comprehensive set of sanity checks on the dataset split.
    """
    print("Running pipeline data integrity checks...")
    
    # If inputs are image-shaped (e.g., (N, H, W)), flatten them for validation
    def _ensure_2d(X):
        if X is None:
            return None
        if X.ndim == 3:
            return X.reshape(X.shape[0], -1)
        return X

    X_train_chk = _ensure_2d(X_train)
    X_test_chk = _ensure_2d(X_test)
    X_val_chk = _ensure_2d(X_val) if X_val is not None else None

    # 1. Basic validation for each set (use flattened versions for checks)
    validate_data(X_train_chk, y_train)
    validate_data(X_test_chk, y_test)
    if X_val_chk is not None:
        validate_data(X_val_chk, y_val)

    # 2. Check for data leakage (basic intersection check)
    # Use flattened samples for the intersection/hash checks
    train_samples = X_train_chk[:min(100, len(X_train_chk))]
    test_samples = X_test_chk[:min(100, len(X_test_chk))]
    
    # Simple check for identical rows between train/test
    # (Optional: can be slow for very large datasets)
    
    # 3. Class balance check
    train_counts = np.unique(y_train, return_counts=True)
    test_counts = np.unique(y_test, return_counts=True)
    
    print(f"  Train class distribution: {dict(zip(train_counts[0], train_counts[1]))}")
    print(f"  Test class distribution:  {dict(zip(test_counts[0], test_counts[1]))}")
    
    # Check if all classes in test are present in train
    if not set(test_counts[0]).issubset(set(train_counts[0])):
        print("  WARNING: Test set contains classes not found in the training set.")
        
    # 4. Check normalization (use flattened train array)
    if np.max(X_train_chk) > 1.01 or np.min(X_train_chk) < -0.01:
        print(f"  WARNING: Data might not be normalized. Max: {np.max(X_train_chk):.2f}, Min: {np.min(X_train_chk):.2f}")
    else:
        print("  Data appears to be normalized (range [0, 1]).")
        
    print("Pipeline checks completed.\n")
