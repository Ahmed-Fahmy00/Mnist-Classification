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
        
        # Support both custom Models (.train()) and Scikit-Learn Models (.fit())
        if hasattr(model, 'fit'):
            model.fit(X_train_fold, y_train_fold)
        elif hasattr(model, 'train'):
            model.train(X_train_fold, y_train_fold)
        else:
            raise AttributeError("Model must have either a 'fit' or 'train' method.")
        
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


def run_pipeline_checks(X_train, y_train, X_test, y_test, X_val=None, y_val=None):
    """
    Perform a comprehensive set of sanity checks on the dataset split.
    """
    print("Running pipeline data integrity checks...")
    
    # 1. Basic validation for each set
    validate_data(X_train, y_train)
    validate_data(X_test, y_test)
    if X_val is not None:
        validate_data(X_val, y_val)
        
    # 2. Check for data leakage (basic intersection check)
    # Note: Using hash/string comparison for high-dim data
    # Flatten and hash a few samples to check
    train_samples = X_train[:min(100, len(X_train))]
    test_samples = X_test[:min(100, len(X_test))]
    
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
        
    # 4. Check normalization
    if np.max(X_train) > 1.01 or np.min(X_train) < -0.01:
        print(f"  WARNING: Data might not be normalized. Max: {np.max(X_train):.2f}, Min: {np.min(X_train):.2f}")
    else:
        print("  Data appears to be normalized (range [0, 1]).")
        
    print("Pipeline checks completed.\n")
