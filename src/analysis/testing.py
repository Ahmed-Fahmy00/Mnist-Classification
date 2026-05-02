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
