import numpy as np

def accuracy_score(y_true, y_pred):
    """Calculate the accuracy of the predictions."""
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total

def precision_score(y_true, y_pred, pos_label=None, average='binary'):
    """Calculate the precision."""
    if average == 'macro':
        labels = np.unique(np.concatenate([y_true, y_pred]))
        precisions = []
        for label in labels:
            tp = np.sum((y_true == label) & (y_pred == label))
            fp = np.sum((y_true != label) & (y_pred == label))
            if tp + fp == 0:
                precisions.append(0.0)
            else:
                precisions.append(tp / (tp + fp))
        return np.mean(precisions)
        
    if pos_label is None:
        pos_label = np.unique(y_true)[1] if len(np.unique(y_true)) > 1 else y_true[0]
        
    # True Positives: y_true == pos_label AND y_pred == pos_label
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    # False Positives: y_true != pos_label AND y_pred == pos_label
    fp = np.sum((y_true != pos_label) & (y_pred == pos_label))
    
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)

def recall_score(y_true, y_pred, pos_label=None, average='binary'):
    """Calculate the recall."""
    if average == 'macro':
        labels = np.unique(np.concatenate([y_true, y_pred]))
        recalls = []
        for label in labels:
            tp = np.sum((y_true == label) & (y_pred == label))
            fn = np.sum((y_true == label) & (y_pred != label))
            if tp + fn == 0:
                recalls.append(0.0)
            else:
                recalls.append(tp / (tp + fn))
        return np.mean(recalls)
        
    if pos_label is None:
        pos_label = np.unique(y_true)[1] if len(np.unique(y_true)) > 1 else y_true[0]
        
    # True Positives: y_true == pos_label AND y_pred == pos_label
    tp = np.sum((y_true == pos_label) & (y_pred == pos_label))
    # False Negatives: y_true == pos_label AND y_pred != pos_label
    fn = np.sum((y_true == pos_label) & (y_pred != pos_label))
    
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)

def f1_score(y_true, y_pred, pos_label=None, average='binary'):
    """Calculate the F1-score."""
    if average == 'macro':
        labels = np.unique(np.concatenate([y_true, y_pred]))
        f1_scores = []
        for label in labels:
            tp = np.sum((y_true == label) & (y_pred == label))
            fp = np.sum((y_true != label) & (y_pred == label))
            fn = np.sum((y_true == label) & (y_pred != label))
            
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            if p + r == 0:
                f1_scores.append(0.0)
            else:
                f1_scores.append(2 * (p * r) / (p + r))
        return np.mean(f1_scores)
        
    precision = precision_score(y_true, y_pred, pos_label)
    recall = recall_score(y_true, y_pred, pos_label)
    
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def confusion_matrix(y_true, y_pred, labels=None):
    """ Calculate the confusion matrix."""
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
        
    n_classes = len(labels)
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    
    # Map labels to indices
    label_to_index = {label: i for i, label in enumerate(labels)}
    
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = label_to_index[true_label]
        pred_idx = label_to_index[pred_label]
        matrix[true_idx, pred_idx] += 1
        
    return matrix

def per_class_accuracy(y_true, y_pred, labels=None):
    """Calculate the accuracy (recall) for each individual class."""
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
        
    accuracies = {}
    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        if tp + fn == 0:
            accuracies[label] = 0.0
        else:
            accuracies[label] = tp / (tp + fn)
            
    return accuracies

def classification_report(y_true, y_pred, labels=None):
    """Build a text report showing the main classification metrics."""
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    
    report = f"{'precision':>12} {'recall':>12} {'f1-score':>12} {'support':>12}\n\n"
    
    for label in labels:
        p = precision_score(y_true, y_pred, pos_label=label)
        r = recall_score(y_true, y_pred, pos_label=label)
        f1 = f1_score(y_true, y_pred, pos_label=label)
        support = np.sum(y_true == label)
        
        report += f"{str(label):>12} {p:12.4f} {r:12.4f} {f1:12.4f} {support:12d}\n"
        
    report += "\n"
    accuracy = accuracy_score(y_true, y_pred)
    report += f"{'accuracy':>12} {'':>12} {'':>12} {accuracy:12.4f} {len(y_true):12d}\n"
    
    macro_p = precision_score(y_true, y_pred, average='macro')
    macro_r = recall_score(y_true, y_pred, average='macro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    report += f"{'macro avg':>12} {macro_p:12.4f} {macro_r:12.4f} {macro_f1:12.4f} {len(y_true):12d}\n"
    
    return report
