import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()   
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.mnist_features import load_mnist, split_data
from src.models.logistic_softmax import SoftmaxClassifier
from src.utils.bias_variance_analysis import learning_curve, complexity_curve, summarize_bias_variance

# Load MNIST data
x_all, y_all = load_mnist('data/mnist.npz')

# Split data into train/val/test
x_train, y_train, x_val, y_val, x_test, y_test = split_data(
    x_all, y_all, test_size=0.2, val_size=0.1, random_state=42
)

print(f"Training set size: {x_train.shape}")
print(f"Validation set size: {x_val.shape}")
print(f"Test set size: {x_test.shape}")

# ═══════════════════════════════════════════════════════════════════
# LEARNING CURVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("LEARNING CURVE ANALYSIS")
print("="*70)

def model_factory():
    """Factory function to create fresh model instances."""
    return SoftmaxClassifier(learning_rate=0.1, lambda_reg=1e-5, batch_size=32)

# Run learning curve
learning_results = learning_curve(
    model_factory=model_factory,
    X_train=x_train,
    y_train=y_train,
    X_val=x_val,
    y_val=y_val,
    train_sizes=(0.1, 0.3, 0.5, 0.7, 1.0),
    seeds=(42,),
    model_name="SoftmaxClassifier"
)

print("\nLearning Curve Results:")
print(learning_results)

# Diagnose bias/variance
learning_diagnosis = summarize_bias_variance(
    learning_results,
    target_score=0.85,
    gap_threshold=0.10
)
print("\nBias-Variance Diagnosis (Learning Curve):")
print(learning_diagnosis[['train_size', 'train_score', 'val_score', 'gap', 'diagnosis']])

# ═══════════════════════════════════════════════════════════════════
# COMPLEXITY CURVE ANALYSIS
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("COMPLEXITY CURVE ANALYSIS (Learning Rate)")
print("="*70)

param_grid = [
    {'learning_rate': 0.01},
    {'learning_rate': 0.05},
    {'learning_rate': 0.1},
    {'learning_rate': 0.2},
    {'learning_rate': 0.5},
]

def model_factory_lr(**kwargs):
    """Factory function for complexity curve with learning rate tuning."""
    lr = kwargs.get('learning_rate', 0.1)
    return SoftmaxClassifier(learning_rate=lr, lambda_reg=1e-5, batch_size=32)

complexity_results = complexity_curve(
    model_factory=model_factory_lr,
    param_grid=param_grid,
    X_train=x_train,
    y_train=y_train,
    X_val=x_val,
    y_val=y_val,
    model_name="SoftmaxClassifier"
)

print("\nComplexity Curve Results:")
print(complexity_results)

# Diagnose bias/variance
complexity_diagnosis = summarize_bias_variance(
    complexity_results,
    target_score=0.85,
    gap_threshold=0.10
)
print("\nBias-Variance Diagnosis (Complexity Curve):")
print(complexity_diagnosis[['learning_rate', 'train_score', 'val_score', 'gap', 'diagnosis']])

# ═══════════════════════════════════════════════════════════════════
# FINAL MODEL TRAINING AND EVALUATION
# ═══════════════════════════════════════════════════════════════════

print("\n" + "="*70)
print("FINAL MODEL TRAINING AND EVALUATION")
print("="*70)

print("\nTraining final SoftmaxClassifier on full training set...")
final_model = SoftmaxClassifier(learning_rate=0.1, lambda_reg=1e-5, batch_size=32)
final_model.fit(x_train, y_train)

# Make predictions on test set
print("\nEvaluating on test set...")
preds = final_model.predict(x_test)
proba = final_model.predict_proba(x_test)
metrics = final_model.evaluate(x_test, y_test)

print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
print(f"Full Metrics: {metrics}")