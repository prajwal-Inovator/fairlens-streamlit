

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
    false_positive_rate,
    false_negative_rate
)
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing functions from bias_detector
from core.bias_detector import load_dataset, preprocess

# ------------------------------------------------------------------
# HELPER: _compute_group_metrics
# ------------------------------------------------------------------
def _compute_group_metrics(y_true, y_pred, sensitive) -> list:
    """
    Compute per-group metrics (selection rate, FPR, FNR, accuracy).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        sensitive: Sensitive attribute array
        
    Returns:
        List of dicts: [{"group": str, "selection_rate": float, ...}]
    """
    metric_frame = MetricFrame(
        metrics={
            'selection_rate': selection_rate,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'accuracy': accuracy_score
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive
    )
    
    groups = [str(g) for g in np.unique(sensitive)]
    by_group = []
    for i, group in enumerate(groups):
        by_group.append({
            'group': group,
            'selection_rate': float(metric_frame.by_group['selection_rate'].iloc[i]),
            'false_positive_rate': float(metric_frame.by_group['false_positive_rate'].iloc[i]),
            'false_negative_rate': float(metric_frame.by_group['false_negative_rate'].iloc[i]),
            'accuracy': float(metric_frame.by_group['accuracy'].iloc[i])
        })
    return by_group

# ------------------------------------------------------------------
# HELPER: _train_fair_model
# ------------------------------------------------------------------
def _train_fair_model(
    X_train, y_train, sensitive_train,
    method: str = 'ExponentiatedGradient',
    constraint: str = 'DemographicParity'
):
    """
    Train a fair model using Fairlearn reduction algorithms.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sensitive_train: Sensitive attribute for training
        method: 'ExponentiatedGradient' or 'GridSearch'
        constraint: 'DemographicParity' or 'EqualizedOdds'
        
    Returns:
        Trained fair model (sklearn-compatible)
    """
    # Base estimator (Logistic Regression)
    estimator = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    
    # Choose constraint
    if constraint == 'DemographicParity':
        constraint_obj = DemographicParity()
    elif constraint == 'EqualizedOdds':
        constraint_obj = EqualizedOdds()
    else:
        raise ValueError(f"Unknown constraint: {constraint}")
    
    # Choose reduction method
    if method == 'ExponentiatedGradient':
        fair_model = ExponentiatedGradient(
            estimator=estimator,
            constraints=constraint_obj,
            eps=0.01,           # Small epsilon for tight constraint
            max_iter=50
        )
    elif method == 'GridSearch':
        from fairlearn.reductions import GridSearch
        fair_model = GridSearch(
            estimator=estimator,
            constraints=constraint_obj,
            grid_size=10,
            grid_limit=5
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"[✓] Training fair model with {method} and {constraint}...")
    fair_model.fit(X_train, y_train, sensitive_features=sensitive_train)
    print("[✓] Fair model training complete")
    return fair_model

# ------------------------------------------------------------------
# MAIN FUNCTION: mitigate_bias
# ------------------------------------------------------------------
def mitigate_bias(
    file_path: str,
    target_col: str,
    sensitive_col: str,
    method: str = 'ExponentiatedGradient',
    constraint: str = 'DemographicParity'
) -> dict:
    """
    Apply fairness mitigation to a dataset and return before/after metrics.
    
    Args:
        file_path: Path to CSV file
        target_col: Target column name (binary)
        sensitive_col: Sensitive attribute column name
        method: 'ExponentiatedGradient' or 'GridSearch'
        constraint: 'DemographicParity' or 'EqualizedOdds'
        
    Returns:
        Dict containing original and fair metrics, improvements, and summary.
    """
    print("\n" + "="*60)
    print("FairLens Fairness Mitigation - Starting")
    print("="*60)
    
    # Step 1: Load and preprocess data
    df = load_dataset(file_path)
    X, y, sensitive = preprocess(df, target_col, sensitive_col)
    
    # Step 2: Train/test split (80/20)
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X, y, sensitive, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[✓] Train/test split: {X_train.shape[0]} train, {X_test.shape[0]} test")
    
    # Step 3: Train original (unfair) model
    original_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    original_model.fit(X_train, y_train)
    y_pred_orig = original_model.predict(X_test)
    
    # Original metrics
    orig_accuracy = accuracy_score(y_test, y_pred_orig)
    orig_dp_diff = demographic_parity_difference(y_test, y_pred_orig, sensitive_features=sensitive_test)
    orig_eo_diff = equalized_odds_difference(y_test, y_pred_orig, sensitive_features=sensitive_test)
    orig_by_group = _compute_group_metrics(y_test, y_pred_orig, sensitive_test)
    
    print(f"[✓] Original model: Acc={orig_accuracy:.4f}, DP diff={orig_dp_diff:.4f}, EO diff={orig_eo_diff:.4f}")
    
    # Step 4: Train fair model
    fair_model = _train_fair_model(X_train, y_train, sensitive_train, method, constraint)
    y_pred_fair = fair_model.predict(X_test)
    
    # Fair model metrics
    fair_accuracy = accuracy_score(y_test, y_pred_fair)
    fair_dp_diff = demographic_parity_difference(y_test, y_pred_fair, sensitive_features=sensitive_test)
    fair_eo_diff = equalized_odds_difference(y_test, y_pred_fair, sensitive_features=sensitive_test)
    fair_by_group = _compute_group_metrics(y_test, y_pred_fair, sensitive_test)
    
    print(f"[✓] Fair model: Acc={fair_accuracy:.4f}, DP diff={fair_dp_diff:.4f}, EO diff={fair_eo_diff:.4f}")
    
    # Step 5: Calculate improvements
    # Improvement in absolute difference (lower is better)
    improvement_dp = (abs(orig_dp_diff) - abs(fair_dp_diff)) / max(abs(orig_dp_diff), 1e-6) * 100
    improvement_eo = (abs(orig_eo_diff) - abs(fair_eo_diff)) / max(abs(orig_eo_diff), 1e-6) * 100
    improvement_dp = max(0, min(100, improvement_dp))  # Clamp 0-100
    improvement_eo = max(0, min(100, improvement_eo))
    
    # Accuracy trade-off
    acc_change = (fair_accuracy - orig_accuracy) * 100
    
    # Step 6: Generate summary
    if improvement_dp > 50 or improvement_eo > 50:
        quality = "significant"
    elif improvement_dp > 20 or improvement_eo > 20:
        quality = "moderate"
    else:
        quality = "minor"
    
    summary = (f"Fairness mitigation using {method} with {constraint} improved demographic parity "
               f"by {improvement_dp:.1f}% and equalized odds by {improvement_eo:.1f}% with an "
               f"accuracy change of {acc_change:+.1f}%. This represents a {quality} fairness improvement.")
    
    # Step 7: Build result dictionary
    result = {
        'original_accuracy': float(orig_accuracy),
        'fair_accuracy': float(fair_accuracy),
        'original_dp_diff': float(orig_dp_diff),
        'fair_dp_diff': float(fair_dp_diff),
        'original_eo_diff': float(orig_eo_diff),
        'fair_eo_diff': float(fair_eo_diff),
        'improvement_dp': float(improvement_dp),
        'improvement_eo': float(improvement_eo),
        'original_by_group': orig_by_group,
        'fair_by_group': fair_by_group,
        'method_used': method,
        'constraint_used': constraint,
        'summary': summary
    }
    
    print("\n" + "="*60)
    print(f"Mitigation Complete. DP improvement: {improvement_dp:.1f}%, EO improvement: {improvement_eo:.1f}%")
    print("="*60 + "\n")
    
    return result

# ------------------------------------------------------------------
# ALTERNATIVE: mitigate_with_gridsearch
# ------------------------------------------------------------------
def mitigate_with_gridsearch(file_path: str, target_col: str, sensitive_col: str) -> dict:
    """
    Convenience wrapper for GridSearch mitigation.
    
    Args:
        file_path: Path to CSV
        target_col: Target column
        sensitive_col: Sensitive column
        
    Returns:
        Same as mitigate_bias with GridSearch method
    """
    return mitigate_bias(file_path, target_col, sensitive_col, method='GridSearch', constraint='DemographicParity')

# ------------------------------------------------------------------
# QUICK TEST (if run as standalone)
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Test with synthetic data
    from sklearn.datasets import make_classification
    
    print("Testing fair_model module with synthetic data...")
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    # Create synthetic sensitive attribute (binary)
    sensitive = np.random.randint(0, 2, size=len(y))
    
    # Save to temp CSV
    import tempfile
    df = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    df['target'] = y
    df['sensitive'] = sensitive
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    result = mitigate_bias(temp_file.name, 'target', 'sensitive')
    print("Result keys:", result.keys())
    print(f"Improvement DP: {result['improvement_dp']:.1f}%")
    print(f"Improvement EO: {result['improvement_eo']:.1f}%")
    
    # Cleanup
    import os
    os.unlink(temp_file.name)