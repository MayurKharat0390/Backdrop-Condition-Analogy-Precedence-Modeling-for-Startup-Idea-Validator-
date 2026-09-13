"""
calibration.py: Probability Calibration, Reliability Analysis, and Brier Score Evaluation.

Why Calibration is Essential:
The Economic Decision Layer directly uses predicted probabilities p_i in:
    EMV = p_i * V - I > 0
If a model predicts p = 0.10, the empirical success rate must truly be ~10%.
Uncalibrated, overconfident models distort the breakeven threshold p* = I / V.

Supported Methods:
1. Brier Score Loss: Mean squared error between predicted probabilities and binary outcomes.
2. Expected Calibration Error (ECE): Bin-weighted absolute difference between confidence and accuracy.
3. Reliability Curves: 10-bin calibration diagrams.
4. Platt Scaling (Sigmoid) & Isotonic Regression calibration.
"""

from typing import Dict, Tuple, Union
import numpy as np
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve, CalibratedClassifierCV


def calculate_expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Calculate Expected Calibration Error (ECE) across n_bins equal-width confidence intervals.
    
    ECE = sum_{b=1}^B ( |B_b| / N ) * | acc(B_b) - conf(B_b) |
    """
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    total_samples = len(y_true)
    
    for b in range(n_bins):
        mask = (bin_indices == b)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_count / total_samples) * np.abs(bin_acc - bin_conf)
            
    return float(ece)


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Evaluate calibration metrics for a set of predicted probabilities.
    
    Returns:
        Dictionary containing Brier score, ECE, fraction_of_positives, and mean_predicted_value.
    """
    brier = brier_score_loss(y_true, y_prob)
    ece = calculate_expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    
    return {
        "brier_score": float(brier),
        "ece": float(ece),
        "prob_true": prob_true,
        "prob_pred": prob_pred
    }


def fit_calibrator(
    base_estimator,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "sigmoid"
) -> CalibratedClassifierCV:
    """
    Calibrate a pre-fitted estimator on an independent validation set.
    
    Parameters:
        base_estimator: Pre-trained scikit-learn compatible classifier.
        X_val: Feature matrix of validation set (separate from test set).
        y_val: Target vector of validation set.
        method: 'sigmoid' (Platt scaling) or 'isotonic' (non-parametric monotonic).
        
    Returns:
        calibrated_model: Fitted CalibratedClassifierCV wrapper.
    """
    calibrated = CalibratedClassifierCV(estimator=base_estimator, method=method, cv="prefit")
    calibrated.fit(X_val, y_val)
    return calibrated
